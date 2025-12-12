import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pickle
import os
from params_config import Config
from soc_manager import SOCManager

config = Config()


class RegionManager:
    """区域管理器 - 处理区域相关操作"""

    def __init__(self, region_file: str):
        self.regions_gdf = gpd.read_file(region_file)
        self.region_centers = self._compute_region_centers()

    def _compute_region_centers(self) -> Dict[int, Tuple[float, float]]:
        """计算每个区域的中心点"""
        centers = {}
        for idx, row in self.regions_gdf.iterrows():
            centroid = row.geometry.centroid
            centers[idx] = (centroid.x, centroid.y)
        return centers

    def get_region_id(self, lat: float, lon: float) -> Optional[int]:
        """根据坐标获取区域ID"""
        point = Point(lon, lat)
        for idx, row in self.regions_gdf.iterrows():
            if row.geometry.contains(point):
                return idx
        return None

    def get_region_center(self, region_id: int) -> Tuple[float, float]:
        """获取区域中心坐标"""
        return self.region_centers.get(region_id, (0.0, 0.0))

    def get_region_polygon(self, region_id: int) -> Optional[Polygon]:
        """获取区域多边形"""
        if region_id < len(self.regions_gdf):
            return self.regions_gdf.iloc[region_id].geometry
        return None

    def generate_random_points_in_region(self, region_id: int, n_points: int) -> List[Tuple[float, float]]:
        """在区域内生成随机点"""
        polygon = self.get_region_polygon(region_id)
        if polygon is None:
            center = self.get_region_center(region_id)
            return [center] * n_points

        minx, miny, maxx, maxy = polygon.bounds
        points = []

        attempts = 0
        max_attempts = n_points * 100

        while len(points) < n_points and attempts < max_attempts:
            random_point = Point(
                np.random.uniform(minx, maxx),
                np.random.uniform(miny, maxy)
            )
            if polygon.contains(random_point):
                points.append((random_point.x, random_point.y))
            attempts += 1

        # 如果未能生成足够的点,用区域中心填充
        while len(points) < n_points:
            center = self.get_region_center(region_id)
            points.append(center)

        return points


class DispatchPointManager:
    """调度点管理器"""

    def __init__(self, dispatch_file: str, region_manager: RegionManager):
        self.dispatch_df = pd.read_csv(dispatch_file)
        self.region_manager = region_manager
        self._assign_regions()

    def _assign_regions(self):
        """为调度点分配区域ID"""
        self.dispatch_df['region_id'] = self.dispatch_df.apply(
            lambda row: self.region_manager.get_region_id(
                row['latitude'],
                row['longitude']
            ),
            axis=1
        )

    def get_dispatch_points_in_region(self, region_id: int) -> List[Dict]:
        """获取区域内的调度点"""
        region_points = self.dispatch_df[
            self.dispatch_df['region_id'] == region_id
            ]
        return region_points.to_dict('records')

    def find_nearest_dispatch_point(self, lat: float, lon: float, region_id: int) -> Optional[Dict]:
        """找到最近的调度点"""
        from distance import haversine_distance

        region_points = self.get_dispatch_points_in_region(region_id)
        if not region_points:
            return None

        min_distance = float('inf')
        nearest_point = None

        for point in region_points:
            distance = haversine_distance(
                lat, lon,
                point['latitude'], point['longitude']
            )
            if distance < min_distance:
                min_distance = distance
                nearest_point = point

        return nearest_point


class DataLoader:
    """独立的数据加载器 - 每个edge_env使用独立实例"""

    def __init__(self,
                 trajectory_file: str,
                 region_file: str,
                 dispatch_file: str,
                 soc_manager: SOCManager,
                 region_manager: RegionManager,
                 dispatch_manager: DispatchPointManager):
        self.trajectory_file = trajectory_file
        self.soc_manager = soc_manager
        self.region_manager = region_manager
        self.dispatch_manager = dispatch_manager

        # 加载轨迹数据
        self.trajectory_df = pd.read_csv(trajectory_file)
        self.trajectory_df['timestamp'] = pd.to_datetime(
            self.trajectory_df['timestamp']
        )
        self.trajectory_df = self.trajectory_df.sort_values(
            ['id', 'timestamp']
        ).reset_index(drop=True)

        # 当前时间索引(每个实例独立)
        self.current_time_index = 0
        self.timestamps = sorted(self.trajectory_df['timestamp'].unique())

        # EV状态(每个实例独立)
        self.ev_states: Dict[int, Dict] = {}
        self._initialize_ev_states()

    def _initialize_ev_states(self):
        """初始化EV状态"""
        first_time = self.timestamps[0]
        first_data = self.trajectory_df[
            self.trajectory_df['timestamp'] == first_time
            ]

        for _, row in first_data.iterrows():
            ev_id = int(row['id'])
            lat, lon = row['latitude'], row['longitude']
            region_id = self.region_manager.get_region_id(lat, lon)

            # 从SOC管理器获取初始电量
            initial_charge = self.soc_manager.get_initial_soc(ev_id)

            self.ev_states[ev_id] = {
                'current_location': (lat, lon),
                'current_charge': initial_charge,
                'region_id': region_id,
                'status': 'running',
                'trajectory_index': 0,
                'charging_request_time': None,
                'target_charging_location': None,
                'assigned_mcs': None,
                'assigned_fcs': None,
                'chose_fcs': False,
                'charging_start_time': None
            }

    def get_current_time(self) -> datetime:
        """获取当前时间"""
        if self.current_time_index < len(self.timestamps):
            return self.timestamps[self.current_time_index]
        return self.timestamps[-1]

    def get_evs_in_region(self, region_id: int) -> Dict[int, Dict]:
        """获取区域内的EV"""
        return {
            ev_id: state for ev_id, state in self.ev_states.items()
            if state.get('region_id') == region_id
        }

    def update_ev_state(self, ev_id: int, updates: Dict):
        """更新EV状态"""
        if ev_id in self.ev_states:
            self.ev_states[ev_id].update(updates)

    def update_ev_movement(self):
        """更新EV移动状态"""
        from distance import haversine_distance

        # 移动到下一个时间步
        self.current_time_index += 1
        if self.current_time_index >= len(self.timestamps):
            return

        current_time = self.timestamps[self.current_time_index]
        current_data = self.trajectory_df[
            self.trajectory_df['timestamp'] == current_time
            ]

        # 更新每个EV的位置和电量
        for _, row in current_data.iterrows():
            ev_id = int(row['id'])
            if ev_id not in self.ev_states:
                continue

            state = self.ev_states[ev_id]

            # 跳过正在充电的EV
            if state['status'] in ['charging', 'waiting']:
                continue

            # 更新位置
            old_location = state['current_location']
            new_location = (row['latitude'], row['longitude'])

            # 计算距离和消耗电量
            if old_location:
                distance = haversine_distance(
                    old_location[0], old_location[1],
                    new_location[0], new_location[1]
                )
                energy_consumed = distance * config.ENERGY_CONSUMPTION
                new_charge = max(0, state['current_charge'] - energy_consumed)
                state['current_charge'] = new_charge

            # 更新位置和区域
            state['current_location'] = new_location
            state['region_id'] = self.region_manager.get_region_id(
                new_location[0], new_location[1]
            )

    def reset_ev_states(self):
        """重置EV状态"""
        self.current_time_index = 0
        self.ev_states.clear()
        self._initialize_ev_states()

    # 委托给管理器的方法
    def get_region_id(self, lat: float, lon: float) -> Optional[int]:
        return self.region_manager.get_region_id(lat, lon)

    def get_region_center(self, region_id: int) -> Tuple[float, float]:
        return self.region_manager.get_region_center(region_id)

    def get_ev_state(self) -> Optional[Dict]:
        return self.ev_states

    def generate_random_points_in_region(self, region_id: int, n_points: int) -> List[Tuple[float, float]]:
        return self.region_manager.generate_random_points_in_region(region_id, n_points)

    def get_dispatch_points_in_region(self, region_id: int) -> List[Dict]:
        return self.dispatch_manager.get_dispatch_points_in_region(region_id)

    def find_nearest_dispatch_point(self, lat: float, lon: float, region_id: int) -> Optional[Dict]:
        return self.dispatch_manager.find_nearest_dispatch_point(lat, lon, region_id)


class DataLoaderFactory:
    """数据加载器工厂 - 创建独立的DataLoader实例"""

    def __init__(self,
                 trajectory_file: str,
                 region_file: str,
                 dispatch_file: str):
        """
        初始化工厂

        Args:
            trajectory_file: 轨迹文件路径
            region_file: 区域文件路径
            dispatch_file: 调度点文件路径
        """
        print("初始化数据加载器工厂...")

        # 创建共享的管理器(这些是只读的,可以共享)
        print("  加载SOC表...")
        self.soc_manager = SOCManager(trajectory_file, config)

        print("  加载区域数据...")
        self.region_manager = RegionManager(region_file)

        print("  加载调度点数据...")
        self.dispatch_manager = DispatchPointManager(
            dispatch_file,
            self.region_manager
        )

        # 保存文件路径
        self.trajectory_file = trajectory_file
        self.region_file = region_file
        self.dispatch_file = dispatch_file

        print("数据加载器工厂初始化完成!")

    def create_dataloader(self) -> DataLoader:
        """创建新的独立DataLoader实例"""
        return DataLoader(
            self.trajectory_file,
            self.region_file,
            self.dispatch_file,
            self.soc_manager,
            self.region_manager,
            self.dispatch_manager
        )


# 使用示例
if __name__ == '__main__':
    # 创建工厂
    factory = DataLoaderFactory(
        trajectory_file='dataset/top1000evs/reallocated/20140818_processed.csv',
        region_file='dataset/fcs_voronoi_regions.geojson',
        dispatch_file='dataset/dispatch_points_400.csv'
    )