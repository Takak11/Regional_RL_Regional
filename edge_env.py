import gym
import numpy as np
import random

import torch
from gym import spaces
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import timedelta
from dataloader import DataLoaderFactory
from params_config import Config
from charging_entities import MCS, MCSStatus, FCS, ChargingRequest, ChargingPile
from distance import haversine_distance

config = Config()


@dataclass
class PointFeatureResult:
    """调度点特征提取的结果容器。"""

    reachable_indices: List[int]


class MCSMatcher:
    def match_mcs_to_points_topk(
            self,
            mcs_list: List,
            dispatch_points: List[Dict],
            action_scores: np.ndarray,
            np_random,
            k: int = 3
    ) -> Dict[int, int]:
        available_mcs_indices = [
            i for i, mcs in enumerate(mcs_list)
            if mcs.status.value == "idle"
        ]

        if len(available_mcs_indices) == 0:
            return {}

        mcs_reachable_map = {}
        for mcs_idx in available_mcs_indices:
            mcs = mcs_list[mcs_idx]
            reachable = self._get_reachable_points(mcs, dispatch_points)
            mcs_reachable_map[mcs_idx] = reachable

        matching = {}
        used_points = set()

        # 随机顺序处理MCS
        mcs_order = np_random.permutation(available_mcs_indices).tolist()

        for mcs_idx in mcs_order:
            mcs = mcs_list[mcs_idx]
            reachable = mcs_reachable_map[mcs_idx]

            if not reachable:
                continue

            available_points = [p for p in reachable if p not in used_points]
            if not available_points:
                continue

            # 获取这些点的scores
            point_scores = [(p, action_scores[p]) for p in available_points]

            # 按score排序，选择top-k
            point_scores.sort(key=lambda x: x[1], reverse=True)
            top_k_points = point_scores[:min(k, len(point_scores))]

            if not top_k_points:
                continue

            selected_point = top_k_points[0][0]
            matching[mcs_idx] = selected_point
            used_points.add(selected_point)

        return matching

    def _get_reachable_points(self, mcs, dispatch_points: List[Dict]) -> List[int]:
        """获取MCS可达的调度点索引"""
        reachable = []
        mcs_lat, mcs_lon = mcs.current_location[1], mcs.current_location[0]

        for idx, point in enumerate(dispatch_points):
            distance = haversine_distance(
                mcs_lat, mcs_lon,
                point['latitude'], point['longitude']
            )
            if distance <= config.MCS_SCHEDULE_R:
                reachable.append(idx)

        return reachable


def validate_action_scores(action_scores: np.ndarray) -> np.ndarray:
    """验证并修复action scores中的异常值"""
    # 检查NaN
    if np.any(np.isnan(action_scores)):
        print("Warning: NaN detected in action scores, replacing with zeros")
        action_scores = np.nan_to_num(action_scores, nan=0.0)

    # 检查Inf
    if np.any(np.isinf(action_scores)):
        print("Warning: Inf detected in action scores, clipping")
        action_scores = np.clip(action_scores, -10, 10)

    return action_scores


class EdgeEnv(gym.Env):

    def __init__(self,
                 region_id: int,
                 factory: DataLoaderFactory,
                 max_steps: int = 100):
        super(EdgeEnv, self).__init__()

        self.np_random = None

        self.region_id = region_id

        self.data_loader = factory.create_dataloader()

        self.max_steps = max_steps
        self.num_mcs = config.MCS_PER_REGION

        # 调度点
        self.dispatch_points = self.data_loader.get_dispatch_points_in_region(region_id)
        self.num_dispatch_points = len(self.dispatch_points)
        self._last_served = 0
        self._last_failed = 0
        self.matcher = MCSMatcher()

        # 区域边界用于请求热力图
        self.region_bounds = self._get_region_bounds()

        # 观察空间: 在第一次 reset 后根据实际状态长度确定
        self.grid_size = 10  # 网格大小
        self.observation_space = None

        # 初始化时间
        dataset_start = self.data_loader.get_current_time()
        self.start_time = dataset_start.to_pydatetime() if hasattr(dataset_start, 'to_pydatetime') else dataset_start
        self.current_step = 0
        self.current_time = self.start_time

        # 初始化FCS和MCS
        self.fcs = self._initialize_fcs()
        self.mcs_list = self._initialize_mcs()

        # 请求管理
        self.pending_requests = []
        self.active_requests = {}  # {ev_id: ChargingRequest}
        self.served_ev_ids = set()

        # 统计信息 - 只关注核心指标
        self.episode_stats = {
            'served_requests': 0,
            'failed_requests': 0,
            'failed_requests_fcs': 0,
            'failed_requests_mcs': 0,
            'timeout_requests': 0,
            'total_wait_time': 0.0,
            'wait_time_count': 0,
            'mcs_served': 0,
            'fcs_served': 0
        }

        # 动作空间: 为每个调度点打分
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_dispatch_points,),
            dtype=np.float32
        )
        self.training_progress = 0.0  # 0.0 到 1.0

        self.reset()

        # 在 reset 后补充 observation_space（依赖初始化后的状态长度）
        if self.observation_space is None:
            initial_obs = self._get_obs()
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=initial_obs.shape,
                dtype=np.float32
            )

    def seed(self, seed=config.random_seed):
        """与 Gym 兼容的种子函数"""
        if seed is None:
            seed = np.random.randint(0, 2 ** 32 - 1)
        self._set_random_seed(seed)
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def _set_random_seed(self, seed: int):
        """统一设置环境随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 关闭cuDNN的非确定性算法（保证完全复现）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _initialize_fcs(self) -> FCS:
        """初始化固定充电站"""
        region_center = self.data_loader.get_region_center(self.region_id)
        return FCS(
            id=self.region_id,
            location=region_center,
            region_id=self.region_id
        )

    def _initialize_mcs(self) -> List[MCS]:
        """初始化移动充电站"""
        mcs_list = []
        point_list = self.data_loader.generate_random_points_in_region(self.region_id, self.num_mcs)
        for i in range(self.num_mcs):
            mcs = MCS(
                id=f"mcs_{self.region_id}_{i}",
                current_location=point_list[i],
                region_id=self.region_id
            )
            mcs_list.append(mcs)
        return mcs_list

    def _get_region_bounds(self) -> Tuple[float, float, float, float]:
        """获取区域边界(最小经度、最小纬度、最大经度、最大纬度)"""
        polygon = self.data_loader.region_manager.get_region_polygon(self.region_id)
        if polygon is not None:
            minx, miny, maxx, maxy = polygon.bounds
            return minx, miny, maxx, maxy

        # 无法获取多边形时，基于调度点或区域中心兜底
        if self.dispatch_points:
            lons = [p['longitude'] for p in self.dispatch_points]
            lats = [p['latitude'] for p in self.dispatch_points]
            return min(lons), min(lats), max(lons), max(lats)

        center_lon, center_lat = self.data_loader.get_region_center(self.region_id)
        delta = 0.01
        return center_lon - delta, center_lat - delta, center_lon + delta, center_lat + delta

    def _get_obs(self) -> np.ndarray:
        """获取简化的观察状态"""
        obs_parts = []

        mcs_part = []
        # MCS状态 (num_mcs * 3) 位置（经、纬度）、空闲状态
        for mcs in self.mcs_list:
            mcs_state = [
                mcs.current_location[0],
                mcs.current_location[1],
                1.0 if mcs.status == MCSStatus.IDLE else 0.0  # 是否空闲
            ]
            mcs_part.extend(mcs_state)
        num_idle_mcs = sum(1 for mcs in self.mcs_list if mcs.status == MCSStatus.IDLE)
        mcs_part.append(num_idle_mcs / self.num_mcs)

        obs_parts.extend(mcs_part)

        # FCS相关状态 可用桩数、队列长队、预计排队时间
        fcs_available_piles = len(self.fcs.get_available_piles())
        fcs_queuing_length = self.fcs.get_queue_length()
        avg_fcs_wait = self.fcs.get_estimated_wait_time(self.current_time)
        obs_parts.extend([fcs_available_piles / len(self.fcs.charging_piles),
                          fcs_queuing_length / config.EXPECTED_MAX_QUEUING_LENGTH,
                          avg_fcs_wait / config.EXPECTED_MAX_FCS_WAIT_TIME])

        # 请求热力图 (grid_size x grid_size)
        heatmap = self._build_request_heatmap()
        obs_parts.extend(heatmap)

        # 区域级负载与需求状态 未服务请求数 平均等待时间
        num_pending = len(self.pending_requests)

        # 当前平均等待时间
        if num_pending > 0:
            current_avg_wait = np.mean([
                (self.current_time - req.request_time).total_seconds() / 60
                for req in self.pending_requests
            ])
        else:
            current_avg_wait = 0.0

        # 历史平均等待时间
        if self.episode_stats['wait_time_count'] > 0:
            historical_avg_wait = self.episode_stats['total_wait_time'] / self.episode_stats['wait_time_count']
        else:
            historical_avg_wait = 0.0

        global_stats = [
            num_pending / 20.0,  # 归一化待处理请求数
            num_idle_mcs / self.num_mcs,  # 空闲MCS比例
            fcs_available_piles / config.FCS_CHARGING_PILES,  # FCS可用桩比例
            min(current_avg_wait / 30.0, 1.0),  # 当前平均等待时间(上限30分钟)
            min(historical_avg_wait / 30.0, 1.0)  # 历史平均等待时间(上限30分钟)
        ]

        obs_parts.extend(global_stats)

        return np.array(obs_parts, dtype=np.float32)

    def _build_request_heatmap(self) -> List[float]:
        """根据待处理请求生成归一化热力图"""
        heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        if not self.pending_requests:
            return heatmap.flatten().tolist()

        minx, miny, maxx, maxy = self.region_bounds
        width = max(maxx - minx, 1e-6)
        height = max(maxy - miny, 1e-6)

        for req in self.pending_requests:
            lon, lat = req.location
            # 归一化到 [0, grid_size)
            x_norm = (lon - minx) / width
            y_norm = (lat - miny) / height
            x_idx = int(np.clip(x_norm * self.grid_size, 0, self.grid_size - 1))
            y_idx = int(np.clip(y_norm * self.grid_size, 0, self.grid_size - 1))
            heatmap[y_idx, x_idx] += 1

        # 归一化到 [0,1]
        heatmap = heatmap / np.max(heatmap)
        return heatmap.flatten().tolist()

    def _extract_point_features(self) -> PointFeatureResult:
        """提取当前可调度的调度点索引，用于构建动作可达掩码。"""

        reachable_indices = set()

        for mcs in self.mcs_list:
            if mcs.status == MCSStatus.IDLE:
                reachable = self.matcher._get_reachable_points(mcs, self.dispatch_points)
                reachable_indices.update(reachable)

        return PointFeatureResult(reachable_indices=sorted(reachable_indices))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步环境交互"""
        action = validate_action_scores(action)
        # 1. 执行MCS调度
        matching = self.matcher.match_mcs_to_points_topk(
            self.mcs_list,
            self.dispatch_points,
            action,
            self.np_random
        )
        for mcs_idx, point_idx in matching.items():
            mcs = self.mcs_list[mcs_idx]
            point = self.dispatch_points[point_idx]
            target_location = (point['longitude'], point['latitude'])

            distance = haversine_distance(
                mcs.current_location[1], mcs.current_location[0],
                target_location[1], target_location[0]
            )
            travel_time = distance / config.MOVING_SPEED * 60
            arrival_time = self.current_time + timedelta(minutes=travel_time)

            mcs.start_movement(target_location, arrival_time, None)
            mcs.total_distance_traveled += distance

        self._update_mcs_status()

        self._process_charging_requests()

        self._update_fcs_status()

        # 5. 更新时间和EV状态
        self.current_step += 1
        self.current_time += timedelta(minutes=config.TIME_STEP)
        self.data_loader.update_ev_movement()

        # 6. 生成新的充电请求
        self._generate_charging_requests()

        # 7. 计算奖励
        reward = self._calculate_simple_reward()

        # 8. 获取新观察
        obs = self._get_obs()

        # 9. 检查是否结束
        done = self.current_step >= self.max_steps

        # 10. 构建信息字典
        info = {
            'current_step': self.current_step,
            'pending_requests': len(self.pending_requests),
            'episode_stats': self.episode_stats.copy()
        }

        return obs, reward, done, info

    def _update_mcs_status(self):
        """更新MCS状态"""
        for mcs in self.mcs_list:
            # 完成充电
            if mcs.status == MCSStatus.CHARGING and mcs.estimated_finish_time:
                if self.current_time >= mcs.estimated_finish_time:
                    if mcs.assigned_ev:
                        ev_id = int(mcs.assigned_ev)
                        ev_state = self.data_loader.get_ev_state().get(ev_id)

                        if ev_state:
                            new_charge = min(
                                ev_state['current_charge'] + mcs.charge_amount,
                                config.BATTERY_CAPACITY
                            )

                            self.data_loader.update_ev_state(ev_id, {
                                'current_charge': new_charge,
                                'status': 'running',
                                'assigned_mcs': None,
                                'target_charging_location': None,
                                'charging_request_time': None
                            })

                    mcs.complete_charging()

            # 完成移动
            elif mcs.status == MCSStatus.MOVING and mcs.arrival_time:
                if self.current_time >= mcs.arrival_time:
                    mcs.complete_movement()

    def _process_charging_requests(self):
        """处理充电请求"""
        served_requests = []

        for request in self.pending_requests[:]:
            # 检查超时
            wait_time = (self.current_time - request.request_time).total_seconds() / 60
            if wait_time > config.MAX_WAITING_TIME:
                self.pending_requests.remove(request)
                if request.ev_id in self.active_requests:
                    del self.active_requests[request.ev_id]

                self.episode_stats['total_wait_time'] += wait_time
                self.episode_stats['wait_time_count'] += 1

                ev_state = self.data_loader.get_ev_state().get(request.ev_id)
                if ev_state and ev_state.get('chose_fcs'):
                    self.episode_stats['failed_requests_fcs'] += 1
                    # 从FCS队列中移除
                    self.fcs.remove_from_queue(request.ev_id, self.current_time)
                else:
                    self.episode_stats['failed_requests_mcs'] += 1

                self.episode_stats['failed_requests'] += 1
                self.episode_stats['timeout_requests'] += 1

                self.data_loader.update_ev_state(request.ev_id, {
                    'status': 'completed',
                    'charging_request_time': None,
                    'target_charging_location': None,
                    'chose_fcs': False
                })
                continue

            # 检查是否去FCS充电
            ev_state = self.data_loader.get_ev_state().get(request.ev_id)
            if ev_state and ev_state.get('chose_fcs'):
                # EV选择了FCS
                available_piles = self.fcs.get_available_piles()

                if available_piles:
                    # 有可用充电桩，立即开始充电
                    self._start_fcs_charging(request.ev_id, available_piles[0])
                    served_requests.append(request)

                    # 记录等待时间
                    self.episode_stats['total_wait_time'] += wait_time
                    self.episode_stats['wait_time_count'] += 1
                    continue
                else:
                    # 无可用充电桩，检查是否已在队列中
                    in_queue = any(req.ev_id == request.ev_id for req in self.fcs.waiting_queue)
                    if not in_queue:
                        # 加入队列
                        target_charge = config.BATTERY_CAPACITY * config.TARGET_CHARGE_LEVEL
                        charge_needed = max(0, target_charge - ev_state.get('current_charge', 0))

                        self.fcs.add_to_queue(
                            ev_id=request.ev_id,
                            request_time=request.request_time,
                            charge_needed=charge_needed,
                            current_time=self.current_time
                        )
                    continue

            # 尝试匹配MCS
            for mcs in self.mcs_list:
                if mcs.status == MCSStatus.ASSIGNED and not mcs.assigned_ev:
                    distance = haversine_distance(
                        mcs.current_location[1], mcs.current_location[0],
                        request.location[1], request.location[0]
                    )

                    if distance <= config.MCS_SCHEDULE_R:
                        charge_time = request.estimated_charge_needed / config.MCS_CHARGING_POWER * 60
                        finish_time = self.current_time + timedelta(minutes=charge_time)

                        mcs.start_charging(
                            request.estimated_charge_needed,
                            self.current_time,
                            finish_time
                        )
                        mcs.assigned_ev = str(request.ev_id)

                        # 记录等待时间
                        self.episode_stats['total_wait_time'] += wait_time
                        self.episode_stats['wait_time_count'] += 1

                        self.data_loader.update_ev_state(request.ev_id, {
                            'status': 'charging',
                            'assigned_mcs': mcs.id,
                            'charging_start_time': self.current_time
                        })

                        served_requests.append(request)
                        break

        # 批量更新统计
        for request in served_requests:
            self.pending_requests.remove(request)
            if request.ev_id in self.active_requests:
                del self.active_requests[request.ev_id]
            self.served_ev_ids.add(request.ev_id)
            self.episode_stats['served_requests'] += 1

            ev_state = self.data_loader.get_ev_state().get(request.ev_id)
            if ev_state:
                if ev_state.get('assigned_fcs') is not None:
                    self.episode_stats['fcs_served'] += 1
                elif ev_state.get('assigned_mcs') is not None:
                    self.episode_stats['mcs_served'] += 1

    def _update_fcs_status(self):
        """更新FCS状态(使用新的队列管理系统)"""
        # 1. 检查并完成正在充电的桩
        for pile in self.fcs.charging_piles:
            if pile.is_occupied and pile.estimated_finish_time:
                if self.current_time >= pile.estimated_finish_time:
                    if pile.ev_id:
                        ev_id = pile.ev_id
                        ev_state = self.data_loader.get_ev_state().get(ev_id)

                        if ev_state:
                            # 记录等待时间(如果有请求时间)
                            if ev_state.get('charging_request_time'):
                                wait_time = (pile.charging_start_time -
                                             ev_state['charging_request_time']).total_seconds() / 60
                                self.episode_stats['total_wait_time'] += wait_time
                                self.episode_stats['wait_time_count'] += 1

                            # 更新电量
                            new_charge = min(
                                ev_state['current_charge'] + pile.charge_amount,
                                config.BATTERY_CAPACITY
                            )

                            self.data_loader.update_ev_state(ev_id, {
                                'current_charge': new_charge,
                                'status': 'running',
                                'assigned_fcs': None,
                                'target_charging_location': None,
                                'charging_request_time': None,
                                'chose_fcs': False
                            })

                    # 释放充电桩
                    pile_id = pile.pile_id
                    pile.is_occupied = False
                    pile.ev_id = None
                    pile.charging_start_time = None
                    pile.estimated_finish_time = None
                    pile.charge_amount = 0

                    # 通知FCS重新调度队列
                    self.fcs.on_pile_released(pile_id, self.current_time)

        # 2. 处理等待队列 - 使用新的队列管理API
        while True:
            # 获取下一个可以开始充电的EV
            next_ev_info = self.fcs.get_next_from_queue(self.current_time)

            if next_ev_info is None:
                break

            ev_id, pile_id = next_ev_info
            pile = self.fcs.charging_piles[pile_id]

            # 开始充电
            self._start_fcs_charging(ev_id, pile)

    def _start_fcs_charging(self, ev_id: int, pile: ChargingPile):
        """开始FCS充电"""
        ev_state = self.data_loader.get_ev_state().get(ev_id)
        if not ev_state:
            return

        target_charge = config.BATTERY_CAPACITY * config.TARGET_CHARGE_LEVEL
        charge_needed = max(0, target_charge - ev_state.get('current_charge', 0))
        charge_time = charge_needed / config.CHARGING_POWER * 60

        pile.is_occupied = True
        pile.ev_id = ev_id
        pile.charging_start_time = self.current_time
        pile.estimated_finish_time = self.current_time + timedelta(minutes=charge_time)
        pile.charge_amount = charge_needed

        # 更新EV状态
        self.data_loader.update_ev_state(ev_id, {
            'status': 'charging',
            'assigned_fcs': self.fcs.id,
            'charging_start_time': self.current_time,
            'chose_fcs': True  # 保持标记
        })

    def _generate_charging_requests(self):
        """生成新的充电请求"""
        evs_in_region = self.data_loader.get_evs_in_region(self.region_id)

        for ev_id, ev_state in evs_in_region.items():
            if ev_id in self.active_requests or ev_id in self.served_ev_ids:
                continue

            current_charge_pct = ev_state['current_charge'] / config.BATTERY_CAPACITY
            current_location = ev_state.get('current_location')

            if current_location:
                lat, lon = current_location
                ev_region_id = self.data_loader.get_region_id(lat, lon)

                if (current_charge_pct < config.REQUEST_THRESHOLD and
                        ev_state.get('status') == 'running' and
                        ev_region_id == self.region_id):
                    fcs_distance = haversine_distance(
                        lat, lon,
                        self.fcs.location[1], self.fcs.location[0]
                    )
                    energy_needed = fcs_distance * config.ENERGY_CONSUMPTION
                    can_reach_fcs = ev_state['current_charge'] > energy_needed * config.SAFE_REACH_THRESHOLD

                    target_charge = config.BATTERY_CAPACITY * config.TARGET_CHARGE_LEVEL
                    charge_needed = max(0, target_charge - ev_state['current_charge'])

                    request = ChargingRequest(
                        ev_id=ev_id,
                        request_time=self.current_time,
                        location=(lon, lat),
                        current_charge=ev_state['current_charge'],
                        region_id=self.region_id,
                        can_reach_fcs=can_reach_fcs,
                        estimated_charge_needed=charge_needed
                    )

                    self.pending_requests.append(request)
                    self.active_requests[ev_id] = request

                    self.data_loader.update_ev_state(ev_id, {
                        'status': 'requesting',
                        'charging_request_time': self.current_time
                    })

                    self._simulate_ev_charging_choice(ev_id, ev_state, request)

    def _simulate_ev_charging_choice(self, ev_id: int, ev_state: Dict, request: ChargingRequest):
        """模拟EV选择充电方式"""
        if not request.can_reach_fcs:
            # 不能到达FCS，选择MCS
            point = self.data_loader.find_nearest_dispatch_point(
                request.location[1], request.location[0], self.region_id
            )
            if point:
                self.data_loader.update_ev_state(ev_id, {
                    'target_charging_location': (point['latitude'], point['longitude']),
                    'status': 'waiting',
                    'chose_fcs': False
                })
        else:
            # 可以到达FCS，根据动态概率选择
            total_piles = len(self.fcs.charging_piles)
            used_piles = total_piles - len(self.fcs.get_available_piles())
            load_ratio = used_piles / total_piles

            base_prob = 0.7
            k = 0.5
            fcs_prob = base_prob * (1 - k * load_ratio)

            # 添加随机扰动
            noise = self.np_random.uniform(-0.05, 0.05)
            fcs_prob = np.clip(fcs_prob + noise, 0.1, 0.9)

            if self.np_random.random() < fcs_prob:
                # 选择FCS
                self.data_loader.update_ev_state(ev_id, {
                    'target_charging_location': (self.fcs.location[1], self.fcs.location[0]),
                    'status': 'going_to_fcs',
                    'chose_fcs': True
                })

                # 扣除前往FCS的电量
                fcs_distance = haversine_distance(
                    request.location[1], request.location[0],
                    self.fcs.location[1], self.fcs.location[0]
                )
                energy_cost = fcs_distance * config.ENERGY_CONSUMPTION
                new_charge = max(0, ev_state['current_charge'] - energy_cost)

                self.data_loader.update_ev_state(ev_id, {
                    'current_charge': new_charge
                })
            else:
                # 选择MCS
                point = self.data_loader.find_nearest_dispatch_point(
                    request.location[1], request.location[0], self.region_id
                )
                if point:
                    self.data_loader.update_ev_state(ev_id, {
                        'target_charging_location': (point['latitude'], point['longitude']),
                        'status': 'waiting',
                        'chose_fcs': False
                    })

    def _calculate_simple_reward(self) -> float:
        """改进的稳定奖励函数"""
        reward = 0.0

        # 1. 增量式奖励 - 避免突变
        served = self.episode_stats['served_requests']
        failed = self.episode_stats['failed_requests']

        newly_served = served - self._last_served
        newly_failed = failed - self._last_failed

        self._last_served = served
        self._last_failed = failed

        # 2. 平滑的奖励组件
        # 服务奖励 (主要激励)
        service_reward = newly_served * 1.0

        # 失败惩罚 (温和惩罚)
        failure_penalty = newly_failed * 0.3

        # 3. 等待时间惩罚 (归一化)
        if self.episode_stats['wait_time_count'] > 0:
            avg_wait = self.episode_stats['total_wait_time'] / self.episode_stats['wait_time_count']
            wait_penalty = np.clip(avg_wait / 30.0, 0, 1.0) * 0.2
        else:
            wait_penalty = 0.0

        # 4. MCS利用率奖励 (鼓励高效调度)
        num_idle_mcs = sum(1 for mcs in self.mcs_list if mcs.status == MCSStatus.IDLE)
        num_pending = len(self.pending_requests)

        # 如果有请求但MCS空闲,给予轻微惩罚
        if num_pending > 0 and num_idle_mcs > 0:
            idle_penalty = 0.1 * (num_idle_mcs / self.num_mcs)
        else:
            idle_penalty = 0.0

        # 5. 组合奖励
        reward = service_reward - failure_penalty - wait_penalty - idle_penalty

        # 6. 平滑裁剪 - 避免极端值
        reward = np.clip(reward, -2.0, 2.0)

        return reward

    def reset(self) -> np.ndarray:
        """重置环境"""
        served = self.episode_stats['served_requests']
        failed = self.episode_stats['failed_requests']
        failed_requests_fcs = self.episode_stats['failed_requests_fcs']
        failed_requests_mcs = self.episode_stats['failed_requests_mcs']
        total = served + failed + len(self.pending_requests)
        mcs_served = self.episode_stats['mcs_served']
        fcs_served = self.episode_stats['fcs_served']
        print(f"region_id:{self.region_id} served:{served}, fcs_served:{fcs_served}, mcs_served:{mcs_served}, total:{total}, "
              f"failed_requests_fcs:{failed_requests_fcs}, failed_requests_mcs:{failed_requests_mcs}")

        self.current_step = 0
        self.current_time = self.start_time
        self.pending_requests = []
        self.active_requests = {}
        self.served_ev_ids = set()
        self._last_served = 0
        self._last_failed = 0

        self.data_loader.reset_ev_states()

        # 重置MCS
        for mcs in self.mcs_list:
            mcs.reset_charging_state()
            point = self.np_random.choice(np.array(self.dispatch_points))
            mcs.current_location = (point['longitude'], point['latitude'])
            mcs.total_distance_traveled = 0.0
            mcs.income = 0.0

        # 重置FCS(包括新的队列管理)
        for pile in self.fcs.charging_piles:
            pile.is_occupied = False
            pile.ev_id = None
            pile.charging_start_time = None
            pile.estimated_finish_time = None
            pile.charge_amount = 0
        self.fcs.waiting_queue = []  # 清空队列

        # 重置统计
        self.episode_stats = {
            'served_requests': 0,
            'failed_requests': 0,
            'failed_requests_fcs': 0,
            'failed_requests_mcs': 0,
            'timeout_requests': 0,
            'total_wait_time': 0.0,
            'wait_time_count': 0,
            'mcs_served': 0,
            'fcs_served': 0
        }

        return self._get_obs()

    def render(self, mode='human'):
        pass

    def get_success_rate(self) -> float:
        served = self.episode_stats['served_requests']
        failed = self.episode_stats['failed_requests']
        total = served + failed + len(self.pending_requests)
        return (served / total * 100) if total > 0 else 0.0

    def get_success_served(self) -> int:
        served = self.episode_stats['served_requests']
        return served

    def get_avg_wait_time(self) -> float:
        total_wait = self.episode_stats['total_wait_time']
        count = self.episode_stats['wait_time_count']

        # 加上当前正在等待的请求
        if len(self.pending_requests) > 0:
            current_waits = [
                (self.current_time - req.request_time).total_seconds() / 60
                for req in self.pending_requests
            ]
            total_wait += sum(current_waits)
            count += len(current_waits)

        return total_wait / count if count > 0 else 0.0

    def _extract_point_features(self):
        pass