import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pickle
import os
from params_config import Config

config = Config()


class SOCManager:
    """SOC(State of Charge)管理器 - 全局电量表"""

    def __init__(self, trajectory_file: str, config: Config):
        self.config = config
        self.soc_file = self._get_soc_filename(trajectory_file)
        self.soc_table: Dict[int, float] = {}

        # 加载或生成SOC表
        if os.path.exists(self.soc_file):
            self._load_soc_table()
        else:
            self._generate_soc_table(trajectory_file)
            self._save_soc_table()

    def _get_soc_filename(self, trajectory_file: str) -> str:
        """根据轨迹文件生成SOC表文件名"""
        base_name = os.path.splitext(os.path.basename(trajectory_file))[0]
        return f'dataset/soc_tables/soc_{base_name}.pkl'

    def _generate_soc_table(self, trajectory_file: str):
        """生成初始SOC表"""
        print(f"生成SOC表: {self.soc_file}")

        # 读取所有唯一的EV ID
        df = pd.read_csv(trajectory_file)
        ev_ids = df['id'].unique()

        # 根据配置的分布模式生成SOC
        np.random.seed(self.config.random_seed)

        if self.config.distribution_mode == 'normal':
            # 正态分布
            soc_values = np.random.normal(
                self.config.mean_percentage,
                self.config.std_percentage,
                size=len(ev_ids)
            )
            # 截断到[min, max]范围
            soc_values = np.clip(
                soc_values,
                self.config.min_percentage,
                self.config.max_percentage
            )
        elif self.config.distribution_mode == 'uniform':
            # 均匀分布
            soc_values = np.random.uniform(
                self.config.min_percentage,
                self.config.max_percentage,
                size=len(ev_ids)
            )
        else:
            raise ValueError(f"未知的分布模式: {self.config.distribution_mode}")

        # 转换为实际电量 (kWh)
        for ev_id, soc_pct in zip(ev_ids, soc_values):
            self.soc_table[int(ev_id)] = soc_pct * self.config.BATTERY_CAPACITY

        print(f"  生成了 {len(self.soc_table)} 个EV的初始电量")
        print(f"  平均SOC: {np.mean(soc_values) * 100:.1f}%")
        print(f"  SOC范围: [{np.min(soc_values) * 100:.1f}%, {np.max(soc_values) * 100:.1f}%]")

    def _save_soc_table(self):
        """保存SOC表到文件"""
        os.makedirs(os.path.dirname(self.soc_file), exist_ok=True)
        with open(self.soc_file, 'wb') as f:
            pickle.dump(self.soc_table, f)
        print(f"  SOC表已保存: {self.soc_file}")

    def _load_soc_table(self):
        """从文件加载SOC表"""
        with open(self.soc_file, 'rb') as f:
            self.soc_table = pickle.load(f)
        print(f"从文件加载SOC表: {self.soc_file} ({len(self.soc_table)} 个EV)")

    def get_initial_soc(self, ev_id: int) -> float:
        """获取EV的初始电量"""
        return self.soc_table.get(ev_id, self.config.BATTERY_CAPACITY * 0.8)


