import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import gym
from gym import spaces

from edge_env import EdgeEnv
from dataloader import DataLoader, DataLoaderFactory
from params_config import Config
from charging_entities import MCS, MCSStatus
from edge_model_manager import EdgeModelManager
config = Config()


class CloudSchedulerEnv(gym.Env):
    """云端跨区域MCS调度环境"""

    def __init__(self,
                 num_regions: int = 18,
                 max_steps: int = 100,
                 max_mcs_per_region: int = 10):
        super(CloudSchedulerEnv, self).__init__()

        self.num_regions = num_regions
        self.max_steps = max_steps
        self.max_mcs_per_region = max_mcs_per_region
        self.initial_mcs_per_region = config.MCS_PER_REGION
        factory = DataLoaderFactory(
            trajectory_file='dataset/top1000evs/reallocated/20140818_processed.csv',
            region_file='dataset/fcs_voronoi_regions.geojson',
            dispatch_file='dataset/dispatch_points_400.csv'
        )
        self.data_loader = factory.create_dataloader()
        self.factory = factory
        # 初始化边缘环境
        self.edge_envs = {
            region_id: EdgeEnv(region_id, factory, max_steps)
            for region_id in range(num_regions)
        }

        # MCS池管理
        self.mcs_pool = {}  # {mcs_id: MCS对象}
        self.region_mcs_mapping = defaultdict(list)  # {region_id: [mcs_ids]}
        self._initialize_mcs_pool()

        # 观察空间: 每个区域的统计信息 + 全局信息
        # 每个区域: [pending_requests, idle_mcs, fcs_load, avg_wait_time, mcs_count]
        self.obs_dim = num_regions * 5 + 10  # 区域信息 + 全局统计
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )

        # 动作空间: 为每对区域决定调度MCS的数量
        # 使用连续动作空间,输出调度矩阵的logits
        self.action_dim = num_regions * num_regions
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.action_dim,), dtype=np.float32
        )

        # 时间管理
        dataset_start = self.data_loader.get_current_time()
        self.start_time = dataset_start.to_pydatetime() if hasattr(dataset_start, 'to_pydatetime') else dataset_start
        self.current_time = self.start_time
        self.current_step = 0

        # 统计信息
        self.episode_stats = {
            'total_dispatches': 0,
            'successful_dispatches': 0,
            'failed_dispatches': 0,
            'total_distance': 0.0,
            'dispatch_history': []
        }

        self.np_random = None
        self.seed()

    def seed(self, seed=config.random_seed):
        if seed is None:
            seed = np.random.randint(0, 2 ** 32 - 1)
        self.np_random = np.random.default_rng(seed)
        # 同步边缘环境的种子
        for env in self.edge_envs.values():
            env.seed(seed)
        return [seed]

    def _initialize_mcs_pool(self):
        """初始化MCS池"""
        mcs_id = 0
        for region_id in range(self.num_regions):
            points = self.data_loader.generate_random_points_in_region(
                region_id, self.initial_mcs_per_region
            )

            for i in range(self.initial_mcs_per_region):
                mcs = MCS(
                    id=f"mcs_{mcs_id}",
                    current_location=points[i],
                    region_id=region_id
                )
                self.mcs_pool[f"mcs_{mcs_id}"] = mcs
                self.region_mcs_mapping[region_id].append(f"mcs_{mcs_id}")
                mcs_id += 1

    def _get_region_stats(self, region_id: int) -> Dict:
        """获取区域统计信息"""
        env = self.edge_envs[region_id]

        pending = len(env.pending_requests)
        idle_mcs = sum(1 for mcs_id in self.region_mcs_mapping[region_id]
                       if self.mcs_pool[mcs_id].status == MCSStatus.IDLE)
        mcs_count = len(self.region_mcs_mapping[region_id])

        fcs_available = len(env.fcs.get_available_piles())
        fcs_load = 1.0 - (fcs_available / config.FCS_CHARGING_PILES)

        avg_wait = env.get_avg_wait_time() if env.episode_stats['wait_time_count'] > 0 else 0

        return {
            'pending': pending,
            'idle_mcs': idle_mcs,
            'mcs_count': mcs_count,
            'fcs_load': fcs_load,
            'avg_wait': avg_wait
        }

    def _get_obs(self) -> np.ndarray:
        """获取全局观察"""
        obs_parts = []

        # 1. 每个区域的统计
        region_stats = []
        for region_id in range(self.num_regions):
            stats = self._get_region_stats(region_id)
            region_feature = [
                stats['pending'] / 20.0,  # 归一化
                stats['idle_mcs'] / self.max_mcs_per_region,
                stats['mcs_count'] / self.max_mcs_per_region,
                stats['fcs_load'],
                min(stats['avg_wait'] / 30.0, 1.0)
            ]
            obs_parts.extend(region_feature)
            region_stats.append(stats)

        # 2. 全局统计
        total_pending = sum(s['pending'] for s in region_stats)
        total_idle = sum(s['idle_mcs'] for s in region_stats)
        total_mcs = sum(s['mcs_count'] for s in region_stats)
        avg_load = np.mean([s['fcs_load'] for s in region_stats])
        avg_wait = np.mean([s['avg_wait'] for s in region_stats if s['avg_wait'] > 0])

        # 需求不均衡度
        pending_std = np.std([s['pending'] for s in region_stats])
        mcs_std = np.std([s['mcs_count'] for s in region_stats])

        # 资源利用率
        utilization = total_idle / total_mcs if total_mcs > 0 else 0

        global_features = [
            total_pending / 100.0,
            total_idle / (self.num_regions * self.max_mcs_per_region),
            total_mcs / (self.num_regions * self.max_mcs_per_region),
            avg_load,
            min(avg_wait / 30.0, 1.0) if avg_wait > 0 else 0,
            pending_std / 10.0,
            mcs_std / 5.0,
            utilization,
            self.current_step / self.max_steps,
            len(self.episode_stats['dispatch_history']) / 100.0
        ]

        obs_parts.extend(global_features)

        return np.array(obs_parts, dtype=np.float32)

    def _parse_action(self, action: np.ndarray) -> Dict[Tuple[int, int], int]:
        """解析动作为调度决策

        Returns:
            {(source_region, target_region): num_mcs}
        """
        # 将action reshape为调度矩阵
        dispatch_logits = action.reshape(self.num_regions, self.num_regions)

        # 使用softmax + 采样策略
        dispatch_matrix = np.zeros((self.num_regions, self.num_regions), dtype=int)

        for source in range(self.num_regions):
            # 获取源区域的空闲MCS数量
            idle_mcs_ids = [
                mcs_id for mcs_id in self.region_mcs_mapping[source]
                if self.mcs_pool[mcs_id].status == MCSStatus.IDLE
            ]
            available = len(idle_mcs_ids)

            if available == 0:
                continue

            # 对该行应用softmax
            row_scores = dispatch_logits[source].copy()
            row_scores[source] = -np.inf  # 不调度给自己

            # 温度参数控制探索
            temperature = 1.0
            row_probs = np.exp(row_scores / temperature)
            row_probs = row_probs / (row_probs.sum() + 1e-8)

            # 基于概率和需求决定调度
            for target in range(self.num_regions):
                if target == source:
                    continue

                target_stats = self._get_region_stats(target)

                # 只在目标有需求且概率足够高时调度
                if target_stats['pending'] > 0 and row_probs[target] > 0.1:
                    # 调度数量: 考虑需求和概率
                    demand_factor = min(target_stats['pending'] / 5.0, 1.0)
                    num_to_dispatch = int(available * row_probs[target] * demand_factor)
                    num_to_dispatch = min(num_to_dispatch, available, 2)  # 限制单次最多2个

                    if num_to_dispatch > 0:
                        dispatch_matrix[source, target] = num_to_dispatch
                        available -= num_to_dispatch

                    if available <= 0:
                        break

        # 转换为字典格式
        dispatch_decisions = {}
        for source in range(self.num_regions):
            for target in range(self.num_regions):
                if dispatch_matrix[source, target] > 0:
                    dispatch_decisions[(source, target)] = dispatch_matrix[source, target]

        return dispatch_decisions

    def _execute_dispatch(self, dispatch_decisions: Dict[Tuple[int, int], int]):
        """执行MCS调度"""
        from distance import haversine_distance

        for (source, target), num_mcs in dispatch_decisions.items():
            # 获取源区域的空闲MCS
            idle_mcs_ids = [
                mcs_id for mcs_id in self.region_mcs_mapping[source]
                if self.mcs_pool[mcs_id].status == MCSStatus.IDLE
            ]

            # 选择要调度的MCS
            mcs_to_dispatch = idle_mcs_ids[:num_mcs]

            # 目标位置
            target_location = self.data_loader.get_region_center(target)

            for mcs_id in mcs_to_dispatch:
                mcs = self.mcs_pool[mcs_id]

                # 计算距离和时间
                distance = haversine_distance(
                    mcs.current_location[1], mcs.current_location[0],
                    target_location[1], target_location[0]
                )

                travel_time = distance / config.MOVING_SPEED * 60
                arrival_time = self.current_time + timedelta(minutes=travel_time)

                # 开始移动
                mcs.start_movement(target_location, arrival_time)
                mcs.total_distance_traveled += distance

                # 更新区域映射(预约)
                self.region_mcs_mapping[source].remove(mcs_id)
                self.region_mcs_mapping[target].append(mcs_id)
                mcs.region_id = target  # 更新归属

                # 记录调度
                self.episode_stats['dispatch_history'].append({
                    'time': self.current_time,
                    'mcs_id': mcs_id,
                    'source': source,
                    'target': target,
                    'distance': distance
                })
                self.episode_stats['total_dispatches'] += 1
                self.episode_stats['total_distance'] += distance

    def _sync_mcs_to_edge_envs(self):
        """同步MCS状态到边缘环境"""
        for region_id, env in self.edge_envs.items():
            # 获取该区域的MCS列表
            region_mcs_ids = self.region_mcs_mapping[region_id]
            region_mcs_list = [self.mcs_pool[mcs_id] for mcs_id in region_mcs_ids]

            # 更新边缘环境的MCS列表
            env.mcs_list = region_mcs_list
            env.num_mcs = len(region_mcs_list)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步"""
        # 1. 解析并执行调度决策
        dispatch_decisions = self._parse_action(action)
        self._execute_dispatch(dispatch_decisions)

        # 2. 同步MCS到边缘环境
        self._sync_mcs_to_edge_envs()

        # 3. 让每个边缘环境执行一步(使用内部策略)
        total_edge_reward = 0
        for region_id, env in self.edge_envs.items():
            # 边缘环境使用简单策略或不采取动作
            dummy_action = np.zeros(env.action_space.shape)
            _, edge_reward, _, _ = env.step(dummy_action)
            total_edge_reward += edge_reward

        # 4. 更新时间
        self.current_step += 1
        self.current_time += timedelta(minutes=config.TIME_STEP)

        # 5. 计算云端调度奖励
        cloud_reward = self._calculate_cloud_reward(dispatch_decisions)

        # 6. 总奖励 = 云端调度奖励 + 边缘服务奖励
        total_reward = cloud_reward + total_edge_reward * 0.5

        # 7. 获取新观察
        obs = self._get_obs()

        # 8. 检查是否结束
        done = self.current_step >= self.max_steps

        # 9. 构建信息
        info = {
            'current_step': self.current_step,
            'dispatch_decisions': dispatch_decisions,
            'cloud_reward': cloud_reward,
            'edge_reward': total_edge_reward,
            'total_dispatches': self.episode_stats['total_dispatches'],
            'region_mcs_counts': {r: len(ids) for r, ids in self.region_mcs_mapping.items()}
        }

        return obs, total_reward, done, info

    def _calculate_cloud_reward(self, dispatch_decisions: Dict) -> float:
        """计算云端调度奖励"""
        reward = 0.0

        # 1. 负载均衡奖励
        region_stats = [self._get_region_stats(r) for r in range(self.num_regions)]

        # MCS分布均衡性
        mcs_counts = [s['mcs_count'] for s in region_stats]
        mcs_std = np.std(mcs_counts)
        mcs_balance_reward = -mcs_std * 0.1

        # 需求-资源匹配度
        for stats in region_stats:
            if stats['pending'] > 0 and stats['idle_mcs'] > 0:
                # 有需求且有资源,给予奖励
                match_score = min(stats['idle_mcs'] / stats['pending'], 1.0)
                reward += match_score * 0.2
            elif stats['pending'] > 5 and stats['idle_mcs'] == 0:
                # 需求高但无资源,惩罚
                reward -= 0.3

        # 2. 调度效率奖励
        if len(dispatch_decisions) > 0:
            # 调度成功奖励
            reward += 0.1 * len(dispatch_decisions)

            # 但要惩罚过度调度
            total_dispatched = sum(dispatch_decisions.values())
            if total_dispatched > 5:
                reward -= 0.05 * (total_dispatched - 5)

        # 3. 距离成本惩罚
        recent_distance = sum(
            h['distance'] for h in self.episode_stats['dispatch_history'][-10:]
        )
        distance_penalty = recent_distance * 0.01

        # 4. 组合奖励
        total_cloud_reward = reward + mcs_balance_reward - distance_penalty

        return np.clip(total_cloud_reward, -2.0, 2.0)

    def reset(self) -> np.ndarray:
        """重置环境"""
        # 重置时间
        self.current_step = 0
        self.current_time = self.start_time

        # 重置统计
        self.episode_stats = {
            'total_dispatches': 0,
            'successful_dispatches': 0,
            'failed_dispatches': 0,
            'total_distance': 0.0,
            'dispatch_history': []
        }

        # 重置MCS池 - 重新分配到初始区域
        self.region_mcs_mapping = defaultdict(list)
        for mcs_id, mcs in self.mcs_pool.items():
            initial_region = int(mcs_id.split('_')[1]) // self.initial_mcs_per_region
            mcs.region_id = initial_region
            mcs.reset_charging_state()

            # 重置位置
            points = self.data_loader.generate_random_points_in_region(initial_region, 1)
            mcs.current_location = points[0]

            self.region_mcs_mapping[initial_region].append(mcs_id)

        # 重置边缘环境
        for env in self.edge_envs.values():
            env.reset()

        # 同步MCS
        self._sync_mcs_to_edge_envs()

        return self._get_obs()

    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
            print(f"\n=== Cloud Scheduler Step {self.current_step} ===")
            print(f"Total MCS: {len(self.mcs_pool)}")
            print(f"Total Dispatches: {self.episode_stats['total_dispatches']}")
            print(f"Total Distance: {self.episode_stats['total_distance']:.2f} km")

            print("\nRegion MCS Distribution:")
            for region_id in range(self.num_regions):
                count = len(self.region_mcs_mapping[region_id])
                stats = self._get_region_stats(region_id)
                print(f"  Region {region_id}: {count} MCS, "
                      f"{stats['pending']} pending, "
                      f"{stats['idle_mcs']} idle")


class HierarchicalCloudEnv(CloudSchedulerEnv):
    """使用边缘模型的分层云端环境"""

    def __init__(self,
                 edge_model_manager: EdgeModelManager,
                 num_regions: int = 18,
                 max_steps: int = 100,
                 max_mcs_per_region: int = 10):
        super().__init__(num_regions, max_steps, max_mcs_per_region)
        self.edge_model_manager = edge_model_manager
        print("\n🔗 初始化分层云端环境:")
        print(f"  - 区域数量: {num_regions}")
        print(f"  - 已加载边缘模型: {sum(edge_model_manager.is_loaded.values())}/{num_regions}")

    def step(self, action: np.ndarray):
        """重写step函数,使用边缘模型"""
        # 1. 云端执行跨区域调度
        dispatch_decisions = self._parse_action(action)
        self._execute_dispatch(dispatch_decisions)

        # 2. 同步MCS到边缘环境
        self._sync_mcs_to_edge_envs()

        # 3. 每个边缘环境使用训练好的模型执行动作
        total_edge_reward = 0
        edge_actions_taken = 0

        for region_id, env in self.edge_envs.items():
            # 获取边缘环境状态
            edge_state = env._get_obs()

            # 使用边缘模型生成动作
            edge_action = self.edge_model_manager.get_action(region_id, edge_state)

            # 执行边缘动作
            _, edge_reward, _, edge_info = env.step(edge_action)

            total_edge_reward += edge_reward
            edge_actions_taken += 1

        # 4. 更新时间
        self.current_step += 1
        self.current_time += timedelta(minutes=config.TIME_STEP)

        # 5. 计算云端调度奖励
        cloud_reward = self._calculate_cloud_reward(dispatch_decisions)

        # 6. 总奖励 = 云端调度奖励 + 边缘服务奖励(加权)
        total_reward = cloud_reward + total_edge_reward * 0.5

        # 7. 获取新观察
        obs = self._get_obs()
        done = self.current_step >= self.max_steps

        # 8. 构建信息
        info = {
            'current_step': self.current_step,
            'dispatch_decisions': dispatch_decisions,
            'cloud_reward': cloud_reward,
            'edge_reward': total_edge_reward,
            'avg_edge_reward': total_edge_reward / edge_actions_taken if edge_actions_taken > 0 else 0,
            'total_dispatches': self.episode_stats['total_dispatches'],
            'region_mcs_counts': {r: len(ids) for r, ids in self.region_mcs_mapping.items()}
        }

        return obs, total_reward, done, info
