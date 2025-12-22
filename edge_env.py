import datetime
import gym
import numpy as np
import random
import torch
from gym import spaces
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import timedelta
from collections import deque
from dataloader import DataLoaderFactory
from params_config import Config
from charging_entities import MCS, MCSStatus, FCS, ChargingRequest, ChargingPile
from distance import haversine_distance

config = Config()


@dataclass
class PointFeatureResult:
    """调度点特征提取的结果容器。"""
    reachable_indices: List[int]


# ============================================================
# 简化的调度追踪器
# ============================================================
class SimpleDispatchTracker:
    """简化的调度追踪器 - 基于统计变化"""

    def __init__(self, env):
        self.env = env
        self.last_served = 0
        self.last_failed = 0
        self.step_count = 0

    def update(self, matching, step_reward):
        """每步更新 - 使用启发式规则"""
        current_served = self.env.episode_stats['served_requests']
        current_failed = self.env.episode_stats['failed_requests']

        # 计算本步的变化
        delta_served = current_served - self.last_served
        delta_failed = current_failed - self.last_failed

        # 更新记录
        self.last_served = current_served
        self.last_failed = current_failed
        self.step_count += 1

        # 如果有调度动作
        if len(matching) > 0:
            # 计算平均响应时间（估计）
            if self.env.episode_stats['wait_time_count'] > 0:
                avg_response = (
                        self.env.episode_stats['total_wait_time'] /
                        self.env.episode_stats['wait_time_count']
                )
            else:
                avg_response = 8.0  # 默认估计

            # 为每个调度点更新历史
            for mcs_idx, point_idx in matching.items():
                # 启发式判断成功：
                # 1. 如果有新服务的请求 -> 成功
                # 2. 如果reward为正 -> 可能成功
                # 3. 否则 -> 可能失败

                if delta_served > 0:
                    # 有新服务，按比例分配成功
                    success_prob = delta_served / len(matching)
                    is_success = (hash(f"{point_idx}_{self.step_count}") % 100) < (success_prob * 100)
                    response_time = avg_response
                elif step_reward > 0.1:
                    # Reward为正，可能是好的调度
                    is_success = True
                    response_time = avg_response * 1.2
                else:
                    # 默认情况
                    is_success = delta_failed == 0  # 如果没有新失败，算成功
                    response_time = avg_response * 1.5

                # 更新状态构建器
                if hasattr(self.env, 'state_builder'):
                    self.env.state_builder.update_point_history(
                        point_idx,
                        is_success,
                        response_time
                    )

    def reset(self):
        self.last_served = 0
        self.last_failed = 0
        self.step_count = 0


# ============================================================
# 增强的状态构建器
# ============================================================
class EnhancedStateBuilder:
    """增强的状态构建器 - 提供更丰富的特征"""

    def __init__(self, env):
        self.env = env
        self.grid_size = 10

        # 历史统计
        self.point_dispatch_history = {}  # {point_idx: [success, fail, total]}
        self.point_response_times = {}  # {point_idx: deque of response times}

        for i in range(env.num_dispatch_points):
            self.point_dispatch_history[i] = [0, 0, 0]  # [成功, 失败, 总数]
            self.point_response_times[i] = deque(maxlen=20)

    def update_point_history(self, point_idx: int, success: bool, response_time: float):
        """更新调度点历史"""
        if point_idx >= len(self.point_dispatch_history):
            return

        if success:
            self.point_dispatch_history[point_idx][0] += 1
        else:
            self.point_dispatch_history[point_idx][1] += 1
        self.point_dispatch_history[point_idx][2] += 1

        self.point_response_times[point_idx].append(response_time)

    def get_point_features(self, point_idx: int) -> Dict:
        """获取单个调度点的特征"""
        point = self.env.dispatch_points[point_idx]
        location = (point['longitude'], point['latitude'])

        # 1. 计算附近的请求数量和紧急度
        nearby_requests = 0
        urgency_scores = []

        for req in self.env.pending_requests:
            distance = haversine_distance(
                location[1], location[0],
                req.location[1], req.location[0]
            )

            if distance <= 3.0:  # 3公里范围内
                nearby_requests += 1
                # 紧急度 = 等待时间 / 最大等待时间
                wait_time = (self.env.current_time - req.request_time).total_seconds() / 60
                urgency = min(wait_time / 15.0, 1.0)
                urgency_scores.append(urgency)

        avg_urgency = np.mean(urgency_scores) if urgency_scores else 0.0

        # 2. 计算最近MCS距离和可达MCS数量
        mcs_distances = []
        reachable_mcs = 0

        for mcs in self.env.mcs_list:
            if mcs.status.value == "idle":
                distance = haversine_distance(
                    location[1], location[0],
                    mcs.current_location[1], mcs.current_location[0]
                )
                mcs_distances.append(distance)

                if distance <= config.MCS_SCHEDULE_R:
                    reachable_mcs += 1

        nearest_mcs_dist = min(mcs_distances) if mcs_distances else 10.0
        is_reachable = reachable_mcs > 0

        # 3. 历史特征
        history = self.point_dispatch_history[point_idx]
        success_rate = history[0] / max(history[2], 1)

        response_times = list(self.point_response_times[point_idx])
        avg_response = np.mean(response_times) if response_times else 0.0

        return {
            'nearby_requests': nearby_requests,
            'nearest_mcs_distance': nearest_mcs_dist,
            'avg_urgency': avg_urgency,
            'historical_success_rate': success_rate,
            'avg_response_time': avg_response,
            'is_reachable': is_reachable,
            'num_reachable_mcs': reachable_mcs
        }

    def build_enhanced_observation(self) -> np.ndarray:
        """构建增强的观察状态"""
        obs_parts = []

        # ============ 部分1: 全局统计 (10维) ============
        num_pending = len(self.env.pending_requests)
        num_idle_mcs = sum(1 for mcs in self.env.mcs_list if mcs.status.value == "idle")
        num_moving_mcs = sum(1 for mcs in self.env.mcs_list if mcs.status.value == "moving")
        num_charging_mcs = sum(1 for mcs in self.env.mcs_list if mcs.status.value == "charging")

        # FCS状态
        fcs_available = len(self.env.fcs.get_available_piles())
        fcs_total = len(self.env.fcs.charging_piles)
        fcs_load = 1.0 - (fcs_available / fcs_total)

        # 请求紧急度统计
        urgency_scores = []
        for req in self.env.pending_requests:
            wait_time = (self.env.current_time - req.request_time).total_seconds() / 60
            urgency = min(wait_time / 15.0, 1.0)
            urgency_scores.append(urgency)

        avg_urgency = np.mean(urgency_scores) if urgency_scores else 0.0
        max_urgency = np.max(urgency_scores) if urgency_scores else 0.0

        # 历史性能
        served = self.env.episode_stats['served_requests']
        failed = self.env.episode_stats['failed_requests']
        total_req = served + failed
        success_rate = served / max(total_req, 1)

        global_features = [
            num_pending / 20.0,
            num_idle_mcs / len(self.env.mcs_list),
            num_moving_mcs / len(self.env.mcs_list),
            num_charging_mcs / len(self.env.mcs_list),
            fcs_load,
            avg_urgency,
            max_urgency,
            success_rate,
            self.env.current_step / self.env.max_steps,
            len(urgency_scores) / 20.0
        ]

        obs_parts.extend(global_features)
        #
        # # ============ 部分2: 每个调度点的详细特征 (7维 * num_points) ============
        # point_features_list = []
        #
        # for point_idx in range(self.env.num_dispatch_points):
        #     features = self.get_point_features(point_idx)
        #
        #     point_features = [
        #         min(features['nearby_requests'] / 5.0, 1.0),
        #         np.clip(features['nearest_mcs_distance'] / 10.0, 0, 1),
        #         features['avg_urgency'],
        #         features['historical_success_rate'],
        #         np.clip(features['avg_response_time'] / 10.0, 0, 1),
        #         1.0 if features['is_reachable'] else 0.0,
        #         min(features['num_reachable_mcs'] / 3.0, 1.0)
        #     ]
        #
        #     point_features_list.extend(point_features)
        #
        # obs_parts.extend(point_features_list)

        # ============ 部分3: 空间热力图 (grid_size * grid_size * 2) ============
        # 请求热力图
        # request_heatmap = self._build_request_heatmap()
        # obs_parts.extend(request_heatmap)
        #
        # # MCS位置热力图
        # mcs_heatmap = self._build_mcs_heatmap()
        # obs_parts.extend(mcs_heatmap)

        # ============ 部分4: MCS摘要特征 (5维) ============
        mcs_summary = self._build_mcs_summary()
        obs_parts.extend(mcs_summary)

        # 转换为数组并进行安全处理
        obs_array = np.array(obs_parts, dtype=np.float32)
        obs_array = np.clip(obs_array, -10, 10)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=10.0, neginf=-10.0)

        return obs_array

    def _build_request_heatmap(self) -> List[float]:
        """构建请求热力图 - 考虑紧急度加权"""
        heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        if not self.env.pending_requests:
            return heatmap.flatten().tolist()

        minx, miny, maxx, maxy = self.env.region_bounds
        width = max(maxx - minx, 1e-6)
        height = max(maxy - miny, 1e-6)

        for req in self.env.pending_requests:
            lon, lat = req.location
            x_norm = np.clip((lon - minx) / width, 0, 0.999)
            y_norm = np.clip((lat - miny) / height, 0, 0.999)
            x_idx = int(x_norm * self.grid_size)
            y_idx = int(y_norm * self.grid_size)

            # 加权：紧急度越高，热力值越大
            wait_time = (self.env.current_time - req.request_time).total_seconds() / 60
            urgency_weight = 1.0 + min(wait_time / 15.0, 2.0)

            heatmap[y_idx, x_idx] += urgency_weight

        # 归一化
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = np.log1p(heatmap) / np.log1p(max_val)

        return heatmap.flatten().tolist()

    def _build_mcs_heatmap(self) -> List[float]:
        """构建MCS位置热力图 - 显示空闲MCS分布"""
        heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        minx, miny, maxx, maxy = self.env.region_bounds
        width = max(maxx - minx, 1e-6)
        height = max(maxy - miny, 1e-6)

        for mcs in self.env.mcs_list:
            if mcs.status.value == "idle":
                lon, lat = mcs.current_location
                x_norm = np.clip((lon - minx) / width, 0, 0.999)
                y_norm = np.clip((lat - miny) / height, 0, 0.999)
                x_idx = int(x_norm * self.grid_size)
                y_idx = int(y_norm * self.grid_size)

                heatmap[y_idx, x_idx] += 1

        # 归一化
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.flatten().tolist()

    def _build_mcs_summary(self) -> List[float]:
        """构建MCS摘要特征"""
        if self.env.mcs_list:
            lons = [mcs.current_location[0] for mcs in self.env.mcs_list]
            lats = [mcs.current_location[1] for mcs in self.env.mcs_list]
            spatial_std = (np.std(lons) + np.std(lats)) / 2
        else:
            spatial_std = 0

        total_income = sum(mcs.income for mcs in self.env.mcs_list)
        avg_income = total_income / len(self.env.mcs_list) if self.env.mcs_list else 0

        total_distance = sum(mcs.total_distance_traveled for mcs in self.env.mcs_list)
        avg_distance = total_distance / len(self.env.mcs_list) if self.env.mcs_list else 0

        # MCS状态多样性 (熵)
        status_counts = {}
        for mcs in self.env.mcs_list:
            status = mcs.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        total_mcs = len(self.env.mcs_list)
        status_entropy = 0
        for count in status_counts.values():
            p = count / total_mcs
            status_entropy -= p * np.log(p + 1e-8)

        return [
            np.clip(spatial_std / 0.1, 0, 1),
            np.clip(avg_income / 50, 0, 1),
            np.clip(avg_distance / 20, 0, 1),
            status_entropy / 2.0,
            len(self.env.mcs_list) / 10
        ]

    def reset(self):
        """重置历史统计"""
        for i in range(self.env.num_dispatch_points):
            self.point_dispatch_history[i] = [0, 0, 0]
            self.point_response_times[i].clear()


# ============================================================
# MCS匹配器
# ============================================================
class MCSMatcher:
    def match_mcs_to_points_topk(
            self,
            mcs_list: List,
            dispatch_points: List[Dict],
            action_scores: np.ndarray,
            np_random,
            k: int = 3,
            epsilon: float = 1.0
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

        mcs_order = np_random.permutation(available_mcs_indices).tolist()
        explore_prob = np.clip(epsilon * 0.7, 0.0, 0.7)

        for mcs_idx in mcs_order:
            mcs = mcs_list[mcs_idx]
            reachable = mcs_reachable_map[mcs_idx]

            if not reachable:
                continue

            available_points = [p for p in reachable if p not in used_points]
            if not available_points:
                continue

            point_scores = [(p, float(action_scores[p])) for p in available_points]
            candidate_points = [p for p, _ in point_scores]
            candidate_scores = np.array([s for _, s in point_scores], dtype=float)

            if len(candidate_points) == 1:
                selected_point = candidate_points[0]
            else:
                temperature = 0.5 + 2.0 * explore_prob
                stabilized = candidate_scores - np.max(candidate_scores)
                stabilized = np.clip(stabilized / max(1e-6, temperature), -10, 10)
                score_probs = np.exp(stabilized)
                score_sum = np.sum(score_probs)

                if score_sum > 1e-8:
                    score_probs /= score_sum
                else:
                    score_probs = np.ones_like(score_probs) / len(score_probs)

                uniform_probs = np.ones_like(score_probs) / len(score_probs)
                mix_probs = explore_prob * uniform_probs + (1 - explore_prob) * score_probs
                mix_probs /= np.sum(mix_probs)

                selected_point = int(np_random.choice(candidate_points, p=mix_probs))

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
    if np.any(np.isnan(action_scores)):
        action_scores = np.nan_to_num(action_scores, nan=0.0)

    if np.any(np.isinf(action_scores)):
        action_scores = np.clip(action_scores, -10, 10)

    return action_scores


# ============================================================
# 主环境类
# ============================================================
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

        self.point_ema = np.zeros(len(self.dispatch_points))  # shape = (20,)
        self.point_count = np.zeros(len(self.dispatch_points))  # 当前 step 的请求计数
        self.nearby_mcs_count = np.zeros(len(self.dispatch_points))  # 供给能力的EMA统计
        self.cluster_count = np.zeros(len(self.dispatch_points))

        # 观察空间: 在第一次 reset 后根据实际状态长度确定
        self.grid_size = 10
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
        self.active_requests = {}
        self.served_ev_ids = set()

        # 奖励平滑缓冲区
        self.reward_buffer = deque(maxlen=10)
        self.success_rate_buffer = deque(maxlen=20)

        # 统计追踪的稳定性
        self.stats_ema_alpha = 0.1
        self.ema_wait_time = 0.0
        self.ema_success_rate = 0.0

        # 统计信息
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
        self.training_progress = 0.0

        # ===== 新增: 状态构建器和调度追踪器 =====
        self.state_builder = EnhancedStateBuilder(self)
        self.dispatch_tracker = SimpleDispatchTracker(self)

        self.reset()

        # 在 reset 后补充 observation_space
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

        if self.dispatch_points:
            lons = [p['longitude'] for p in self.dispatch_points]
            lats = [p['latitude'] for p in self.dispatch_points]
            return min(lons), min(lats), max(lons), max(lats)

        center_lon, center_lat = self.data_loader.get_region_center(self.region_id)
        delta = 0.01
        return center_lon - delta, center_lat - delta, center_lon + delta, center_lat + delta

    def _get_obs(self) -> np.ndarray:
        """使用增强的状态构建器获取观察"""
        return self.state_builder.build_enhanced_observation()

    def _extract_point_features(self) -> PointFeatureResult:
        """提取当前可调度的调度点索引,用于构建动作可达掩码。"""
        reachable_indices = set()

        for mcs in self.mcs_list:
            if mcs.status == MCSStatus.IDLE:
                reachable = self.matcher._get_reachable_points(mcs, self.dispatch_points)
                reachable_indices.update(reachable)

        return PointFeatureResult(reachable_indices=sorted(reachable_indices))

    def _compute_dispatch_epsilon(self) -> float:
        """根据训练进度动态调整调度阶段的epsilon - 使用更平滑的衰减"""
        progress = np.clip(self.training_progress, 0.0, 1.0)
        return 1.0 - np.sqrt(progress)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步环境交互"""
        action = validate_action_scores(action)
        dispatch_epsilon = self._compute_dispatch_epsilon()

        # 1. 执行MCS调度
        matching = self.matcher.match_mcs_to_points_topk(
            self.mcs_list,
            self.dispatch_points,
            action,
            self.np_random,
            epsilon=dispatch_epsilon
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

        # 2-6. 环境更新步骤
        self._generate_charging_requests()
        self._update_mcs_status()
        self._process_charging_requests()
        self._update_fcs_status()

        # 7. 更新时间和EV状态
        self.current_step += 1
        self.current_time += timedelta(minutes=config.TIME_STEP)
        self.data_loader.update_ev_movement()

        # 8. 计算稳定的奖励
        reward = self._calculate_stable_reward()

        # 9. 获取新观察
        obs = self._get_obs()

        # 10. 检查是否结束
        done = self.current_step >= self.max_steps

        # 11. 构建信息字典
        info = {
            'current_step': self.current_step,
            'pending_requests': len(self.pending_requests),
            'episode_stats': self.episode_stats.copy(),
            'matching': matching
        }

        # ===== 新增：更新调度追踪 =====
        self.dispatch_tracker.update(matching, reward)

        self.update_ema()
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
                piles = self.fcs.charging_piles
                for pile in piles:
                    if pile.estimated_finish_time is not None:
                        continue
                    if pile.is_occupied and pile.ev_id == request.ev_id:
                        self._start_fcs_charging(request.ev_id, pile)
                        served_requests.append(request)
                        self.episode_stats['total_wait_time'] += wait_time
                        self.episode_stats['wait_time_count'] += 1
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
            if request in self.pending_requests:
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
        """更新FCS状态"""
        for pile in self.fcs.charging_piles:
            if pile.is_occupied and pile.estimated_finish_time:
                if self.current_time >= pile.estimated_finish_time:
                    if pile.ev_id:
                        ev_id = pile.ev_id
                        ev_state = self.data_loader.get_ev_state().get(ev_id)

                        if ev_state:
                            if ev_state.get('charging_request_time'):
                                wait_time = (pile.charging_start_time -
                                             ev_state['charging_request_time']).total_seconds() / 60
                                self.episode_stats['total_wait_time'] += wait_time
                                self.episode_stats['wait_time_count'] += 1

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

                    pile.is_occupied = False
                    pile.ev_id = None
                    pile.charging_start_time = None
                    pile.estimated_finish_time = None
                    pile.charge_amount = 0

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

        self.data_loader.update_ev_state(ev_id, {
            'status': 'charging',
            'assigned_fcs': self.fcs.id,
            'charging_start_time': self.current_time,
            'chose_fcs': True
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
            point = self.data_loader.find_nearest_dispatch_point(
                request.location[1], request.location[0], self.region_id
            )
            if point:
                self.data_loader.update_ev_state(ev_id, {
                    'target_charging_location': (point['latitude'], point['longitude']),
                    'status': 'waiting',
                    'chose_fcs': False
                })
                self.point_count[point['id']] += 1
        else:
            total_piles = len(self.fcs.charging_piles)
            used_piles = total_piles - len(self.fcs.get_available_piles())
            load_ratio = used_piles / total_piles

            if load_ratio != 1:
                self.data_loader.update_ev_state(ev_id, {
                    'target_charging_location': (self.fcs.location[1], self.fcs.location[0]),
                    'status': 'waiting',
                    'chose_fcs': True
                })

                fcs_distance = haversine_distance(
                    request.location[1], request.location[0],
                    self.fcs.location[1], self.fcs.location[0]
                )
                energy_cost = fcs_distance * config.ENERGY_CONSUMPTION
                new_charge = max(0, ev_state['current_charge'] - energy_cost)

                self.data_loader.update_ev_state(ev_id, {
                    'current_charge': new_charge
                })

                assigned_fcs = self.fcs.get_available_piles()[0]
                assigned_fcs.is_occupied = True
                assigned_fcs.ev_id = request.ev_id
            else:
                point = self.data_loader.find_nearest_dispatch_point(
                    request.location[1], request.location[0], self.region_id
                )
                if point:
                    self.data_loader.update_ev_state(ev_id, {
                        'target_charging_location': (point['latitude'], point['longitude']),
                        'status': 'waiting',
                        'chose_fcs': False
                    })
                self.point_count[point['id']] += 1

    def update_ema(self):
        alpha = 0.2
        self.point_ema = (
                alpha * self.point_count
                + (1 - alpha) * self.point_ema
        )

        # 供给能力：当前点2km范围内MCS数量
        supply_counts = np.zeros_like(self.nearby_mcs_count)

        for idx, point in enumerate(self.dispatch_points):
            for mcs in self.mcs_list:
                distance = haversine_distance(
                    point['latitude'], point['longitude'],
                    mcs.current_location[1], mcs.current_location[0]
                )
                if distance <= 2.0:
                    supply_counts[idx] += 1

        self.nearby_mcs_count = (
                alpha * supply_counts
                + (1 - alpha) * self.nearby_mcs_count
        )
        # cluster_count 暂时与供给能力保持一致，防止空值使用
        self.cluster_count = self.nearby_mcs_count.copy()
        self.point_count[:] = 0  # 清空，准备下一个 step

    def score_point(self, p):
        demand = self.point_ema[p]
        supply = self.nearby_mcs_count[p]  # 或同一簇已有车数
        cluster = self.cluster_count[p]  # 可与 supply 合并
        return demand - 0.8 * supply - 0.5 * cluster

    def _calculate_stable_reward(self) -> float:
        """计算更稳定的奖励函数"""
        reward = 0.0

        # 1. 使用更平滑的增量奖励
        served = self.episode_stats['served_requests']
        failed = self.episode_stats['failed_requests']

        newly_served = served - self._last_served
        newly_failed = failed - self._last_failed

        self._last_served = served
        self._last_failed = failed

        # 2. 平滑的奖励组件
        service_reward = np.sqrt(newly_served + 1) - 1
        failure_penalty = newly_failed * 0.2

        # 3. 使用归一化的等待时间惩罚
        if self.episode_stats['wait_time_count'] > 0:
            avg_wait = self.episode_stats['total_wait_time'] / self.episode_stats['wait_time_count']
            wait_penalty = 0.3 / (1 + np.exp(-0.1 * (avg_wait - 15)))
        else:
            wait_penalty = 0.0

        # 4. 更温和的MCS利用率激励
        num_idle_mcs = sum(1 for mcs in self.mcs_list if mcs.status == MCSStatus.IDLE)
        num_pending = len(self.pending_requests)

        if num_pending > 0 and num_idle_mcs > 0:
            idle_penalty = 0.05 * np.log1p(num_idle_mcs / max(self.num_mcs, 1))
        else:
            idle_penalty = 0.0

        # 5. 添加成功率激励
        total_requests = served + failed
        if total_requests > 0:
            current_success_rate = served / total_requests
            self.success_rate_buffer.append(current_success_rate)

            if len(self.success_rate_buffer) > 5:
                avg_success_rate = np.mean(self.success_rate_buffer)
                if current_success_rate > avg_success_rate:
                    success_bonus = 0.1
                else:
                    success_bonus = 0.0
            else:
                success_bonus = 0.0
        else:
            success_bonus = 0.0

        # 6. 组合奖励
        raw_reward = service_reward - failure_penalty - wait_penalty - idle_penalty + success_bonus

        # 7. 使用移动平均平滑奖励
        self.reward_buffer.append(raw_reward)
        if len(self.reward_buffer) > 1:
            smoothed_reward = np.mean(self.reward_buffer)
        else:
            smoothed_reward = raw_reward

        # 8. 使用更温和的裁剪
        final_reward = np.clip(smoothed_reward, -1.5, 1.5)

        return final_reward

    def reset(self) -> np.ndarray:
        """重置环境"""
        # 打印统计信息
        served = self.episode_stats['served_requests']
        failed = self.episode_stats['failed_requests']
        failed_requests_fcs = self.episode_stats['failed_requests_fcs']
        failed_requests_mcs = self.episode_stats['failed_requests_mcs']
        total = served + failed + len(self.pending_requests)
        mcs_served = self.episode_stats['mcs_served']
        fcs_served = self.episode_stats['fcs_served']

        print(f"region_id:{self.region_id} served:{served}, fcs_served:{fcs_served}, "
              f"mcs_served:{mcs_served}, total:{total}, "
              f"failed_requests_fcs:{failed_requests_fcs}, failed_requests_mcs:{failed_requests_mcs}")

        # 重置时间
        self.current_step = 0
        self.current_time = self.start_time

        # 重置请求管理
        self.pending_requests = []
        self.active_requests = {}
        self.served_ev_ids = set()
        self._last_served = 0
        self._last_failed = 0

        # 重置平滑缓冲区
        self.reward_buffer.clear()
        self.success_rate_buffer.clear()
        self.ema_wait_time = 0.0
        self.ema_success_rate = 0.0

        # 重置数据加载器
        self.data_loader.reset_ev_states()

        # 重置MCS
        for mcs in self.mcs_list:
            mcs.reset_charging_state()
            point = self.np_random.choice(np.array(self.dispatch_points))
            mcs.current_location = (point['longitude'], point['latitude'])
            mcs.total_distance_traveled = 0.0
            mcs.income = 0.0

        # 重置FCS
        for pile in self.fcs.charging_piles:
            pile.is_occupied = False
            pile.ev_id = None
            pile.charging_start_time = None
            pile.estimated_finish_time = None
            pile.charge_amount = 0

        # 重置统计
        self.episode_stats = {
            'served_requests': 0,
            'failed_requests': 0,
            'failed_requests_fcs': 0,
            'failed_requests_mcs': 0,
            'timeout_requests': 0,
            'total_wait_time': 0,
            'wait_time_count': 0,
            'mcs_served': 0,
            'fcs_served': 0
            }
        # ===== 新增：重置状态构建器和追踪器 =====
        self.state_builder.reset()
        self.dispatch_tracker.reset()

        # 重置EMA相关统计
        self.point_ema[:] = 0
        self.point_count[:] = 0
        self.nearby_mcs_count[:] = 0
        self.cluster_count[:] = 0

        return self._get_obs()

    def render(self, mode='human'):
        pass

    def get_success_rate(self) -> float:
        served = self.episode_stats['served_requests']
        failed = self.episode_stats['failed_requests']
        total = served + failed + len(self.pending_requests)
        return (served / total * 100) if total > 0 else 0.0

    def get_success_served(self) -> int:
        return self.episode_stats['served_requests']

    def get_avg_wait_time(self) -> float:
        total_wait = self.episode_stats['total_wait_time']
        count = self.episode_stats['wait_time_count']

        if len(self.pending_requests) > 0:
            current_waits = [
                (self.current_time - req.request_time).total_seconds() / 60
                for req in self.pending_requests
            ]
            total_wait += sum(current_waits)
            count += len(current_waits)

        return total_wait / count if count > 0 else 0.0
