import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Dict
import os
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter

from dataloader import DataLoader
from params_config import Config
from edge_env import EdgeEnv

config = Config()

# 经验回放缓冲区
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'reachable_mask'))


class OptimalPointTracker:
    """追踪历史最优调度点"""

    def __init__(self, num_points: int, window_size: int = 100, alpha: float = 0.95):
        self.num_points = num_points
        self.alpha = alpha
        self.point_rewards = np.zeros(num_points)
        self.point_counts = np.zeros(num_points)
        self.ema_rewards = np.zeros(num_points)

    def update(self, point_indices: List[int], rewards: List[float]):
        for idx, reward in zip(point_indices, rewards):
            if 0 <= idx < self.num_points:
                self.point_counts[idx] += 1
                self.point_rewards[idx] += reward
                if self.point_counts[idx] == 1:
                    self.ema_rewards[idx] = reward
                else:
                    self.ema_rewards[idx] = (self.alpha * self.ema_rewards[idx] +
                                             (1 - self.alpha) * reward)

    def get_exploration_distribution(self, reachable_mask: np.ndarray,
                                     temperature: float = 1.0) -> np.ndarray:
        scores = np.copy(self.ema_rewards)
        mask = self.point_counts > 0
        # 未探索点给予平均分
        if np.any(mask):
            scores[~mask] = np.mean(self.ema_rewards[mask])
        scores[~reachable_mask] = -np.inf

        exp_scores = np.exp(scores / temperature)
        exp_scores[~reachable_mask] = 0

        if exp_scores.sum() > 0:
            return exp_scores / exp_scores.sum()
        else:
            probs = reachable_mask.astype(float)
            return probs / probs.sum() if probs.sum() > 0 else probs


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""

    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta  # 重要性采样指数
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, state, action, next_state, reward, done, reachable_mask):
        """添加经验(使用最大优先级)"""
        transition = Transition(state, action, next_state, reward, done, reachable_mask)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """优先级采样"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        # 计算采样概率
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class ImprovedDQNNetwork(nn.Module):
    """改进的DQN网络 - 使用Dueling架构"""

    def __init__(self, state_dim: int, max_action_dim: int, hidden_dims: List[int] = [256, 256, 128]):
        super(ImprovedDQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.max_action_dim = max_action_dim

        # 共享特征提取层
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Dueling 架构
        # Value stream (状态价值)
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )

        # Advantage stream (动作优势)
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], max_action_dim)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - Dueling DQN
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        features = self.feature_extractor(state)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # 组合: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))

        return q_values


class RewardNormalizer:
    """奖励归一化器 - 使用运行统计"""

    def __init__(self, clip_range: float = 10.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.clip_range = clip_range

    def update(self, reward: float):
        """更新统计信息"""
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        self.var += delta * (reward - self.mean)

    def normalize(self, reward: float) -> float:
        """归一化奖励"""
        if self.count < 2:
            return reward

        std = np.sqrt(self.var / (self.count - 1))
        std = max(std, 1e-6)  # 避免除零

        normalized = (reward - self.mean) / std
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        std = np.sqrt(self.var / (self.count - 1)) if self.count > 1 else 1.0
        return {
            'mean': self.mean,
            'std': std,
            'count': self.count
        }


class ImprovedDQNAgent:
    """改进的DQN智能体 - 支持Double DQN和优先级回放"""

    def __init__(self,
                 state_dim: int,
                 max_action_dim: int,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 10000,
                 target_update_freq: int = 1000,
                 use_double_dqn: bool = True,
                 use_reward_norm: bool = True,
                 local_explore_prob: float = 0.6,  # 新增参数
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.state_dim = state_dim
        self.max_action_dim = max_action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_reward_norm = use_reward_norm
        self.device = device
        self.local_explore_prob = local_explore_prob  # 新增

        # Q网络和目标网络
        self.q_network = ImprovedDQNNetwork(state_dim, max_action_dim).to(device)
        self.target_network = ImprovedDQNNetwork(state_dim, max_action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # 优化器 - 使用AdamW
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=1e-5)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2, eta_min=1e-6
        )

        # 学习步数计数
        self.learn_step_counter = 0

        # Q值统计(用于监控)
        self.q_values_history = deque(maxlen=1000)

        # 奖励归一化器
        self.reward_normalizer = RewardNormalizer() if use_reward_norm else None
        self.optimal_tracker = OptimalPointTracker(max_action_dim)

    def select_action(self, state: np.ndarray, reachable_mask: np.ndarray,
                      epsilon: float = None) -> np.ndarray:
        """改进的动作选择 - 使用局部探索"""
        if epsilon is None:
            epsilon = self.epsilon

        action_scores = np.full(len(reachable_mask), -1e9, dtype=np.float32)
        reachable_indices = np.where(reachable_mask)[0]

        if len(reachable_indices) == 0:
            return action_scores

        if random.random() < epsilon:
            # 探索模式
            if random.random() < self.local_explore_prob:
                # 局部探索: 基于历史最优点
                explore_probs = self.optimal_tracker.get_exploration_distribution(
                    reachable_mask, temperature=0.5
                )

                if explore_probs.sum() > 0:
                    # 从分布中采样
                    try:
                        action_scores[reachable_indices] = np.random.dirichlet(
                            explore_probs[reachable_indices] * 10 + 0.1
                        )
                    except:
                        # fallback
                        action_scores[reachable_indices] = np.random.randn(len(reachable_indices))
                else:
                    action_scores[reachable_indices] = np.random.randn(len(reachable_indices))
            else:
                # 全局探索: 完全随机
                action_scores[reachable_indices] = np.random.randn(len(reachable_indices))
        else:
            # 利用模式
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)[0]
                action_scores = q_values.cpu().numpy()
                action_scores[~reachable_mask] = -1e9

                valid_q = q_values[reachable_mask].cpu().numpy()
                if len(valid_q) > 0:
                    self.q_values_history.append(valid_q.mean())

        return action_scores

    def update_optimal_tracker(self, matching: Dict[int, int], step_reward: float):
        """更新最优点追踪器"""
        if len(matching) > 0:
            point_indices = list(matching.values())
            rewards = [step_reward / len(matching)] * len(matching)
            self.optimal_tracker.update(point_indices, rewards)

    def learn(self, batch: List[Transition], weights: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        从经验回放中学习(改进版 - 带奖励归一化)
        Returns:
            loss, td_errors
        """
        batch_size = len(batch)

        # 解包batch
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)

        # 奖励归一化
        rewards_raw = [t.reward for t in batch]
        if self.use_reward_norm and self.reward_normalizer:
            rewards = torch.FloatTensor([self.reward_normalizer.normalize(r) for r in rewards_raw]).to(self.device)
        else:
            rewards = torch.FloatTensor(rewards_raw).to(self.device)

        dones = torch.FloatTensor([t.done for t in batch]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # 提取动作和mask
        actions = []
        next_masks = []
        for t in batch:
            action_scores = t.action
            reachable_mask = t.reachable_mask

            # 找出实际选择的动作
            masked_scores = action_scores.copy()
            masked_scores[~reachable_mask] = -1e9
            selected_action = np.argmax(masked_scores)
            actions.append(selected_action)
            next_masks.append(reachable_mask)

        actions = torch.LongTensor(actions).to(self.device)

        # 计算当前Q值
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: 使用在线网络选择动作,目标网络评估
                next_q_online = self.q_network(next_states)
                next_q_target = self.target_network(next_states)

                # 为每个样本找最佳动作
                max_next_q = []
                for i in range(batch_size):
                    mask = torch.BoolTensor(next_masks[i]).to(self.device)

                    # 使用在线网络选择
                    masked_q_online = next_q_online[i].clone()
                    masked_q_online[~mask] = -1e9
                    best_action = masked_q_online.argmax()

                    # 使用目标网络评估
                    max_next_q.append(next_q_target[i, best_action])

                max_next_q = torch.stack(max_next_q)
            else:
                # 标准DQN
                next_q_values = self.target_network(next_states)
                max_next_q = []
                for i in range(batch_size):
                    mask = torch.BoolTensor(next_masks[i]).to(self.device)
                    masked_q = next_q_values[i].clone()
                    masked_q[~mask] = -1e9
                    max_next_q.append(masked_q.max())
                max_next_q = torch.stack(max_next_q)

            # 使用Huber损失的目标(减少异常值影响)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
            target_q = torch.clamp(target_q, -100, 100)  # 限制目标Q值范围

        # 计算TD误差
        td_errors = (current_q - target_q).detach().cpu().numpy()

        # 加权Huber损失
        loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (loss * weights).mean()

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)

        self.optimizer.step()
        self.scheduler.step()

        # 软更新目标网络
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self._soft_update_target_network(tau=0.005)

        return loss.item(), np.abs(td_errors)

    def _soft_update_target_network(self, tau: float = 0.005):
        """软更新目标网络"""
        for target_param, param in zip(self.target_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update_epsilon(self, step: int):
        """更新探索率"""
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-1.0 * step / self.epsilon_decay)

    def get_q_stats(self) -> Dict:
        """获取Q值统计信息"""
        if len(self.q_values_history) > 0:
            return {
                'mean_q': np.mean(self.q_values_history),
                'std_q': np.std(self.q_values_history),
                'max_q': np.max(self.q_values_history),
                'min_q': np.min(self.q_values_history)
            }
        return {'mean_q': 0, 'std_q': 0, 'max_q': 0, 'min_q': 0}

    def save(self, path: str):
        """保存模型"""
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter
        }

        # 保存奖励归一化器
        if self.reward_normalizer:
            checkpoint['reward_normalizer'] = {
                'mean': self.reward_normalizer.mean,
                'var': self.reward_normalizer.var,
                'count': self.reward_normalizer.count
            }

        torch.save(checkpoint, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epsilon = checkpoint['epsilon']
        self.learn_step_counter = checkpoint['learn_step_counter']

        # 加载奖励归一化器
        if 'reward_normalizer' in checkpoint and self.reward_normalizer:
            self.reward_normalizer.mean = checkpoint['reward_normalizer']['mean']
            self.reward_normalizer.var = checkpoint['reward_normalizer']['var']
            self.reward_normalizer.count = checkpoint['reward_normalizer']['count']


def train_improved_dqn(
        trajectory_file: str,
        region_file: str,
        dispatch_points_file: str,
        region_id: int = 0,
        num_episodes: int = 100,
        local_explore_prob: float = 0.7,  # 新增参数
        max_steps: int = 100,
        batch_size: int = 64,
        buffer_capacity: int = 100000,
        learning_start: int = 100,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000,
        target_update_freq: int = 100,
        save_freq: int = 100,
        state_dim: int = 128,
        use_double_dqn: bool = True,
        use_prioritized_replay: bool = True,
        use_reward_norm: bool = True,
        log_dir: str = './logs',
        model_dir: str = './models'
):
    """
    改进的DQN训练主函数 - 带进度条和详细统计
    """
    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 初始化TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(log_dir, f'run_{timestamp}'))

    # 初始化数据加载器
    print("初始化数据加载器...")
    data_loader = DataLoader(
        trajectory_file=trajectory_file,
        region_file=region_file,
        dispatch_points_file=dispatch_points_file
    )

    # 创建环境
    print(f"创建区域 {region_id} 的环境...")
    env = EdgeEnv(
        region_id=region_id,
        data_loader=data_loader,
        max_steps=max_steps,
        state_dim=state_dim,
        matching_method='hungarian'
    )

    # 创建智能体
    print("创建改进的DQN智能体...")
    agent = ImprovedDQNAgent(
        state_dim=state_dim,
        max_action_dim=env.num_dispatch_points,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
        use_double_dqn=use_double_dqn,
        use_reward_norm=use_reward_norm,
        local_explore_prob=local_explore_prob  # 新增
    )

    # 创建经验回放缓冲区
    if use_prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)
        print("✓ 使用优先级经验回放")
    else:
        replay_buffer = deque(maxlen=buffer_capacity)
        print("✓ 使用标准经验回放")

    if use_reward_norm:
        print("✓ 使用奖励归一化")
    if use_double_dqn:
        print("✓ 使用Double DQN")

    # 训练循环统计
    total_steps = 0
    best_reward = -float('inf')
    best_success_rate = 0.0
    reward_window = deque(maxlen=100)
    success_rate_window = deque(maxlen=100)
    wait_time_window = deque(maxlen=100)

    print("\n" + "=" * 100)
    print(f"{'开始训练 - 区域 ' + str(region_id):^100}")
    print("=" * 100)
    print(
        f"{'Episode':<10}{'Reward':<12}{'Avg(100)':<12}{'Success%':<12}{'AvgWait(min)':<15}{'ε':<10}")
    print("-" * 100)

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_reward_raw = 0  # 未归一化的奖励
        episode_losses = []

        for step in range(max_steps):
            # 获取当前可达点信息
            point_result = env._extract_point_features()
            reachable_indices = point_result.reachable_indices

            # 构建reachable mask
            reachable_mask = np.zeros(env.num_dispatch_points, dtype=bool)
            if len(reachable_indices) > 0:
                reachable_mask[reachable_indices] = True

            # 选择动作
            action = agent.select_action(state, reachable_mask)

            # 执行动作
            next_state, reward, done, info = env.step(action)
            matching = info.get('matching', {})
            agent.update_optimal_tracker(matching, reward)
            # 更新奖励归一化器
            if agent.reward_normalizer:
                agent.reward_normalizer.update(reward)

            episode_reward_raw += reward

            # 存储经验
            if use_prioritized_replay:
                replay_buffer.push(state, action, next_state, reward, done, reachable_mask)
            else:
                replay_buffer.append(Transition(state, action, next_state, reward, done, reachable_mask))

            state = next_state
            total_steps += 1

            # 开始学习
            if len(replay_buffer) >= learning_start and len(replay_buffer) >= batch_size:
                if use_prioritized_replay:
                    batch, indices, weights = replay_buffer.sample(batch_size)
                    loss, td_errors = agent.learn(batch, weights)
                    replay_buffer.update_priorities(indices, td_errors + 1e-6)
                else:
                    batch = random.sample(replay_buffer, batch_size)
                    weights = np.ones(batch_size)
                    loss, _ = agent.learn(batch, weights)

                episode_losses.append(loss)

                # 记录训练指标
                if total_steps % 10 == 0:
                    writer.add_scalar('Training/Loss', loss, total_steps)
                    writer.add_scalar('Training/Epsilon', agent.epsilon, total_steps)
                    writer.add_scalar('Training/Learning_Rate',
                                      agent.optimizer.param_groups[0]['lr'], total_steps)

                    q_stats = agent.get_q_stats()
                    writer.add_scalar('Q_Values/Mean', q_stats['mean_q'], total_steps)
                    writer.add_scalar('Q_Values/Std', q_stats['std_q'], total_steps)

                    # 记录奖励归一化统计
                    if agent.reward_normalizer:
                        reward_stats = agent.reward_normalizer.get_stats()
                        writer.add_scalar('Reward/Mean', reward_stats['mean'], total_steps)
                        writer.add_scalar('Reward/Std', reward_stats['std'], total_steps)

            agent.update_epsilon(total_steps)

            if done:
                break

        # Episode统计
        episode_reward = episode_reward_raw  # 使用原始奖励显示
        reward_window.append(episode_reward)
        avg_reward_100 = np.mean(reward_window)
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0

        # 计算性能指标
        stats = info['episode_stats']
        served = stats.get('served_requests', 0)
        failed = stats.get('failed_requests', 0)
        total_requests = served + failed
        success_rate = (served / total_requests * 100) if total_requests > 0 else 0

        # 计算平均等待时间
        if stats.get('wait_time_count', 0) > 0:
            avg_wait_time = stats['total_wait_time'] / stats['wait_time_count']
        else:
            avg_wait_time = 0.0

        success_rate_window.append(success_rate)
        wait_time_window.append(avg_wait_time)
        avg_success_rate_100 = np.mean(success_rate_window)
        avg_wait_time_100 = np.mean(wait_time_window)

        # 记录到TensorBoard
        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/Avg_Reward_100', avg_reward_100, episode)
        writer.add_scalar('Episode/Avg_Loss', avg_loss, episode)
        writer.add_scalar('Performance/Success_Rate', success_rate, episode)
        writer.add_scalar('Performance/Avg_Success_Rate_100', avg_success_rate_100, episode)
        writer.add_scalar('Performance/Avg_Wait_Time', avg_wait_time, episode)
        writer.add_scalar('Performance/Avg_Wait_Time_100', avg_wait_time_100, episode)
        writer.add_scalar('Performance/Served_Requests', served, episode)
        writer.add_scalar('Performance/Failed_Requests', failed, episode)

        # MCS统计
        total_mcs_income = sum(mcs.income for mcs in env.mcs_list)
        total_mcs_distance = sum(mcs.total_distance_traveled for mcs in env.mcs_list)
        busy_mcs = sum(1 for mcs in env.mcs_list if mcs.status.value != 'idle')
        mcs_utilization = busy_mcs / len(env.mcs_list) * 100

        writer.add_scalar('MCS/Total_Income', total_mcs_income, episode)
        writer.add_scalar('MCS/Total_Distance', total_mcs_distance, episode)
        writer.add_scalar('MCS/Utilization', mcs_utilization, episode)

        print(f"{episode}/{num_episodes}"
              f"{episode_reward:>10.2f} "
              f"{avg_reward_100:>10.2f} {success_rate:>10.1f}% "
              f"{avg_wait_time:>13.2f} {agent.epsilon:>8.3f} ")

        # 保存最佳模型（基于奖励）
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_path = os.path.join(model_dir, f'best_reward_region_{region_id}.pth')
            agent.save(best_model_path)
            print(f"\n{'':>10}✓ 新最佳奖励模型! Avg Reward: {best_reward:.2f}")

        # 定期保存检查点
        if episode % save_freq == 0:
            checkpoint_path = os.path.join(model_dir, f'checkpoint_ep{episode}_region_{region_id}.pth')
            agent.save(checkpoint_path)
            print(f"\n{'':>10}💾 检查点已保存: Episode {episode}")

        # 每100个episode输出详细统计
        if episode % 100 == 0:
            print("\n" + "-" * 100)
            print(f"Episode {episode} 统计摘要:")
            print(
                f"  平均奖励(100): {avg_reward_100:.2f} | 成功率(100): {avg_success_rate_100:.1f}% | 等待时间(100): {avg_wait_time_100:.2f}min")
            print(
                f"  MCS总收入: ¥{total_mcs_income:.2f} | MCS利用率: {mcs_utilization:.1f}% | 总行驶距离: {total_mcs_distance:.2f}km")
            if agent.reward_normalizer:
                reward_stats = agent.reward_normalizer.get_stats()
                print(f"  奖励统计: μ={reward_stats['mean']:.2f}, σ={reward_stats['std']:.2f}")
            q_stats = agent.get_q_stats()
            print(f"  Q值统计: μ={q_stats['mean_q']:.2f}, σ={q_stats['std_q']:.2f}")
            print("-" * 100)

    # 保存最终模型
    print()  # 换行
    final_model_path = os.path.join(model_dir, f'final_model_region_{region_id}.pth')
    agent.save(final_model_path)

    writer.close()

    print("\n" + "=" * 100)
    print("训练完成!")
    print(f"最佳平均奖励: {best_reward:.2f}")
    print(f"最佳平均成功率: {best_success_rate:.1f}%")
    print(f"最终模型: {final_model_path}")
    print(f"TensorBoard: tensorboard --logdir={log_dir}")
    print("=" * 100)

    return agent, env


if __name__ == '__main__':
    # 配置参数
    TRAJECTORY_FILE = 'dataset/top1000evs/reallocated/20140818_processed.csv'
    REGION_FILE = 'dataset/fcs_voronoi_regions.geojson'
    DISPATCH_POINTS_FILE = 'dataset/dispatch_points_400.csv'

    agent, env = train_improved_dqn(
        trajectory_file=TRAJECTORY_FILE,
        region_file=REGION_FILE,
        dispatch_points_file=DISPATCH_POINTS_FILE,
        region_id=9,
        num_episodes=500,
        max_steps=100,
        batch_size=128,
        buffer_capacity=50000,
        learning_start=500,
        lr=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=15000,
        target_update_freq=500,
        save_freq=100,
        state_dim=128,
        use_double_dqn=True,
        use_prioritized_replay=True,
        use_reward_norm=False,  # 启用奖励归一化
        log_dir='./logs',
        model_dir='./models'
    )