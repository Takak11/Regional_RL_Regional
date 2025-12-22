"""
PPO (Proximal Policy Optimization) 训练器
相比DQN更稳定，更适合连续动作空间
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from typing import List, Tuple, Dict
from collections import deque
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from dataloader import DataLoaderFactory
from params_config import Config
from edge_env import EdgeEnv

config = Config()


# ============================================================
# Actor-Critic 网络
# ============================================================
class ActorCriticNetwork(nn.Module):
    """Actor-Critic网络 - 共享特征提取"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCriticNetwork, self).__init__()

        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Actor头 - 输出动作分数
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Critic头 - 输出状态价值
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """正交初始化提升稳定性"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        """前向传播"""
        features = self.shared_layers(state)
        action_scores = self.actor_head(features)
        state_value = self.critic_head(features)
        return action_scores, state_value

    def get_action(self, state, reachable_mask=None, deterministic=False):
        """
        获取动作

        Args:
            state: 状态
            reachable_mask: 可达掩码 (bool array)
            deterministic: 是否确定性输出
        """
        action_scores, value = self.forward(state)

        if reachable_mask is not None and reachable_mask.shape[-1] == action_scores.shape[-1]:
            # 对不可达的点设置极小值（仅在维度匹配时启用）
            reachable_mask = torch.FloatTensor(reachable_mask).to(action_scores.device)
            action_scores = action_scores * reachable_mask + (1 - reachable_mask) * (-1e9)

        if deterministic:
            # 确定性：直接返回最高分数
            return action_scores, value
        else:
            # 随机性：添加高斯噪声
            noise = torch.randn_like(action_scores) * 0.1
            action_scores = action_scores + noise
            return action_scores, value


# ============================================================
# PPO 训练器
# ============================================================
class PPOTrainer:
    """PPO训练器"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # 创建网络
        self.network = ActorCriticNetwork(state_dim, action_dim).to(device)

        # 优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.95
        )

        # 经验缓冲区
        self.reset_buffer()

    def reset_buffer(self):
        """重置经验缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []

    def select_action(self, state, reachable_mask=None, deterministic=False):
        """
        选择动作

        Returns:
            action: 动作
            value: 状态价值
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_scores, value = self.network.get_action(
                state_tensor, reachable_mask, deterministic
            )

        logits = action_scores.squeeze(0)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = torch.zeros_like(action, dtype=torch.float32)
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        action = action.cpu().item()
        log_prob = log_prob.cpu().item()
        value = value.item()

        return action, value, log_prob

    def store_transition(self, state, action, reward, value, done, log_prob):
        """存储转换"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def compute_gae(self, next_value):
        """
        计算广义优势估计 (GAE)

        Args:
            next_value: 下一个状态的价值
        """
        advantages = []
        gae = 0

        values = self.values + [next_value]

        # 从后向前计算GAE
        for t in reversed(range(len(self.rewards))):
            delta = (self.rewards[t] +
                     self.gamma * values[t + 1] * (1 - self.dones[t]) -
                     values[t])

            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        # 计算回报
        returns = [adv + val for adv, val in zip(advantages, self.values)]

        return advantages, returns

    def update(self, next_state, n_epochs=4, batch_size=64):
        """
        PPO更新

        Args:
            next_state: 最后一个状态
            n_epochs: 更新轮数
            batch_size: 批量大小
        """
        if len(self.states) < 2:
            self.reset_buffer()
            return {}

        # 计算下一个状态的价值
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.network(next_state_tensor)
            next_value = next_value.item()

        # 计算GAE和回报
        advantages, returns = self.compute_gae(next_value)

        # 转换为tensor
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        n_updates = 0

        for epoch in range(n_epochs):
            # 小批量更新
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)

            for start in range(0, len(self.states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # 获取批量数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # 前向传播
                action_scores, values = self.network(batch_states)
                values = values.squeeze(-1)
                dist = Categorical(logits=action_scores)
                log_probs = dist.log_prob(batch_actions)

                # 策略损失 (PPO剪切比率)
                ratios = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # 值函数损失
                value_pred_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss1 = (values - batch_returns).pow(2)
                value_loss2 = (value_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.mean(torch.max(value_loss1, value_loss2))

                # 熵正则化（鼓励探索）
                entropy = dist.entropy().mean()
                entropy_loss = -entropy

                # 总损失
                loss = (policy_loss +
                        self.value_coef * value_loss +
                        self.entropy_coef * entropy_loss)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # 累积损失
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                n_updates += 1

        # 更新学习率
        self.scheduler.step()

        # 清空缓冲区
        self.reset_buffer()

        # 返回统计信息
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy_loss': total_entropy_loss / n_updates,
            'total_loss': total_loss / n_updates,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    def save(self, path):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


# ============================================================
# 训练函数
# ============================================================
def train_ppo(
        trajectory_file: str,
        region_file: str,
        dispatch_points_file: str,
        region_id: int = 0,
        num_episodes: int = 500,
        max_steps: int = 100,
        update_interval: int = 10,  # 每N个episode更新一次
        n_epochs: int = 4,
        batch_size: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        log_dir: str = './logs',
        model_dir: str = './models',
        seed: int = config.random_seed
):
    """
    使用PPO训练边缘调度器
    """
    print("=" * 80)
    print(f"{'PPO训练 - 区域 ' + str(region_id):^80}")
    print("=" * 80)

    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 创建环境
    print("\n初始化环境...")
    factory = DataLoaderFactory(
        trajectory_file=trajectory_file,
        region_file=region_file,
        dispatch_file=dispatch_points_file
    )

    env = EdgeEnv(region_id=region_id, factory=factory, max_steps=max_steps)
    if seed is not None:
        env.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]

    print(f"✓ 状态维度: {state_dim}")
    print(f"✓ 动作维度: {action_dim}")

    # 创建训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n创建PPO训练器 (设备: {device})...")

    trainer = PPOTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        device=device
    )

    # TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'{log_dir}/ppo_region_{region_id}_{timestamp}')

    # 训练统计
    best_reward = -float('inf')
    best_success_rate = 0.0
    episode_rewards = deque(maxlen=100)
    episode_success_rates = deque(maxlen=100)
    episode_wait_times = deque(maxlen=100)

    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    print(f"{'Episode':<10}{'Reward':<12}{'Avg(100)':<12}{'Success%':<12}"
          f"{'Wait(min)':<12}{'Loss':<10}")
    print("-" * 80)

    # 训练循环
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0

        # 收集轨迹
        for step in range(max_steps):
            # 更新训练进度
            env.training_progress = min(1.0, episode / num_episodes)

            # 获取可达掩码
            point_result = env._extract_point_features()
            reachable_mask = np.zeros(env.num_dispatch_points, dtype=bool)
            if len(point_result.reachable_indices) > 0:
                reachable_mask[point_result.reachable_indices] = True

            # 选择动作
            action, value, log_prob = trainer.select_action(state, reachable_mask)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储转换
            trainer.store_transition(state, action, reward, value, done, log_prob)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break

        # 定期更新
        update_info = {}
        if episode % update_interval == 0:
            update_info = trainer.update(state, n_epochs=n_epochs, batch_size=batch_size)

        # 统计信息
        stats = info['episode_stats']
        served = stats['served_requests']
        failed = stats['failed_requests']
        pending = info['pending_requests']
        total_requests = served + failed + pending
        success_rate = (served / total_requests * 100) if total_requests > 0 else 0

        if stats['wait_time_count'] > 0:
            avg_wait_time = stats['total_wait_time'] / stats['wait_time_count']
        else:
            avg_wait_time = 0.0

        # 更新滑动窗口
        episode_rewards.append(episode_reward)
        episode_success_rates.append(success_rate)
        episode_wait_times.append(avg_wait_time)

        # 记录到TensorBoard
        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/Success_Rate', success_rate, episode)
        writer.add_scalar('Episode/Avg_Wait_Time', avg_wait_time, episode)
        writer.add_scalar('Episode/Served', served, episode)
        writer.add_scalar('Episode/Failed', failed, episode)

        if update_info:
            writer.add_scalar('Loss/Policy', update_info['policy_loss'], episode)
            writer.add_scalar('Loss/Value', update_info['value_loss'], episode)
            writer.add_scalar('Loss/Entropy', update_info['entropy_loss'], episode)
            writer.add_scalar('Loss/Total', update_info['total_loss'], episode)
            writer.add_scalar('Training/Learning_Rate', update_info['learning_rate'], episode)

        # 控制台输出
        avg_reward = np.mean(episode_rewards)
        loss_str = f"{update_info.get('total_loss', 0):.4f}" if update_info else "---"

        print(f"{episode:<10}{episode_reward:>10.2f}{avg_reward:>10.2f}"
              f"{success_rate:>10.1f}%{avg_wait_time:>10.2f}{loss_str:>10}")

        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            save_path = f'{model_dir}/best_model_region_{region_id}.pth'
            trainer.save(save_path)
            writer.add_scalar('Best/Reward', best_reward, episode)

        if success_rate > best_success_rate:
            best_success_rate = success_rate
            save_path = f'{model_dir}/best_success_region_{region_id}.pth'
            trainer.save(save_path)
            writer.add_scalar('Best/Success_Rate', best_success_rate, episode)

        # 定期保存检查点
        if episode % 100 == 0:
            checkpoint_path = f'{model_dir}/checkpoint_ep{episode}_region_{region_id}.pth'
            trainer.save(checkpoint_path)
            print(f"\n{'':>10}💾 检查点已保存")

            # 输出详细统计
            print(f"{'':>10}📊 最近100轮统计:")
            print(f"{'':>15}平均奖励: {np.mean(episode_rewards):.2f}")
            print(f"{'':>15}平均成功率: {np.mean(episode_success_rates):.1f}%")
            print(f"{'':>15}平均等待时间: {np.mean(episode_wait_times):.2f}分钟")
            print()

    # 训练完成
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"\n📈 最终统计 (最后100轮平均):")
    print(f"  平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"  平均成功率: {np.mean(episode_success_rates):.2f}%")
    print(f"  平均等待时间: {np.mean(episode_wait_times):.2f}分钟")
    print(f"\n🏆 最佳记录:")
    print(f"  最佳奖励: {best_reward:.2f}")
    print(f"  最佳成功率: {best_success_rate:.2f}%")
    print(f"\n💾 模型保存在: {model_dir}")
    print(f"📊 日志保存在: {log_dir}")
    print("=" * 80)

    writer.close()

    return trainer, env


# ============================================================
# 评估函数
# ============================================================
def evaluate_ppo(
        model_path: str,
        trajectory_file: str,
        region_file: str,
        dispatch_points_file: str,
        region_id: int = 0,
        n_episodes: int = 10,
        max_steps: int = 100
):
    """评估训练好的PPO模型"""
    print("\n" + "=" * 80)
    print(f"{'评估PPO模型 - 区域 ' + str(region_id):^80}")
    print("=" * 80)

    # 创建环境
    factory = DataLoaderFactory(
        trajectory_file=trajectory_file,
        region_file=region_file,
        dispatch_file=dispatch_points_file
    )

    env = EdgeEnv(region_id=region_id, factory=factory, max_steps=max_steps)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = PPOTrainer(state_dim=state_dim, action_dim=action_dim, device=device)
    trainer.load(model_path)

    print(f"✓ 模型已加载: {model_path}")
    print(f"\n开始评估 ({n_episodes} 轮)...")

    # 评估
    episode_rewards = []
    episode_success_rates = []
    episode_wait_times = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 获取可达掩码
            point_result = env._extract_point_features()
            reachable_mask = np.zeros(env.num_dispatch_points, dtype=bool)
            if len(point_result.reachable_indices) > 0:
                reachable_mask[point_result.reachable_indices] = True

            # 确定性动作
            action, value, log_prob = trainer.select_action(state, reachable_mask, deterministic=True)

            state, reward, done, info = env.step(action)
            episode_reward += reward

            if done:
                break

        # 统计
        stats = info['episode_stats']
        served = stats['served_requests']
        failed = stats['failed_requests']
        pending = info['pending_requests']
        total = served + failed + pending
        success_rate = (served / total * 100) if total > 0 else 0

        if stats['wait_time_count'] > 0:
            avg_wait = stats['total_wait_time'] / stats['wait_time_count']
        else:
            avg_wait = 0.0

        episode_rewards.append(episode_reward)
        episode_success_rates.append(success_rate)
        episode_wait_times.append(avg_wait)

        print(f"  Episode {episode + 1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Success={success_rate:.1f}%, "
              f"Wait={avg_wait:.2f}min")

    # 输出结果
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均成功率: {np.mean(episode_success_rates):.2f}% ± {np.std(episode_success_rates):.2f}%")
    print(f"平均等待时间: {np.mean(episode_wait_times):.2f} ± {np.std(episode_wait_times):.2f} 分钟")
    print("=" * 80)


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    # 配置
    TRAJECTORY_FILE = 'dataset/top1000evs/reallocated/20140818_processed.csv'
    REGION_FILE = 'dataset/fcs_regions.geojson'
    DISPATCH_POINTS_FILE = 'dataset/dispatch_points.csv'
    REGION_ID = 0

    # 训练
    trainer, env = train_ppo(
        trajectory_file=TRAJECTORY_FILE,
        region_file=REGION_FILE,
        dispatch_points_file=DISPATCH_POINTS_FILE,
        region_id=REGION_ID,
        num_episodes=1000,
        max_steps=100,
        update_interval=10,  # 每10个episode更新一次
        n_epochs=4,
        batch_size=256,
        lr=1e-4,
        log_dir='./logs',
        model_dir='./models'
    )

    # 评估
    print("\n" + "=" * 80)
    input("按Enter键开始评估...")

    evaluate_ppo(
        model_path=f'./models/best_model_region_{REGION_ID}.pth',
        trajectory_file=TRAJECTORY_FILE,
        region_file=REGION_FILE,
        dispatch_points_file=DISPATCH_POINTS_FILE,
        region_id=REGION_ID,
        n_episodes=10,
        max_steps=100
    )
