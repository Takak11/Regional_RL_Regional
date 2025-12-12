import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time
from datetime import datetime
from typing import Dict, List, Tuple

from edge_env import EdgeEnv
from dataloader import DataLoaderFactory
from params_config import Config
from a2c import ActorCritic

config = Config()


class PPOTrainer:
    def __init__(
            self,
            env: EdgeEnv,
            state_dim: int,
            action_dim: int,
            lr: float = 3e-5,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.env = env
        self.device = device

        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # 网络
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 经验缓冲
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state: np.ndarray, training: bool = True):
        """改进的动作选择 - 减少噪声"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_scores, value = self.model(state_tensor)

        # 自适应噪声 - 训练后期减少探索
        if training:
            # 根据训练进度调整噪声
            noise_scale = 0.1 * (1.0 - self.env.training_progress * 0.8)
            noise = torch.randn_like(action_scores) * noise_scale
            action_scores = action_scores + noise

        action = action_scores.squeeze(0).cpu().numpy()
        value = value.item()

        return action, value

    def store_transition(self, state, action, reward, value, done):
        """存储转换"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, next_value: float) -> Tuple[List[float], List[float]]:
        """计算GAE优势函数"""
        advantages = []
        gae = 0

        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]

        return advantages, returns

    def update(self, next_state: np.ndarray, n_epochs: int = 4, batch_size: int = 256):
        """改进的PPO更新 - 修复shape问题"""
        if len(self.states) == 0:
            return {}

        # 计算优势函数
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.model(next_state_tensor)
            next_value = next_value.item()

        advantages, returns = self.compute_gae(next_value)

        # *** FIX: 确保至少有2个样本才能计算std ***
        if len(advantages) < 2:
            # 如果样本太少，直接返回
            self.clear_buffer()
            return {
                'loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'kl': 0.0
            }

        # 转换为tensor
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(self.actions)).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # *** FIX: 安全的标准化 ***
        adv_std = advantages_tensor.std()
        if adv_std > 1e-8:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / adv_std
        else:
            # 如果std太小，只中心化不缩放
            advantages_tensor = advantages_tensor - advantages_tensor.mean()

        # 计算旧的动作概率
        with torch.no_grad():
            old_action_scores, old_values = self.model(states_tensor)

        # 多轮更新
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        n_updates = 0

        early_stop = False

        for epoch in range(n_epochs):
            if early_stop:
                break

            # 随机打乱数据
            indices = np.random.permutation(len(self.states))

            for start_idx in range(0, len(self.states), batch_size):
                end_idx = min(start_idx + batch_size, len(self.states))
                batch_indices = indices[start_idx:end_idx]

                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_old_scores = old_action_scores[batch_indices]
                batch_old_values = old_values[batch_indices]

                # 前向传播
                action_scores, values = self.model(batch_states)

                # *** FIX: 策略损失 - 确保维度匹配 ***
                # 扩展advantages维度以匹配action_scores
                adv_expanded = batch_advantages.unsqueeze(-1).expand_as(action_scores)
                policy_loss = nn.MSELoss()(
                    action_scores,
                    batch_old_scores + adv_expanded * 0.1
                )

                # *** FIX: 值函数损失 - 确保维度一致 ***
                values_squeezed = values.squeeze(-1)  # 确保是1D
                old_values_squeezed = batch_old_values.squeeze(-1)  # 确保是1D

                values_clipped = old_values_squeezed + torch.clamp(
                    values_squeezed - old_values_squeezed,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss_unclipped = nn.MSELoss()(values_squeezed, batch_returns)
                value_loss_clipped = nn.MSELoss()(values_clipped, batch_returns)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

                # 熵正则化
                action_std = action_scores.std(dim=-1).mean()
                entropy = action_std

                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                self.optimizer.step()

                # 计算KL散度
                with torch.no_grad():
                    new_action_scores, _ = self.model(batch_states)
                    kl = ((action_scores - new_action_scores) ** 2).mean()
                    total_kl += kl.item()

                # 记录
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

                # 早停检查
                if kl.item() > 0.02:
                    early_stop = True
                    break

        # 清空缓冲
        self.clear_buffer()

        return {
            'loss': total_loss / n_updates if n_updates > 0 else 0,
            'policy_loss': total_policy_loss / n_updates if n_updates > 0 else 0,
            'value_loss': total_value_loss / n_updates if n_updates > 0 else 0,
            'entropy': total_entropy / n_updates if n_updates > 0 else 0,
            'kl': total_kl / n_updates if n_updates > 0 else 0
        }

    def clear_buffer(self):
        """清空经验缓冲"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []


class MetricsTracker:
    """指标追踪器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_success_rates = deque(maxlen=window_size)
        self.episode_avg_wait_times = deque(maxlen=window_size)
        self.episode_incomes = deque(maxlen=window_size)
        self.episode_served = deque(maxlen=window_size)

    def add_episode(self, reward: float, success_rate: float,
                    avg_wait_time: float, income: float, served: int):
        """添加episode数据"""
        self.episode_rewards.append(reward)
        self.episode_success_rates.append(success_rate)
        self.episode_avg_wait_times.append(avg_wait_time)
        self.episode_incomes.append(income)
        self.episode_served.append(served)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_success_rate': np.mean(self.episode_success_rates) if self.episode_success_rates else 0,
            'avg_wait_time': np.mean(self.episode_avg_wait_times) if self.episode_avg_wait_times else 0,
            'avg_income': np.mean(self.episode_incomes) if self.episode_incomes else 0,
            'avg_served': np.mean(self.episode_served) if self.episode_served else 0
        }


def train(
        region_id: int = 0,
        trajectory_file: str = 'trajectories.csv',
        region_file: str = 'regions.geojson',
        dispatch_file: str = 'dispatch_points.csv',
        n_episodes: int = 1000,
        max_steps: int = 100,
        update_freq: int = 10,
        save_freq: int = 50,
        log_dir: str = './runs',
        model_dir: str = './models'
):

    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 初始化数据加载器
    print("正在加载数据...")
    # 创建环境
    print(f"正在创建区域 {region_id} 的环境...")
    factory = DataLoaderFactory(trajectory_file, region_file, dispatch_file)
    env = EdgeEnv(region_id=region_id, factory=factory, max_steps=max_steps)

    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")

    # 创建训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    trainer = PPOTrainer(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-4,
        device=device
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=n_episodes,
        eta_min=1e-5
    )

    # TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'{log_dir}/region_{region_id}_{timestamp}')

    # 指标追踪器
    metrics_tracker = MetricsTracker(window_size=100)

    # 最佳模型追踪
    best_success_rate = 0
    best_reward = -float('inf')

    print("\n开始训练...")
    print("=" * 80)

    start_time = time.time()

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0

        for step in range(max_steps):
            # 选择动作
            action, value = trainer.select_action(state, training=True)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储转换
            trainer.store_transition(state, action, reward, value, done)

            episode_reward += reward
            episode_steps += 1

            state = next_state

            if done:
                break

                # 定期更新网络
            if (episode + 1) % update_freq == 0:
                update_info = trainer.update(state, n_epochs=4, batch_size=256)

                # 更新学习率
                scheduler.step()

                # 记录学习率
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar('Training/learning_rate', current_lr, episode)

                # 记录训练损失
                if update_info:
                    writer.add_scalar('Loss/total', update_info['loss'], episode)
                    writer.add_scalar('Loss/policy', update_info['policy_loss'], episode)
                    writer.add_scalar('Loss/value', update_info['value_loss'], episode)
                    writer.add_scalar('Loss/entropy', update_info['entropy'], episode)
                    writer.add_scalar('Loss/kl_divergence', update_info['kl'], episode)

        # 收集episode指标
        success_rate = env.get_success_rate()
        avg_wait_time = env.get_avg_wait_time()
        served = env.get_success_served()

        # 计算总收入
        total_income = sum(mcs.income for mcs in env.mcs_list)

        # 添加到追踪器
        metrics_tracker.add_episode(
            episode_reward, success_rate, avg_wait_time, total_income, served
        )

        # 记录到TensorBoard
        writer.add_scalar('Episode/reward', episode_reward, episode)
        writer.add_scalar('Episode/success_rate', success_rate, episode)
        writer.add_scalar('Episode/avg_wait_time', avg_wait_time, episode)
        writer.add_scalar('Episode/income', total_income, episode)
        writer.add_scalar('Episode/served', served, episode)
        writer.add_scalar('Episode/steps', episode_steps, episode)

        # 记录滑动平均
        stats = metrics_tracker.get_stats()
        writer.add_scalar('Average/reward', stats['avg_reward'], episode)
        writer.add_scalar('Average/success_rate', stats['avg_success_rate'], episode)
        writer.add_scalar('Average/wait_time', stats['avg_wait_time'], episode)
        writer.add_scalar('Average/income', stats['avg_income'], episode)
        writer.add_scalar('Average/served', stats['avg_served'], episode)

        # 打印进度
        elapsed_time = time.time() - start_time
        episodes_per_sec = (episode + 1) / elapsed_time
        eta = (n_episodes - episode - 1) / episodes_per_sec if episodes_per_sec > 0 else 0

        print(f"Episode {episode + 1}/{n_episodes} | "
              f"Reward: {episode_reward:.3f} | "
              f"Success: {success_rate:.1f}% | "
              f"Wait: {avg_wait_time:.1f}min | "
              f"Income: {total_income:.1f} | "
              f"Served: {served} | "
              f"ETA: {eta / 60:.1f}min")

        # 保存最佳模型
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_reward = stats['avg_reward']
            torch.save({
                'episode': episode,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'success_rate': success_rate,
                'avg_reward': stats['avg_reward']
            }, f'{model_dir}/best_model_region_{region_id}.pth')
            print(f"✓ 保存最佳模型 (成功率: {success_rate:.2f}%)")

        # 定期保存检查点
        if (episode + 1) % save_freq == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'success_rate': success_rate,
                'avg_reward': stats['avg_reward']
            }, f'{model_dir}/checkpoint_region_{region_id}_ep{episode + 1}.pth')
            print(f"✓ 保存检查点 (Episode {episode + 1})")

    # 训练结束
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("训练完成!")
    print(f"总时间: {total_time / 60:.1f} 分钟")
    print(f"最佳成功率: {best_success_rate:.2f}%")
    print(f"最佳平均奖励: {best_reward:.3f}")

    # 最终统计
    final_stats = metrics_tracker.get_stats()
    print("\n最终统计 (最近100轮平均):")
    print(f"  平均奖励: {final_stats['avg_reward']:.3f}")
    print(f"  平均成功率: {final_stats['avg_success_rate']:.2f}%")
    print(f"  平均等待时间: {final_stats['avg_wait_time']:.2f} 分钟")
    print(f"  平均收入: {final_stats['avg_income']:.2f}")
    print(f"  平均服务数: {final_stats['avg_served']:.1f}")

    writer.close()

    return trainer, final_stats


def monitor_gradients(model, writer, episode):
    """监控梯度统计"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    writer.add_scalar('Training/gradient_norm', total_norm, episode)
    return total_norm


if __name__ == '__main__':
    # 训练配置
    REGION_ID = 0
    TRAJECTORY_FILE = 'dataset/top1000evs/reallocated/20140818_processed.csv'
    REGION_FILE = 'dataset/fcs_voronoi_regions.geojson'
    DISPATCH_FILE = 'dataset/dispatch_points_400.csv'

    # 训练
    trainer, stats = train(
        region_id=REGION_ID,
        trajectory_file=TRAJECTORY_FILE,
        region_file=REGION_FILE,
        dispatch_file=DISPATCH_FILE,
        n_episodes=1000,
        max_steps=100,
        update_freq=20,
        save_freq=50
    )

    print("\n查看训练过程:")
    print("运行命令: tensorboard --logdir=./runs")