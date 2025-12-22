from datetime import datetime

import torch
import numpy as np
import os

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from a2c import ActorCritic
from params_config import Config
from dataloader import DataLoader, DataLoaderFactory, RegionManager
from edge_model_manager import EdgeModelManager
from cloud_env import HierarchicalCloudEnv, CloudSchedulerEnv
from edge_env import EdgeEnv

config = Config()


class CloudPPOTrainer:
    """云端PPO训练器"""

    def __init__(self, env: CloudSchedulerEnv, lr: float = 1e-4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = device

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]

        self.model = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 超参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5

        # 缓冲区
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'dones': []
        }

    def select_action(self, state: np.ndarray, training: bool = True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits, value = self.model(state_tensor)

        if training:
            noise = torch.randn_like(action_logits) * 0.1
            action_logits = action_logits + noise

        action = action_logits.squeeze(0).cpu().numpy()
        value = value.item()

        return action, value

    def store_transition(self, state, action, reward, value, done):
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['dones'].append(done)

    def update(self, next_state: np.ndarray, n_epochs: int = 4):
        if len(self.buffer['states']) < 2:
            self.clear_buffer()
            return {}

        # 计算GAE
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.model(next_state_tensor)
            next_value = next_value.item()

        advantages, returns = self._compute_gae(next_value)

        # 转换为tensor
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # 标准化advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / advantages.std()

        # 获取旧策略
        with torch.no_grad():
            old_action_logits, old_values = self.model(states)

        # 多轮更新
        total_loss = 0
        n_updates = 0

        for _ in range(n_epochs):
            action_logits, values = self.model(states)

            # 策略损失
            adv_expanded = advantages.unsqueeze(-1).expand_as(action_logits)
            policy_loss = nn.MSELoss()(
                action_logits,
                old_action_logits + adv_expanded * 0.1
            )

            # 值函数损失
            values = values.squeeze(-1)
            old_values = old_values.squeeze(-1)
            values_clipped = old_values + torch.clamp(
                values - old_values, -self.clip_epsilon, self.clip_epsilon
            )
            value_loss1 = nn.MSELoss()(values, returns)
            value_loss2 = nn.MSELoss()(values_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)

            # 熵正则
            entropy = action_logits.std()

            # 总损失
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # 更新
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            n_updates += 1

        self.clear_buffer()

        return {'loss': total_loss / n_updates if n_updates > 0 else 0}

    def _compute_gae(self, next_value: float):
        advantages = []
        gae = 0
        values = self.buffer['values'] + [next_value]

        for t in reversed(range(len(self.buffer['rewards']))):
            delta = (self.buffer['rewards'][t] +
                     self.gamma * values[t + 1] * (1 - self.buffer['dones'][t]) -
                     values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - self.buffer['dones'][t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.buffer['values'])]
        return advantages, returns

    def clear_buffer(self):
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'dones': []
        }


def train_hierarchical_cloud(
        factory: DataLoaderFactory,
        edge_model_dir: str = './models',
        n_episodes: int = 500,
        max_steps: int = 100,
        log_dir: str = './runs/cloud',
        save_dir: str = './models/cloud',
        num_regions: int = 18,
):
    """使用边缘模型训练云端调度器 - 增强版监控"""

    print("=" * 80)
    print("🚀 开始分层云端训练")
    print("=" * 80)

    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 2. 创建临时边缘环境以获取状态/动作维度
    print("\n🔍 获取边缘环境维度...")
    region_dict = []
    for region_id in range(num_regions):
        temp_env = EdgeEnv(region_id, factory, max_steps)
        edge_state_dim = temp_env.observation_space.shape[0]
        edge_action_dim = temp_env.action_space.n if hasattr(temp_env.action_space, "n") else temp_env.action_space.shape[0]
        region_dict.append({
            'state_dim': edge_state_dim,
            'action_dim': edge_action_dim,
        })

    # 3. 加载边缘模型
    print(f"\n📦 正在从 {edge_model_dir} 加载边缘模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    edge_manager = EdgeModelManager(num_regions, edge_model_dir, device)
    loaded_count = edge_manager.load_all_models(region_dict)

    if loaded_count == 0:
        print("\n❌ 错误: 未能加载任何边缘模型!")
        print("请先训练边缘模型或检查模型路径")
        return None

    print(f"\n✅ 成功加载 {loaded_count}/{num_regions} 个边缘模型")

    # 4. 创建分层云端环境
    print("\n☁️  创建分层云端环境...")
    env = HierarchicalCloudEnv(
        edge_model_manager=edge_manager,
        num_regions=num_regions,
        max_steps=max_steps
    )

    # 5. 创建云端训练器
    print(f"\n🎓 创建云端训练器 (设备: {device})...")
    trainer = CloudPPOTrainer(env, lr=1e-4, device=device)

    # 6. TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'{log_dir}/hierarchical_{timestamp}')

    # 7. 训练循环
    print("\n" + "=" * 80)
    print("🎯 开始云端训练")
    print("=" * 80)

    best_reward = -float('inf')
    best_success_rate = 0.0

    # 用于计算滑动平均的缓冲区
    from collections import deque
    reward_buffer = deque(maxlen=50)
    success_rate_buffer = deque(maxlen=50)
    wait_time_buffer = deque(maxlen=50)
    income_buffer = deque(maxlen=50)

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_cloud_reward = 0
        episode_edge_reward = 0

        # 收集边缘环境的详细统计
        edge_stats = {
            'total_served': 0,
            'total_failed': 0,
            'total_wait_time': 0.0,
            'wait_time_count': 0,
            'total_income': 0.0,
            'mcs_served': 0,
            'fcs_served': 0
        }

        for step in range(max_steps):
            # 云端选择跨区域调度动作
            action, value = trainer.select_action(state)

            # 执行(边缘模型会自动被调用)
            next_state, reward, done, info = env.step(action)

            # 存储转换
            trainer.store_transition(state, action, reward, value, done)

            episode_reward += reward
            episode_cloud_reward += info['cloud_reward']
            episode_edge_reward += info['edge_reward']

            state = next_state

            if done:
                break

        # 收集所有边缘环境的统计信息
        for region_id, edge_env in env.edge_envs.items():
            stats = edge_env.episode_stats
            edge_stats['total_served'] += stats['served_requests']
            edge_stats['total_failed'] += stats['failed_requests']
            edge_stats['total_wait_time'] += stats['total_wait_time']
            edge_stats['wait_time_count'] += stats['wait_time_count']
            edge_stats['total_income'] += sum(mcs.income for mcs in edge_env.mcs_list)
            edge_stats['mcs_served'] += stats['mcs_served']
            edge_stats['fcs_served'] += stats['fcs_served']

        # 计算汇总指标
        total_requests = edge_stats['total_served'] + edge_stats['total_failed']
        success_rate = (edge_stats['total_served'] / total_requests * 100) if total_requests > 0 else 0
        avg_wait_time = (edge_stats['total_wait_time'] / edge_stats['wait_time_count']) if edge_stats[
                                                                                               'wait_time_count'] > 0 else 0
        total_income = edge_stats['total_income']

        # 更新缓冲区
        reward_buffer.append(episode_reward)
        success_rate_buffer.append(success_rate)
        wait_time_buffer.append(avg_wait_time)
        income_buffer.append(total_income)

        # 定期更新
        if (episode + 1) % 10 == 0:
            update_info = trainer.update(state)
            if update_info:
                writer.add_scalar('Loss/total', update_info['loss'], episode)

        # 记录到TensorBoard - 基础指标
        writer.add_scalar('Episode/total_reward', episode_reward, episode)
        writer.add_scalar('Episode/cloud_reward', episode_cloud_reward, episode)
        writer.add_scalar('Episode/edge_reward', episode_edge_reward, episode)

        # 记录边缘环境聚合指标
        writer.add_scalar('Edge/success_rate', success_rate, episode)
        writer.add_scalar('Edge/total_served', edge_stats['total_served'], episode)
        writer.add_scalar('Edge/total_failed', edge_stats['total_failed'], episode)
        writer.add_scalar('Edge/avg_wait_time', avg_wait_time, episode)
        writer.add_scalar('Edge/total_income', total_income, episode)
        writer.add_scalar('Edge/mcs_served', edge_stats['mcs_served'], episode)
        writer.add_scalar('Edge/fcs_served', edge_stats['fcs_served'], episode)

        # 记录滑动平均
        if len(reward_buffer) > 0:
            writer.add_scalar('Average/reward', np.mean(reward_buffer), episode)
            writer.add_scalar('Average/success_rate', np.mean(success_rate_buffer), episode)
            writer.add_scalar('Average/wait_time', np.mean(wait_time_buffer), episode)
            writer.add_scalar('Average/income', np.mean(income_buffer), episode)

        # 云端调度指标
        writer.add_scalar('Cloud/dispatches', env.episode_stats['total_dispatches'], episode)
        writer.add_scalar('Cloud/total_distance', env.episode_stats['total_distance'], episode)

        # 计算负载均衡得分
        mcs_counts = [len(ids) for ids in env.region_mcs_mapping.values()]
        balance_score = -np.std(mcs_counts)
        writer.add_scalar('Metrics/mcs_balance_score', balance_score, episode)
        writer.add_scalar('Metrics/mcs_std', np.std(mcs_counts), episode)
        writer.add_scalar('Metrics/avg_mcs_per_region', np.mean(mcs_counts), episode)

        # 实时控制台输出 - 简洁单行模式
        avg_mcs_income = total_income / (num_regions * config.MCS_PER_REGION) if num_regions > 0 else 0

        print(f"Ep {episode + 1}/{n_episodes} | "
              f"Reward: {episode_reward:>6.2f} | "
              f"Success: {success_rate:>5.1f}% | "
              f"Dispatch: {env.episode_stats['total_dispatches']:>3d} | "
              f"Wait: {avg_wait_time:>5.1f}min | "
              f"AvgIncome: {avg_mcs_income:>6.2f}")

        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'episode': episode,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_reward': best_reward,
                'success_rate': success_rate,
                'avg_wait_time': avg_wait_time,
                'total_income': total_income
            }, f'{save_dir}/best_cloud_model.pth')
            print(f"\n  ✨ 保存最佳奖励模型 (奖励: {best_reward:.2f})")

        if success_rate > best_success_rate:
            best_success_rate = success_rate
            torch.save({
                'episode': episode,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'success_rate': success_rate,
                'avg_wait_time': avg_wait_time,
                'total_income': total_income
            }, f'{save_dir}/best_success_model.pth')
            print(f"  🎯 保存最佳成功率模型 (成功率: {success_rate:.2f}%)")

    # 训练完成总结
    print("\n" + "=" * 80)
    print("🎉 云端训练完成!")
    print("=" * 80)
    print(f"\n📈 最终统计 (最后50轮平均):")
    print(f"  平均奖励:     {np.mean(reward_buffer):.2f}")
    print(f"  平均成功率:   {np.mean(success_rate_buffer):.2f}%")
    print(f"  平均等待时间: {np.mean(wait_time_buffer):.2f} 分钟")
    print(f"  平均收入:     {np.mean(income_buffer):.2f} 元")
    print(f"\n🏆 最佳记录:")
    print(f"  最佳奖励:     {best_reward:.2f}")
    print(f"  最佳成功率:   {best_success_rate:.2f}%")
    print("=" * 80)

    writer.close()
    return trainer, env
def evaluate_cloud_model(
        model_path: str,
        edge_model_dir: str,
        n_episodes: int = 10,
        max_steps: int = 100
):
    """评估云端模型"""
    print("\n🔬 评估云端模型...")

    # 加载环境
    factory = DataLoaderFactory(
        trajectory_file='dataset/top1000evs/reallocated/20140818_processed.csv',
        region_file='dataset/fcs_voronoi_regions.geojson',
        dispatch_file='dataset/dispatch_points_400.csv'
    )
    region_dict = []
    for region_id in range(18):
        temp_env = EdgeEnv(region_id, factory, max_steps)
        edge_state_dim = temp_env.observation_space.shape[0]
        edge_action_dim = temp_env.action_space.n if hasattr(temp_env.action_space, "n") else temp_env.action_space.shape[0]
        region_dict.append(
            {
                'state_dim': edge_state_dim,
                'action_dim': edge_action_dim,
            }
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    edge_manager = EdgeModelManager(18, edge_model_dir, device)
    edge_manager.load_all_models(region_dict)

    env = HierarchicalCloudEnv(edge_manager, max_steps=max_steps)

    # 加载云端模型
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    model = ActorCritic(obs_dim, action_dim).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 评估
    rewards = []
    dispatches = []
    distances = []
    balance_scores = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _ = model(state_tensor)
            action = action.squeeze(0).cpu().numpy()

            state, reward, done, info = env.step(action)
            episode_reward += reward

            if done:
                break

        mcs_counts = [len(ids) for ids in env.region_mcs_mapping.values()]
        balance_score = -np.std(mcs_counts)

        rewards.append(episode_reward)
        dispatches.append(env.episode_stats['total_dispatches'])
        distances.append(env.episode_stats['total_distance'])
        balance_scores.append(balance_score)

    print("\n📊 评估结果:")
    print(f"  平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  平均调度次数: {np.mean(dispatches):.1f}")
    print(f"  平均移动距离: {np.mean(distances):.1f} km")
    print(f"  平均负载均衡得分: {np.mean(balance_scores):.2f}")


if __name__ == '__main__':
    # 配置
    EDGE_MODEL_DIR = './models'  # 边缘模型目录
    factory = DataLoaderFactory(
        trajectory_file='dataset/top1000evs/reallocated/20140818_processed.csv',
        region_file='dataset/fcs_voronoi_regions.geojson',
        dispatch_file='dataset/dispatch_points_400.csv'
    )
    # 训练云端调度器
    trainer, env = train_hierarchical_cloud(
        factory=factory,
        edge_model_dir=EDGE_MODEL_DIR,
        n_episodes=500,
        max_steps=100,
        num_regions=18
    )

    # 评估
    if trainer is not None:
        evaluate_cloud_model(
            model_path='./models/cloud/best_cloud_model.pth',
            edge_model_dir=EDGE_MODEL_DIR,
            n_episodes=10
        )
