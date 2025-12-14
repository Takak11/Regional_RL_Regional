"""
多日数据训练策略
支持利用7天轨迹数据进行训练
"""

import torch
import numpy as np
import os
from typing import List, Dict, Tuple
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataloader import DataLoaderFactory
from edge_env import EdgeEnv
from train import PPOTrainer
from params_config import Config

config = Config()


class MultiDayDataManager:
    """多日数据管理器"""

    def __init__(self,
                 trajectory_files: List[str],
                 region_file: str,
                 dispatch_file: str,
                 dates: List[str] = None):
        """
        Args:
            trajectory_files: 轨迹文件列表
            region_file: 区域文件
            dispatch_file: 调度点文件
            dates: 日期标识列表（可选，用于记录）
        """
        self.trajectory_files = trajectory_files
        self.region_file = region_file
        self.dispatch_file = dispatch_file
        self.dates = dates or [f"day_{i}" for i in range(len(trajectory_files))]

        print(f"多日数据管理器初始化:")
        print(f"  - 数据文件数量: {len(trajectory_files)}")
        print(f"  - 日期: {', '.join(self.dates)}")

        # 为每个数据文件创建工厂
        self.factories = []
        for i, traj_file in enumerate(trajectory_files):
            print(f"  - 加载 {self.dates[i]}: {traj_file}")
            factory = DataLoaderFactory(
                trajectory_file=traj_file,
                region_file=region_file,
                dispatch_file=dispatch_file
            )
            self.factories.append(factory)

        print(f"✓ {len(self.factories)} 个数据工厂已创建")

    def create_env(self, region_id: int, max_steps: int, day_index: int = None):
        """
        创建环境

        Args:
            region_id: 区域ID
            max_steps: 最大步数
            day_index: 指定使用哪一天的数据（None则随机）
        """
        if day_index is None:
            day_index = np.random.randint(0, len(self.factories))

        factory = self.factories[day_index]
        env = EdgeEnv(region_id=region_id, factory=factory, max_steps=max_steps)

        return env, day_index

    def get_day_count(self):
        """获取总天数"""
        return len(self.factories)


class CurriculumScheduler:
    """课程学习调度器 - 控制数据难度"""

    def __init__(self, num_days: int, warmup_episodes: int = 50):
        """
        Args:
            num_days: 总天数
            warmup_episodes: 预热轮数（只用第一天）
        """
        self.num_days = num_days
        self.warmup_episodes = warmup_episodes
        self.current_episode = 0

    def get_day_distribution(self) -> np.ndarray:
        """
        获取当前应该使用的日期分布

        Returns:
            概率分布 (num_days,)
        """
        if self.current_episode < self.warmup_episodes:
            # 预热阶段：只用第一天
            probs = np.zeros(self.num_days)
            probs[0] = 1.0
        else:
            # 线性增加日期多样性
            progress = min(1.0, (self.current_episode - self.warmup_episodes) / 200)

            if progress < 0.5:
                # 前期：主要用前几天
                probs = np.array([0.4, 0.3, 0.2, 0.1] + [0.0] * (self.num_days - 4))
            else:
                # 后期：均匀分布
                probs = np.ones(self.num_days) / self.num_days

        return probs / probs.sum()

    def sample_day(self) -> int:
        """采样一天"""
        probs = self.get_day_distribution()
        return np.random.choice(self.num_days, p=probs)

    def step(self):
        """进入下一轮"""
        self.current_episode += 1


def train_ppo_multiday(
        trajectory_files: List[str],
        region_file: str,
        dispatch_points_file: str,
        dates: List[str] = None,
        region_id: int = 0,
        num_episodes: int = 500,
        max_steps: int = 100,
        update_interval: int = 10,
        strategy: str = 'curriculum',  # 'curriculum', 'mixed', 'sequential'
        lr: float = 3e-4,
        log_dir: str = './logs',
        model_dir: str = './models',
        seed: int = config.random_seed
):

    print("\n" + "=" * 80)
    print(f"{'多日数据PPO训练 - 区域 ' + str(region_id):^80}")
    print("=" * 80)
    print(f"策略: {strategy}")
    print(f"数据天数: {len(trajectory_files)}")

    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 设置种子
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 创建多日数据管理器
    print("\n初始化多日数据管理器...")
    data_manager = MultiDayDataManager(
        trajectory_files=trajectory_files,
        region_file=region_file,
        dispatch_file=dispatch_points_file,
        dates=dates
    )

    # 创建第一个环境以获取状态/动作维度
    print("\n获取环境维度...")
    temp_env, _ = data_manager.create_env(region_id, max_steps, day_index=0)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    print(f"✓ 状态维度: {state_dim}")
    print(f"✓ 动作维度: {action_dim}")

    # 创建训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n创建PPO训练器 (设备: {device})...")
    trainer = PPOTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        device=device
    )

    # 创建调度器
    if strategy == 'curriculum':
        scheduler = CurriculumScheduler(
            num_days=data_manager.get_day_count(),
            warmup_episodes=50
        )
        print(f"✓ 使用课程学习策略 (预热: 50轮)")

    # TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'{log_dir}/multiday_region_{region_id}_{timestamp}')

    # 训练统计
    from collections import deque
    best_reward = -float('inf')
    best_success_rate = 0.0
    episode_rewards = deque(maxlen=100)
    episode_success_rates = deque(maxlen=100)

    # 每天的统计
    day_stats = {i: {'episodes': 0, 'avg_reward': 0.0}
                 for i in range(data_manager.get_day_count())}

    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    print(f"{'Episode':<10}{'Day':<8}{'Reward':<12}{'Avg(100)':<12}"
          f"{'Success%':<12}{'Loss':<10}")
    print("-" * 80)

    # 训练循环
    for episode in range(1, num_episodes + 1):

        # 选择今天使用哪一天的数据
        if strategy == 'curriculum':
            day_index = scheduler.sample_day()
            scheduler.step()
        elif strategy == 'mixed':
            day_index = np.random.randint(0, data_manager.get_day_count())
        elif strategy == 'sequential':
            day_index = (episode - 1) % data_manager.get_day_count()
        else:
            day_index = 0

        # 创建环境
        env, actual_day = data_manager.create_env(region_id, max_steps, day_index)
        if seed is not None:
            env.seed(seed + episode)

        # 收集轨迹
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            env.training_progress = min(1.0, episode / num_episodes)

            # 获取可达掩码
            point_result = env._extract_point_features()
            reachable_mask = np.zeros(env.num_dispatch_points, dtype=bool)
            if len(point_result.reachable_indices) > 0:
                reachable_mask[point_result.reachable_indices] = True

            # 选择动作
            action, value = trainer.select_action(state, reachable_mask)

            # 执行
            next_state, reward, done, info = env.step(action)

            # 存储
            trainer.store_transition(state, action, reward, value, done)

            episode_reward += reward
            state = next_state

            if done:
                break

        # 定期更新
        update_info = {}
        if episode % update_interval == 0:
            update_info = trainer.update(state, n_epochs=4, batch_size=64)

        # 统计
        stats = info['episode_stats']
        served = stats['served_requests']
        failed = stats['failed_requests']
        total = served + failed + info['pending_requests']
        success_rate = (served / total * 100) if total > 0 else 0

        # 更新统计
        episode_rewards.append(episode_reward)
        episode_success_rates.append(success_rate)

        # 更新每天的统计
        day_stats[actual_day]['episodes'] += 1
        old_avg = day_stats[actual_day]['avg_reward']
        n = day_stats[actual_day]['episodes']
        day_stats[actual_day]['avg_reward'] = old_avg + (episode_reward - old_avg) / n

        # TensorBoard
        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/Success_Rate', success_rate, episode)
        writer.add_scalar('Episode/Day_Used', actual_day, episode)

        for day_idx, stats_dict in day_stats.items():
            if stats_dict['episodes'] > 0:
                writer.add_scalar(f'Day_{day_idx}/Avg_Reward',
                                  stats_dict['avg_reward'], episode)

        if update_info:
            writer.add_scalar('Loss/Total', update_info['total_loss'], episode)

        # 控制台输出
        avg_reward = np.mean(episode_rewards)
        loss_str = f"{update_info.get('total_loss', 0):.4f}" if update_info else "---"
        day_name = dates[actual_day] if dates else f"Day{actual_day}"

        print(f"{episode:<10}{day_name:<8}{episode_reward:>10.2f}{avg_reward:>10.2f}"
              f"{success_rate:>10.1f}%{loss_str:>10}")

        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            save_path = f'{model_dir}/best_model_region_{region_id}.pth'
            trainer.save(save_path)

        if success_rate > best_success_rate:
            best_success_rate = success_rate

        # 定期检查点
        if episode % 100 == 0:
            checkpoint_path = f'{model_dir}/checkpoint_ep{episode}_region_{region_id}.pth'
            trainer.save(checkpoint_path)

            print(f"\n{'':>10}💾 检查点已保存")
            print(f"{'':>10}📊 每日数据使用统计:")
            for day_idx, stats_dict in day_stats.items():
                day_name = dates[day_idx] if dates else f"Day{day_idx}"
                print(f"{'':>15}{day_name}: {stats_dict['episodes']}次, "
                      f"平均奖励={stats_dict['avg_reward']:.2f}")
            print()

    # 训练完成
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"\n📈 最终统计:")
    print(f"  平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"  平均成功率: {np.mean(episode_success_rates):.2f}%")
    print(f"\n🏆 最佳记录:")
    print(f"  最佳奖励: {best_reward:.2f}")
    print(f"  最佳成功率: {best_success_rate:.2f}%")
    print(f"\n📊 数据使用统计:")
    for day_idx, stats_dict in day_stats.items():
        day_name = dates[day_idx] if dates else f"Day{day_idx}"
        print(f"  {day_name}: 使用{stats_dict['episodes']}次, "
              f"平均奖励={stats_dict['avg_reward']:.2f}")
    print("=" * 80)

    writer.close()
    return trainer


def evaluate_multiday(
        model_path: str,
        trajectory_files: List[str],
        region_file: str,
        dispatch_points_file: str,
        dates: List[str],
        region_id: int = 0,
        n_episodes_per_day: int = 3,
        max_steps: int = 100
):
    """
    在多日数据上评估模型

    Args:
        model_path: 模型路径
        trajectory_files: 轨迹文件列表
        region_file: 区域文件
        dispatch_points_file: 调度点文件
        dates: 日期列表
        region_id: 区域ID
        n_episodes_per_day: 每天评估轮数
        max_steps: 最大步数
    """
    print("\n" + "=" * 80)
    print(f"{'多日数据评估 - 区域 ' + str(region_id):^80}")
    print("=" * 80)

    # 创建数据管理器
    data_manager = MultiDayDataManager(
        trajectory_files=trajectory_files,
        region_file=region_file,
        dispatch_file=dispatch_points_file,
        dates=dates
    )

    # 获取维度
    temp_env, _ = data_manager.create_env(region_id, max_steps, 0)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = PPOTrainer(state_dim=state_dim, action_dim=action_dim, device=device)
    trainer.load(model_path)
    print(f"✓ 模型已加载: {model_path}\n")

    # 每天的评估结果
    results = {}

    for day_idx in range(data_manager.get_day_count()):
        day_name = dates[day_idx]
        print(f"\n{'=' * 80}")
        print(f"评估 {day_name}")
        print(f"{'=' * 80}")

        day_rewards = []
        day_success_rates = []
        day_wait_times = []

        for ep in range(n_episodes_per_day):
            # 创建环境
            env, _ = data_manager.create_env(region_id, max_steps, day_idx)

            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                point_result = env._extract_point_features()
                reachable_mask = np.zeros(env.num_dispatch_points, dtype=bool)
                if len(point_result.reachable_indices) > 0:
                    reachable_mask[point_result.reachable_indices] = True

                action, _ = trainer.select_action(state, reachable_mask, deterministic=True)
                state, reward, done, info = env.step(action)
                episode_reward += reward

                if done:
                    break

            # 统计
            stats = info['episode_stats']
            served = stats['served_requests']
            failed = stats['failed_requests']
            total = served + failed + info['pending_requests']
            success_rate = (served / total * 100) if total > 0 else 0

            wait_time = 0
            if stats['wait_time_count'] > 0:
                wait_time = stats['total_wait_time'] / stats['wait_time_count']

            day_rewards.append(episode_reward)
            day_success_rates.append(success_rate)
            day_wait_times.append(wait_time)

            print(f"  Episode {ep + 1}/{n_episodes_per_day}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Success={success_rate:.1f}%, "
                  f"Wait={wait_time:.2f}min")

        # 保存结果
        results[day_name] = {
            'reward': (np.mean(day_rewards), np.std(day_rewards)),
            'success_rate': (np.mean(day_success_rates), np.std(day_success_rates)),
            'wait_time': (np.mean(day_wait_times), np.std(day_wait_times))
        }

    # 输出汇总
    print("\n" + "=" * 80)
    print("评估汇总")
    print("=" * 80)
    print(f"\n{'日期':<12}{'平均奖励':<20}{'成功率':<20}{'等待时间(分钟)':<20}")
    print("-" * 80)

    for day_name, res in results.items():
        reward_str = f"{res['reward'][0]:.2f} ± {res['reward'][1]:.2f}"
        success_str = f"{res['success_rate'][0]:.1f}% ± {res['success_rate'][1]:.1f}%"
        wait_str = f"{res['wait_time'][0]:.2f} ± {res['wait_time'][1]:.2f}"
        print(f"{day_name:<12}{reward_str:<20}{success_str:<20}{wait_str:<20}")

    # 整体平均
    overall_reward = np.mean([res['reward'][0] for res in results.values()])
    overall_success = np.mean([res['success_rate'][0] for res in results.values()])
    overall_wait = np.mean([res['wait_time'][0] for res in results.values()])

    print("-" * 80)
    print(f"{'整体平均':<12}{overall_reward:<20.2f}{overall_success:<20.1f}%"
          f"{overall_wait:<20.2f}")
    print("=" * 80)

    return results


# ============================================================
# 便捷函数：自动扫描并训练
# ============================================================
def auto_train_with_available_data(
        data_dir: str = 'dataset/top1000evs/reallocated',
        region_file: str = 'dataset/fcs_regions.geojson',
        dispatch_file: str = 'dataset/dispatch_points.csv',
        region_id: int = 0,
        strategy: str = 'curriculum',
        num_episodes: int = 500
):
    """
    自动扫描目录中的所有数据文件并训练

    Args:
        data_dir: 数据目录
        region_file: 区域文件
        dispatch_file: 调度点文件
        region_id: 区域ID
        strategy: 训练策略
        num_episodes: 训练轮数
    """
    import glob

    # 扫描CSV文件
    pattern = os.path.join(data_dir, '*.csv')
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print(f"❌ 在 {data_dir} 中未找到CSV文件")
        return None

    print(f"\n发现 {len(files)} 个数据文件:")
    for i, f in enumerate(files):
        print(f"  {i + 1}. {os.path.basename(f)}")

    # 提取日期
    dates = []
    for f in files:
        basename = os.path.basename(f)
        # 假设文件名格式: 20140818_processed.csv
        date_part = basename.split('_')[0]
        dates.append(date_part)

    print(f"\n使用策略: {strategy}")
    print(f"日期范围: {dates[0]} ~ {dates[-1]}")

    # 训练
    trainer = train_ppo_multiday(
        trajectory_files=files,
        region_file=region_file,
        dispatch_points_file=dispatch_file,
        dates=dates,
        region_id=region_id,
        num_episodes=num_episodes,
        strategy=strategy
    )

    return trainer, files, dates


if __name__ == '__main__':
    # 示例：自动训练
    auto_train_with_available_data(
        data_dir='dataset/top1000evs/reallocated',
        region_id=0,
        strategy='curriculum',  # 可选: 'curriculum', 'mixed', 'sequential'
        num_episodes=500
    )