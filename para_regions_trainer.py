"""
增强版多区域并行训练脚本
支持断点续训、实时监控、资源优化
"""

import os
import glob
import json
import torch
import torch.multiprocessing as mp
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path
import time
import threading
from collections import defaultdict

from dataloader import DataLoaderFactory
from edge_env import EdgeEnv
from train import PPOTrainer
from params_config import Config
from multiday_runner import MultiDayDataManager, CurriculumScheduler

config = Config()


class TrainingMonitor:
    """训练监控器 - 实时追踪训练进度"""

    def __init__(self, num_regions: int, checkpoint_dir: str):
        self.num_regions = num_regions
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 训练状态
        self.region_status = {}  # {region_id: 'pending'|'training'|'completed'|'failed'}
        self.region_progress = {}  # {region_id: episode}
        self.region_metrics = {}  # {region_id: metrics_dict}

        # 锁
        self.lock = threading.Lock()

        # 加载检查点
        self._load_checkpoint()

    def _checkpoint_file(self):
        return self.checkpoint_dir / 'training_checkpoint.json'

    def _load_checkpoint(self):
        """加载检查点"""
        checkpoint_file = self._checkpoint_file()
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                self.region_status = data.get('region_status', {})
                self.region_progress = {int(k): v for k, v in data.get('region_progress', {}).items()}
                self.region_metrics = {int(k): v for k, v in data.get('region_metrics', {}).items()}
            print(f"✓ 加载检查点: {checkpoint_file}")
            print(
                f"  - 已完成区域: {sum(1 for s in self.region_status.values() if s == 'completed')}/{self.num_regions}")
        else:
            # 初始化所有区域为pending
            for region_id in range(self.num_regions):
                self.region_status[region_id] = 'pending'
                self.region_progress[region_id] = 0
                self.region_metrics[region_id] = {}

    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint_file = self._checkpoint_file()
        data = {
            'region_status': self.region_status,
            'region_progress': self.region_progress,
            'region_metrics': self.region_metrics,
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)

    def update_status(self, region_id: int, status: str, episode: int = None, metrics: Dict = None):
        """更新区域状态"""
        with self.lock:
            self.region_status[region_id] = status
            if episode is not None:
                self.region_progress[region_id] = episode
            if metrics is not None:
                self.region_metrics[region_id] = metrics
            self._save_checkpoint()

    def get_pending_regions(self) -> List[int]:
        """获取待训练的区域"""
        with self.lock:
            return [rid for rid, status in self.region_status.items()
                    if status in ['pending', 'failed']]

    def get_summary(self) -> Dict:
        """获取训练摘要"""
        with self.lock:
            status_counts = defaultdict(int)
            for status in self.region_status.values():
                status_counts[status] += 1

            return {
                'total': self.num_regions,
                'completed': status_counts['completed'],
                'training': status_counts['training'],
                'pending': status_counts['pending'],
                'failed': status_counts['failed'],
                'progress': sum(self.region_progress.values()) / (self.num_regions * 1.0)
            }

    def print_summary(self):
        """打印训练摘要"""
        summary = self.get_summary()
        print(f"\n📊 训练进度摘要:")
        print(f"  总区域数: {summary['total']}")
        print(f"  ✅ 已完成: {summary['completed']}")
        print(f"  🔄 训练中: {summary['training']}")
        print(f"  ⏸️  待训练: {summary['pending']}")
        print(f"  ❌ 失败: {summary['failed']}")

        # 显示详细进度
        if summary['training'] > 0 or summary['pending'] > 0:
            print(f"\n  详细状态:")
            for region_id in range(self.num_regions):
                status = self.region_status[region_id]
                if status in ['training', 'pending']:
                    progress = self.region_progress.get(region_id, 0)
                    emoji = '🔄' if status == 'training' else '⏸️'
                    print(f"    {emoji} 区域 {region_id}: {status} (进度: {progress})")


class EnhancedGPUManager:
    """增强版GPU管理器 - 支持动态负载均衡"""

    def __init__(self, gpu_ids: List[int] = None, memory_threshold: float = 0.9):
        """
        Args:
            gpu_ids: 可用GPU ID列表
            memory_threshold: 内存使用阈值
        """
        if gpu_ids is None:
            if torch.cuda.is_available():
                self.gpu_ids = list(range(torch.cuda.device_count()))
            else:
                self.gpu_ids = []
        else:
            self.gpu_ids = gpu_ids

        self.num_gpus = len(self.gpu_ids)
        self.memory_threshold = memory_threshold

        # GPU负载追踪
        self.gpu_load = {gpu_id: 0 for gpu_id in self.gpu_ids}
        self.lock = threading.Lock()

        print(f"🎮 增强版GPU管理器初始化:")
        print(f"  - 可用GPU数量: {self.num_gpus}")
        if self.num_gpus > 0:
            for gpu_id in self.gpu_ids:
                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                print(f"  - GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")

    def assign_gpu(self, region_id: int) -> str:
        """
        智能分配GPU（负载均衡）

        Args:
            region_id: 区域ID

        Returns:
            设备字符串
        """
        if self.num_gpus == 0:
            return 'cpu'

        with self.lock:
            # 选择负载最低的GPU
            min_load_gpu = min(self.gpu_ids, key=lambda gpu_id: self.gpu_load[gpu_id])
            self.gpu_load[min_load_gpu] += 1
            return f'cuda:{min_load_gpu}'

    def release_gpu(self, device: str):
        """释放GPU"""
        if device.startswith('cuda:'):
            gpu_id = int(device.split(':')[1])
            with self.lock:
                if gpu_id in self.gpu_load:
                    self.gpu_load[gpu_id] = max(0, self.gpu_load[gpu_id] - 1)

    def get_load_info(self) -> Dict:
        """获取GPU负载信息"""
        with self.lock:
            return self.gpu_load.copy()


def train_single_region_with_monitoring(
        region_id: int,
        trajectory_files: List[str],
        region_file: str,
        dispatch_file: str,
        dates: List[str],
        device: str,
        num_episodes: int,
        max_steps: int,
        strategy: str,
        model_dir: str,
        log_dir: str,
        seed: int,
        monitor: TrainingMonitor,
        gpu_manager: EnhancedGPUManager,
        resume_from: int = 0
):
    """
    带监控的单区域训练

    Args:
        resume_from: 从哪个episode恢复训练
    """
    try:
        # 更新状态
        monitor.update_status(region_id, 'training', resume_from)

        print(f"\n[Region {region_id}] 开始训练 (设备: {device}, 从 Episode {resume_from} 恢复)")

        # 设置随机种子
        if seed is not None:
            np.random.seed(seed + region_id)
            torch.manual_seed(seed + region_id)

        # 创建数据管理器
        data_manager = MultiDayDataManager(
            trajectory_files=trajectory_files,
            region_file=region_file,
            dispatch_file=dispatch_file,
            dates=dates
        )

        # 获取维度
        temp_env, _ = data_manager.create_env(region_id, max_steps, 0)
        state_dim = temp_env.observation_space.shape[0]
        action_dim = temp_env.action_space.shape[0]

        # 创建训练器
        trainer = PPOTrainer(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=3e-4,
            device=device
        )

        # 恢复训练
        if resume_from > 0:
            checkpoint_path = f'{model_dir}/checkpoint_ep{resume_from}_region_{region_id}.pth'
            if os.path.exists(checkpoint_path):
                trainer.load(checkpoint_path)
                print(f"  ✓ 从检查点恢复: {checkpoint_path}")

        # 调度器
        scheduler = CurriculumScheduler(
            num_days=data_manager.get_day_count(),
            warmup_episodes=50
        )
        scheduler.current_episode = resume_from

        # 训练统计
        from collections import deque
        best_reward = -float('inf')
        best_success_rate = 0.0
        episode_rewards = deque(maxlen=100)
        episode_success_rates = deque(maxlen=100)

        # 训练循环
        for episode in range(resume_from + 1, num_episodes + 1):
            # 选择数据
            if strategy == 'curriculum':
                day_index = scheduler.sample_day()
                scheduler.step()
            elif strategy == 'mixed':
                day_index = np.random.randint(0, data_manager.get_day_count())
            else:
                day_index = (episode - 1) % data_manager.get_day_count()

            # 创建环境
            env, actual_day = data_manager.create_env(region_id, max_steps, day_index)
            if seed is not None:
                env.seed(seed + region_id + episode)

            # 收集轨迹
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                env.training_progress = min(1.0, episode / num_episodes)

                point_result = env._extract_point_features()
                reachable_mask = np.zeros(env.num_dispatch_points, dtype=bool)
                if len(point_result.reachable_indices) > 0:
                    reachable_mask[point_result.reachable_indices] = True

                action, value = trainer.select_action(state, reachable_mask)
                next_state, reward, done, info = env.step(action)
                trainer.store_transition(state, action, reward, value, done)

                episode_reward += reward
                state = next_state

                if done:
                    break

            # 更新
            if episode % 10 == 0:
                trainer.update(state, n_epochs=4, batch_size=64)

            # 统计
            stats = info['episode_stats']
            served = stats['served_requests']
            failed = stats['failed_requests']
            total = served + failed + info['pending_requests']
            success_rate = (served / total * 100) if total > 0 else 0

            episode_rewards.append(episode_reward)
            episode_success_rates.append(success_rate)

            # 保存最佳模型
            if episode_reward > best_reward:
                best_reward = episode_reward
                save_path = f'{model_dir}/best_model_region_{region_id}.pth'
                trainer.save(save_path)

            if success_rate > best_success_rate:
                best_success_rate = success_rate

            # 定期保存检查点
            if episode % 100 == 0:
                checkpoint_path = f'{model_dir}/checkpoint_ep{episode}_region_{region_id}.pth'
                trainer.save(checkpoint_path)

                # 更新监控
                metrics = {
                    'avg_reward': float(np.mean(episode_rewards)),
                    'avg_success_rate': float(np.mean(episode_success_rates)),
                    'best_reward': float(best_reward),
                    'best_success_rate': float(best_success_rate)
                }
                monitor.update_status(region_id, 'training', episode, metrics)

            # 定期输出
            if episode % 50 == 0:
                avg_reward = np.mean(episode_rewards)
                avg_success = np.mean(episode_success_rates)
                print(f"[Region {region_id}] Ep {episode}/{num_episodes}: "
                      f"Reward={avg_reward:.2f}, Success={avg_success:.1f}%")

        # 训练完成
        final_reward = np.mean(episode_rewards)
        final_success = np.mean(episode_success_rates)

        final_metrics = {
            'final_reward': float(final_reward),
            'final_success_rate': float(final_success),
            'best_reward': float(best_reward),
            'best_success_rate': float(best_success_rate),
            'completed': True
        }

        monitor.update_status(region_id, 'completed', num_episodes, final_metrics)

        print(f"\n[Region {region_id}] ✅ 训练完成!")
        print(f"  最终奖励: {final_reward:.2f}")
        print(f"  最终成功率: {final_success:.2f}%")

        # 释放GPU
        gpu_manager.release_gpu(device)

        return final_metrics

    except Exception as e:
        print(f"[Region {region_id}] ❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()

        monitor.update_status(region_id, 'failed', resume_from, {'error': str(e)})
        gpu_manager.release_gpu(device)

        return {'error': str(e)}


def enhanced_parallel_train(
        trajectory_files: List[str],
        region_file: str,
        dispatch_file: str,
        dates: List[str] = None,
        num_regions: int = 18,
        num_episodes: int = 500,
        max_steps: int = 100,
        strategy: str = 'curriculum',
        model_dir: str = './models',
        log_dir: str = './logs',
        checkpoint_dir: str = './checkpoints',
        seed: int = config.random_seed,
        gpu_ids: List[int] = None,
        max_parallel: int = None,
        resume: bool = True
):
    """
    增强版并行训练 - 支持断点续训

    Args:
        resume: 是否从检查点恢复
    """
    print("\n" + "=" * 80)
    print(f"{'增强版多区域并行训练':^80}")
    print("=" * 80)

    # 创建目录
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 初始化监控器
    monitor = TrainingMonitor(num_regions, checkpoint_dir)

    # 显示当前状态
    monitor.print_summary()

    # GPU管理器
    gpu_manager = EnhancedGPUManager(gpu_ids)

    # 确定最大并行数
    if max_parallel is None:
        if gpu_manager.num_gpus > 0:
            max_parallel = gpu_manager.num_gpus
        else:
            max_parallel = min(4, mp.cpu_count())

    print(f"\n📊 训练配置:")
    print(f"  - 最大并行进程数: {max_parallel}")
    print(f"  - 训练策略: {strategy}")
    print(f"  - 断点续训: {'启用' if resume else '禁用'}")

    # 获取待训练区域
    if resume:
        pending_regions = monitor.get_pending_regions()
        print(f"\n🔄 将训练 {len(pending_regions)} 个区域")
    else:
        pending_regions = list(range(num_regions))
        print(f"\n🆕 从头开始训练所有区域")

    if not pending_regions:
        print("\n✅ 所有区域已完成训练!")
        return monitor.region_metrics

    # 多进程设置
    mp.set_start_method('spawn', force=True)

    # 训练开始时间
    start_time = time.time()

    # 分批并行训练
    for batch_start in range(0, len(pending_regions), max_parallel):
        batch_end = min(batch_start + max_parallel, len(pending_regions))
        batch_region_ids = pending_regions[batch_start:batch_end]

        print(f"\n🚀 启动批次: {batch_region_ids}")

        # 创建进程
        processes = []
        for region_id in batch_region_ids:
            device = gpu_manager.assign_gpu(region_id)
            resume_from = monitor.region_progress.get(region_id, 0) if resume else 0

            p = mp.Process(
                target=train_single_region_with_monitoring,
                args=(
                    region_id,
                    trajectory_files,
                    region_file,
                    dispatch_file,
                    dates,
                    device,
                    num_episodes,
                    max_steps,
                    strategy,
                    model_dir,
                    log_dir,
                    seed,
                    monitor,
                    gpu_manager,
                    resume_from
                )
            )
            p.start()
            processes.append(p)
            print(f"  - 区域 {region_id} -> {device} (从 Episode {resume_from} 开始)")

        # 等待批次完成
        for p in processes:
            p.join()

        # 显示进度
        monitor.print_summary()

    # 训练完成
    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"{'训练完成!':^80}")
    print("=" * 80)
    print(f"\n⏱️  总时间: {total_time / 3600:.2f} 小时")

    # 最终统计
    completed_regions = [rid for rid, status in monitor.region_status.items()
                         if status == 'completed']

    if completed_regions:
        rewards = [monitor.region_metrics[rid]['final_reward']
                   for rid in completed_regions]
        success_rates = [monitor.region_metrics[rid]['final_success_rate']
                         for rid in completed_regions]

        print(f"\n📊 最终统计:")
        print(f"  - 平均奖励: {np.mean(rewards):.2f}")
        print(f"  - 平均成功率: {np.mean(success_rates):.2f}%")
        print(f"  - 最佳奖励: {np.max(rewards):.2f}")
        print(f"  - 最佳成功率: {np.max(success_rates):.2f}%")

    print("\n💾 模型保存位置:", model_dir)
    print("📊 日志保存位置:", log_dir)
    print("🔖 检查点保存位置:", checkpoint_dir)
    print("=" * 80)

    return monitor.region_metrics


if __name__ == '__main__':
    # 自动扫描并训练
    pattern = 'dataset/top1000evs/reallocated/*.csv'
    files = sorted(glob.glob(pattern))

    if files:
        dates = [os.path.basename(f).split('_')[0] for f in files]

        results = enhanced_parallel_train(
            trajectory_files=files,
            region_file='dataset/fcs_regions.geojson',
            dispatch_file='dataset/dispatch_points.csv',
            dates=dates,
            num_regions=18,
            num_episodes=500,
            max_steps=100,
            strategy='curriculum',
            model_dir='./models',
            log_dir='./logs',
            checkpoint_dir='./checkpoints',
            seed=42,
            gpu_ids=None,  # 自动检测
            max_parallel=None,  # 自动设置
            resume=True  # 启用断点续训
        )
    else:
        print("未找到数据文件!")