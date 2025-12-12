import os
from typing import List, Dict

import torch
import numpy as np

from a2c import ActorCritic


class EdgeModelManager:
    """管理多个区域的边缘模型"""

    def __init__(self, num_regions: int, model_dir: str, device: str = 'cuda'):
        self.num_regions = num_regions
        self.model_dir = model_dir
        self.device = device
        self.models = {}
        self.is_loaded = {}

    def load_model(self, region_id: int, state_dim: int, action_dim: int) -> bool:
        """加载指定区域的模型"""
        if region_id == 3:
            loaded_id = 0
        elif region_id == 7:
            loaded_id = 9
        elif region_id == 13:
            loaded_id = 14
        else:
            loaded_id = region_id
        model_path = f'{self.model_dir}/best_model_region_{loaded_id}.pth'

        if not os.path.exists(model_path):
            print(f"⚠ 区域 {region_id} 的模型不存在: {model_path}")
            return False

        try:
            # 创建模型
            model = ActorCritic(state_dim, action_dim).to(self.device)

            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # 设置为评估模式

            self.models[region_id] = model
            self.is_loaded[region_id] = True

            print(f"✓ 成功加载区域 {region_id} 的模型 (成功率: {checkpoint.get('success_rate', 0):.2f}%)")
            return True

        except Exception as e:
            print(f"✗ 加载区域 {region_id} 模型失败: {e}")
            return False

    def load_all_models(self, region_dict: List[Dict]) -> int:
        """加载所有区域的模型"""
        loaded_count = 0

        for region_id in range(self.num_regions):
            state_dim = region_dict[region_id]['state_dim']
            action_dim = region_dict[region_id]['action_dim']
            if self.load_model(region_id, state_dim, action_dim):
                loaded_count += 1
        return loaded_count

    def get_action(self, region_id: int, state: np.ndarray) -> np.ndarray:
        """使用边缘模型获取动作"""
        if region_id not in self.models:
            # 如果模型未加载,返回零动作
            return np.zeros(state.shape[0] if len(state.shape) > 1 else 1)

        model = self.models[region_id]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_scores, _ = model(state_tensor)

        return action_scores.squeeze(0).cpu().numpy()
