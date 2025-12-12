import torch.nn as nn
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()

        # 共享特征提取层 - 使用残差连接
        self.shared_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for i in range(3)
        ])

        # Actor头
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # 限制输出范围
        )

        # Critic头
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """使用正交初始化提升稳定性"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        # 使用残差连接
        x = state
        for layer in self.shared_layers:
            residual = x if x.shape[-1] == layer[0].out_features else None
            x = layer(x)
            if residual is not None:
                x = x + residual

        action_scores = self.actor(x)
        state_value = self.critic(x)
        return action_scores, state_value
