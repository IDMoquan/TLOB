import torch
import torch.nn as nn

class GatingLayer(nn.Module):
    def __init__(self, num_features, hidden_dim=None):
        super().__init__()
        self.num_features = num_features
        hidden_dim = hidden_dim or num_features * 2
        self.gate_linear1 = nn.Linear(num_features, hidden_dim)
        self.gate_linear2 = nn.Linear(hidden_dim, num_features)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(num_features)
        
    def forward(self, x):
        x_norm = self.norm(x)
        gate = self.gate_linear1(x_norm)
        gate = self.gelu(gate)
        gate = self.gate_linear2(gate)
        gate = self.sigmoid(gate)
        gated_x = x * gate
        return gated_x
    
class LowRankGatingLayer(nn.Module):
    def __init__(self, num_features, hidden_dim=None, low_rank_k=8):
        super().__init__()
        self.num_features = num_features
        self.low_rank_k = min(low_rank_k, num_features)  # 低秩维度（建议8/16）
        # 低秩分解：Linear(f, k) + Linear(k, f) 替代 Linear(f, 2f)
        self.gate_linear1 = nn.Linear(num_features, self.low_rank_k)
        self.gate_linear2 = nn.Linear(self.low_rank_k, num_features)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(num_features)
        # 残差缩放：稳定训练
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        x_norm = self.norm(x)
        # 低秩特征映射
        gate = self.gelu(self.gate_linear1(x_norm))
        gate = self.sigmoid(self.gate_linear2(gate))
        # 残差融合：保留原始特征的基础信息
        gated_x = x * (1 - self.res_scale) + x * gate * self.res_scale
        return gated_x