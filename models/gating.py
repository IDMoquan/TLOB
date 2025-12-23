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
    
class AdaptiveNormGatingLayer(nn.Module):
    def __init__(self, num_features, hidden_dim=None):
        super().__init__()
        self.num_features = num_features
        hidden_dim = hidden_dim or num_features
        # 门控生成：预测归一化的 scale/shift
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_features * 2)  # 输出：scale + shift
        )
        # 基础归一化（无可学习参数）
        self.norm = nn.LayerNorm(num_features, elementwise_affine=False)

    def forward(self, x):
        # Step 1: 基础归一化
        x_norm = self.norm(x)
        # Step 2: 生成自适应 scale/shift（门控信号）
        gate_params = self.gate_mlp(x)
        scale, shift = torch.chunk(gate_params, 2, dim=-1)
        # Step 3: 门控化归一化（替代单纯的乘性门控）
        scale = torch.sigmoid(scale)  # scale: 0-1（权重）
        shift = torch.tanh(shift)     # shift: -1~1（偏移）
        gated_x = x_norm * scale + shift
        # Step 4: 残差融合原始特征
        gated_x = gated_x + x * 0.1
        return gated_x