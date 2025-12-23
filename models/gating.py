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
    
class RegularizedGatingLayer(nn.Module):
    def __init__(self, num_features, hidden_dim=None, drop_path_rate=0.1):
        super().__init__()
        self.num_features = num_features
        hidden_dim = hidden_dim or num_features * 2
        self.gate_linear1 = nn.Linear(num_features, hidden_dim)
        self.gate_linear2 = nn.Linear(hidden_dim, num_features)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(num_features)
        # 正则化组件
        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        # 稀疏约束：门控权重L2归一化
        self.sparse_weight = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        x_norm = self.norm(x)
        # 门控生成 + DropPath 正则
        gate = self.gelu(self.gate_linear1(x_norm))
        gate = self.drop_path(gate)
        gate = self.sigmoid(self.gate_linear2(gate))
        # 稀疏约束：加权门控，强制核心特征权重更高
        gate = gate * self.sparse_weight.unsqueeze(0).unsqueeze(0)
        gate = gate / (gate.norm(dim=-1, keepdim=True) + 1e-6)  # L2归一化
        # 门控加权
        gated_x = x * gate
        return gated_x
    
    # 可选：新增正则化损失，训练时加入总损失
    def get_sparse_loss(self):
        # 鼓励门控权重稀疏（L1正则）
        return torch.norm(self.sparse_weight, p=1) * 1e-4