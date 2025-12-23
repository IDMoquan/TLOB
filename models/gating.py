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
    
class DualBranchGatingLayer(nn.Module):
    def __init__(self, num_features, hidden_dim=None):
        super().__init__()
        self.num_features = num_features
        hidden_dim = hidden_dim or num_features  # 轻量化：隐藏层维度从2倍→1倍
        # 共享基础线性层（减少参数量）
        self.gate_linear1 = nn.Linear(num_features, hidden_dim)
        self.gelu = nn.GELU()
        # 双分支门控：乘性（scale）+ 加性（shift）
        self.scale_linear = nn.Linear(hidden_dim, num_features)  # 乘性门控（0-1）
        self.shift_linear = nn.Linear(hidden_dim, num_features)  # 加性门控（偏移）
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        # x: [batch, seq, num_features]（LOB 标准输入维度）
        x_norm = self.norm(x)
        # 共享特征映射
        feat = self.gelu(self.gate_linear1(x_norm))
        # 双分支门控生成
        scale_gate = self.sigmoid(self.scale_linear(feat))  # 乘性：0-1
        shift_gate = torch.tanh(self.shift_linear(feat))    # 加性：-1~1（避免偏移过大）
        # 融合：乘性筛选 + 加性补偿
        gated_x = x * scale_gate + shift_gate
        return gated_x