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
    
class TemporalAwareGatingLayer(nn.Module):
    def __init__(self, num_features, seq_len=64, kernel_size=3):
        super().__init__()
        self.num_features = num_features
        # 1. 时序特征提取（轻量1D卷积）
        self.temporal_conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=kernel_size,
            padding=kernel_size//2,  # 保持时序长度不变
            groups=num_features      # 分组卷积：轻量化，逐特征捕捉时序
        )
        # 2. 门控生成（融合时序+特征）
        self.gate_linear = nn.Linear(num_features * 2, num_features)  # 输入：当前特征+时序特征
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        # x: [batch, seq, num_features] → LOB 标准输入
        x_norm = self.norm(x)
        
        # 提取时序特征：[batch, seq, f] → [batch, f, seq] → conv → [batch, f, seq] → [batch, seq, f]
        x_temporal = self.temporal_conv(x_norm.permute(0,2,1)).permute(0,2,1)
        x_temporal = self.gelu(x_temporal)
        
        # 融合当前特征+时序特征生成门控
        gate_input = torch.cat([x_norm, x_temporal], dim=-1)
        gate = self.sigmoid(self.gate_linear(gate_input))
        
        # 门控加权
        gated_x = x * gate
        return gated_x