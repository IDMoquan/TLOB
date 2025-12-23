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
    
class TemporalLowRankGatingLayer(nn.Module):
    def __init__(self, 
                 num_features: int, 
                 seq_len: int = 64,  # LOB 时序长度（需与输入匹配）
                 low_rank_k: int = 8, # 低秩维度（建议8/16，k<<num_features）
                 kernel_size: int = 3, # 时序卷积核大小（奇数保证padding后长度不变）
                 drop_path_rate: float = 0.05): # 轻量正则
        super().__init__()
        self.num_features = num_features
        self.low_rank_k = min(low_rank_k, num_features)  # 防止k超过特征数
        self.seq_len = seq_len

        # 1. 时序感知分支（轻量分组1D卷积）
        self.temporal_conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # 保持时序长度不变
            groups=num_features,       # 分组卷积：逐特征捕捉时序，参数量仅O(f*kernel_size)
            bias=False                 # 减少冗余参数
        )
        self.temporal_norm = nn.LayerNorm(num_features)  # 时序特征归一化
        self.gelu = nn.GELU()

        # 2. 低秩门控生成（替代原全连接层，参数量从O(f²)→O(f*k)）
        # 低秩分解：Linear(f, k) + Linear(k, f)
        self.gate_linear1 = nn.Linear(num_features * 2, self.low_rank_k)  # 输入：当前特征+时序特征
        self.gate_linear2 = nn.Linear(self.low_rank_k, num_features)
        self.sigmoid = nn.Sigmoid()

        # 3. 基础归一化
        self.norm = nn.LayerNorm(num_features)

        # 4. 正则化 + 稳定训练组件
        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)  # 残差缩放系数（初始小值，稳定训练）

    def forward(self, x):
        # Step 1: 基础归一化（当前特征）
        x_norm = self.norm(x)  # [batch, seq, f]

        # Step 2: 时序特征提取
        # 维度调整：Conv1d要求[batch, channels, seq_len] → 转置后卷积
        x_perm = x_norm.permute(0, 2, 1)  # [batch, f, seq]
        x_temporal = self.temporal_conv(x_perm)  # [batch, f, seq]
        x_temporal = x_temporal.permute(0, 2, 1)  # 转回 [batch, seq, f]
        x_temporal = self.temporal_norm(x_temporal)
        x_temporal = self.gelu(x_temporal)  # 时序特征非线性激活

        # Step 3: 融合当前特征 + 时序特征
        feat_fused = torch.cat([x_norm, x_temporal], dim=-1)  # [batch, seq, 2f]
        feat_fused = self.drop_path(feat_fused)  # 轻量正则

        # Step 4: 低秩门控生成
        gate = self.gelu(self.gate_linear1(feat_fused))  # [batch, seq, k]
        gate = self.sigmoid(self.gate_linear2(gate))     # [batch, seq, f] → 0-1门控

        # Step 5: 残差融合（保留原始特征，避免门控饱和）
        gated_x = x * (1 - self.res_scale) + x * gate * self.res_scale

        return gated_x