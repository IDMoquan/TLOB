import torch
from torch import nn
import constants as cst
from einops import rearrange

class BiN(nn.Module):
    def __init__(self, d1, t1):
        super().__init__()
        self.t1 = t1
        self.d1 = d1

        bias1 = torch.Tensor(t1, 1)
        self.B1 = nn.Parameter(bias1)
        nn.init.constant_(self.B1, 0)

        l1 = torch.Tensor(t1, 1)
        self.l1 = nn.Parameter(l1)
        nn.init.xavier_normal_(self.l1)

        bias2 = torch.Tensor(d1, 1)
        self.B2 = nn.Parameter(bias2)
        nn.init.constant_(self.B2, 0)

        l2 = torch.Tensor(d1, 1)
        self.l2 = nn.Parameter(l2)
        nn.init.xavier_normal_(self.l2)

        y1 = torch.Tensor(1, )
        self.y1 = nn.Parameter(y1)
        nn.init.constant_(self.y1, 0.5)

        y2 = torch.Tensor(1, )
        self.y2 = nn.Parameter(y2)
        nn.init.constant_(self.y2, 0.5)

    def forward(self, x):

        # if the two scalars are negative then we setting them to 0
        if (self.y1[0] < 0):
            y1 = torch.cuda.FloatTensor(1, )
            self.y1 = nn.Parameter(y1)
            nn.init.constant_(self.y1, 0.01)

        if (self.y2[0] < 0):
            y2 = torch.cuda.FloatTensor(1, )
            self.y2 = nn.Parameter(y2)
            nn.init.constant_(self.y2, 0.01)

        # normalization along the temporal dimensione
        T2 = torch.ones([self.t1, 1], device=cst.DEVICE)
        x2 = torch.mean(x, dim=2)
        x2 = torch.reshape(x2, (x2.shape[0], x2.shape[1], 1))
        
        std = torch.std(x, dim=2)
        std = torch.reshape(std, (std.shape[0], std.shape[1], 1))
        # it can be possible that the std of some temporal slices is 0, and this produces inf values, so we have to set them to one
        std[std < 1e-4] = 1
        diff = x - (x2 @ (T2.T))
        Z2 = diff / (std @ (T2.T))

        X2 = self.l2 @ T2.T
        X2 = X2 * Z2
        X2 = X2 + (self.B2 @ T2.T)

        # normalization along the feature dimension
        T1 = torch.ones([self.d1, 1], device=cst.DEVICE)
        x1 = torch.mean(x, dim=1)
        x1 = torch.reshape(x1, (x1.shape[0], x1.shape[1], 1))

        std = torch.std(x, dim=1)
        std = torch.reshape(std, (std.shape[0], std.shape[1], 1))

        op1 = x1 @ T1.T
        op1 = torch.permute(op1, (0, 2, 1))

        op2 = std @ T1.T
        op2 = torch.permute(op2, (0, 2, 1))

        z1 = (x - op1) / (op2)
        X1 = (T1 @ self.l1.T)
        X1 = X1 * z1
        X1 = X1 + (T1 @ self.B1.T)

        # weighing the imporance of temporal and feature normalization
        x = self.y1 * X1 + self.y2 * X2

        return x
    
class CovarianceAwareBiN(nn.Module):
    def __init__(self, 
                 num_features: int, 
                 seq_size: int,
                 low_rank_k: int = 8,  # 协方差低秩近似维度（远小于num_features）
                 gamma_init: float = 0.1):
        super().__init__()
        # 1. 保留原始 BiN 线性归一化
        self.bi_norm = BiN(num_features, seq_size)
        
        # 2. 协方差感知核心参数
        self.num_features = num_features
        self.seq_size = seq_size
        self.low_rank_k = min(low_rank_k, num_features)  # 防止k超过特征数
        
        # 可学习协方差强度系数（γ）
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        # 特征投影矩阵（替代纯统计协方差，可学习）
        self.cov_proj = nn.Parameter(torch.randn(num_features, self.low_rank_k))
        nn.init.orthogonal_(self.cov_proj)  # 正交初始化，稳定投影
        
        # 3. 门控融合层（自适应权衡协方差特征/原始特征）
        self.gate = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.LayerNorm(num_features),
            nn.Sigmoid()  # 输出0-1权重
        )
        
        # 4. 轻量正则：防止协方差过拟合
        self.dropout = nn.Dropout(0.05)
        self.res_scale = nn.Parameter(torch.ones(1))

    def compute_feature_covariance(self, x):
        """
        计算特征协方差矩阵（轻量化，低秩近似）
        x: [batch, features, seq] → 输出协方差投影特征 [batch, features, seq]
        """
        # Step 1: 维度重排 → [batch*seq, features]（合并批次和时序，计算全局协方差）
        x_flat = rearrange(x, 'b f s -> (b s) f')  # [BS, f]
        
        # Step 2: 去均值（特征维度）
        x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)  # [BS, f]
        
        # Step 3: 低秩协方差计算（避免O(f²)）
        # 方法：先投影到低维，再计算协方差，最后投影回原维度
        x_low = x_centered @ self.cov_proj  # [BS, k] → 低维投影
        cov_low = x_low.T @ x_low / (x_low.shape[0] - 1)  # [k, k] → 低维协方差
        
        # Step 4: 协方差引导的特征变换
        x_cov_low = x_low @ cov_low  # [BS, k] → 协方差增强
        x_cov = x_cov_low @ self.cov_proj.T  # [BS, f] → 投影回原维度
        
        # Step 5: 恢复原维度 + 强度控制
        x_cov = rearrange(x_cov, '(b s) f -> b f s', b=x.shape[0], s=self.seq_size)
        x_cov = x_cov * torch.abs(self.gamma)  # 用abs保证非负强度
        
        return x_cov

    def forward(self, x):
        """
        x: [batch, features, seq] → 原始输入（与BiN一致）
        """
        # Step 1: 原始 BiN 归一化
        x_bi = self.bi_norm(x)  # [b, f, s]
        
        # Step 2: 协方差感知特征增强
        x_cov = self.compute_feature_covariance(x_bi)  # [b, f, s]
        
        # Step 3: 门控融合（维度适配）
        # 门控输入：[b, f, s] → [b*s, f] → 生成逐特征门控权重
        gate_input = rearrange(x_bi, 'b f s -> (b s) f')
        gate_weight = self.gate(gate_input)  # [BS, f]
        gate_weight = rearrange(gate_weight, '(b s) f -> b f s', b=x.shape[0], s=self.seq_size)
        
        # Step 4: 门控融合 + 残差
        x_fused = gate_weight * x_cov + (1 - gate_weight) * x_bi
        x_out = x_bi + self.res_scale * self.dropout(x_fused)
        
        return x_out