import torch
from torch import nn
import constants as cst

class BiN(nn.Module):
    def __init__(self, 
                 d1, 
                 t1, 
                 group_indices,  # LOB分组索引（仅用于生成掩码）
                 use_robust_stats=False  # 默认关闭鲁棒统计量，先对齐原始分数
                 ):
        super().__init__()
        self.t1 = t1
        self.d1 = d1
        self.group_indices = group_indices
        self.use_robust_stats = use_robust_stats

        # ========== 完全保留原始BiN的全局参数（核心！确保分数贴近原始） ==========
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

        # ========== 新增：LOB分组特征掩码（轻量级适配，不破坏原始逻辑） ==========
        # 对不同分组的特征应用不同的缩放系数（仅1个参数/分组，参数量少）
        self.group_masks = nn.ParameterDict()
        for group_name, indices in group_indices.items():
            # 每个分组的掩码系数（初始为1，不改变原始数据）
            self.group_masks[group_name] = nn.Parameter(torch.ones(len(indices), 1))

        # 预定义全1张量（优化效率，不改变逻辑）
        self.register_buffer('T1', torch.ones([d1, 1], device=cst.DEVICE))
        self.register_buffer('T2', torch.ones([t1, 1], device=cst.DEVICE))
        self.register_buffer('eps_tensor', torch.tensor(1e-4, device=cst.DEVICE))

    # 可选鲁棒统计量（默认关闭，如需开启再设置use_robust_stats=True）
    def _get_stats(self, x, dim):
        if self.use_robust_stats:
            mean = torch.median(x, dim=dim, keepdim=False).values
            q75 = torch.quantile(x, 0.75, dim=dim, keepdim=False)
            q25 = torch.quantile(x, 0.25, dim=dim, keepdim=False)
            std = q75 - q25
            std = torch.clamp(std, min=1e-4)
        else:
            mean = torch.mean(x, dim=dim, keepdim=False)
            std = torch.std(x, dim=dim, keepdim=False)
            std = torch.where(std < self.eps_tensor, torch.ones_like(std), std)
        return mean, std

    # 新增：应用分组掩码（轻量级适配LOB维度）
    def _apply_group_mask(self, x):
        """对不同LOB分组应用不同的缩放掩码，不改变原始归一化逻辑"""
        x_masked = torch.zeros_like(x)
        for group_name, indices in self.group_indices.items():
            mask = self.group_masks[group_name]  # [group_d1, 1]
            x_masked[:, indices, :] = x[:, indices, :] * mask  # 仅缩放该组特征
        return x_masked

    def forward(self, x):
        # ========== 完全保留原始BiN的核心归一化逻辑 ==========
        # 修复原始bug：用clamp约束y1/y2≥0.01，不修改Parameter
        y1 = torch.clamp(self.y1[0], min=0.01)
        y2 = torch.clamp(self.y2[0], min=0.01)

        # 时间维度归一化（完全复用原始逻辑）
        x2, std2 = self._get_stats(x, dim=2)
        x2 = torch.reshape(x2, (x2.shape[0], x2.shape[1], 1))
        std2 = torch.reshape(std2, (std2.shape[0], std2.shape[1], 1))
        
        diff = x - (x2 @ (self.T2.T))
        Z2 = diff / (std2 @ (self.T2.T))
        X2 = self.l2 @ self.T2.T
        X2 = X2 * Z2
        X2 = X2 + (self.B2 @ self.T2.T)

        # 特征维度归一化（完全复用原始逻辑）
        x1, std1 = self._get_stats(x, dim=1)
        x1 = torch.reshape(x1, (x1.shape[0], x1.shape[1], 1))
        std1 = torch.reshape(std1, (std1.shape[0], std1.shape[1], 1))
        
        op1 = x1 @ self.T1.T
        op1 = torch.permute(op1, (0, 2, 1))
        op2 = std1 @ self.T1.T
        op2 = torch.permute(op2, (0, 2, 1))
        z1 = (x - op1) / (op2)
        X1 = (self.T1 @ self.l1.T)
        X1 = X1 * z1
        X1 = X1 + (self.T1 @ self.B1.T)

        # ========== 轻量级适配：应用LOB分组掩码 ==========
        X1 = self._apply_group_mask(X1)
        X2 = self._apply_group_mask(X2)

        # 加权融合（完全保留原始逻辑）
        x = y1 * X1 + y2 * X2

        return x