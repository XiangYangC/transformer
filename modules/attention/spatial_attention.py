import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """空间注意力模块
        Args:
            kernel_size (int): 卷积核大小
        """
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
        Returns:
            torch.Tensor: 注意力加权后的特征图 [B, C, H, W]
        """
        # 计算通道维度的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接并通过卷积层
        out = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(out))
        return x * attn 