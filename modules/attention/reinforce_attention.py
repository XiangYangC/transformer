import torch
import torch.nn as nn
import torch.nn.functional as F
from .spatial_attention import SpatialAttention

class ReinforceAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, warmup_rounds=5):
        """强化学习注意力模块
        Args:
            in_channels (int): 输入通道数
            hidden_channels (int): 隐藏层通道数
            warmup_rounds (int): 预热轮数
        """
        super(ReinforceAttention, self).__init__()
        self.warmup_rounds = warmup_rounds
        self.round_counter = 0

        # 注意力层
        self.query = nn.Conv2d(in_channels, hidden_channels, 1)
        self.key = nn.Conv2d(in_channels, hidden_channels, 1)
        self.value = nn.Conv2d(in_channels, hidden_channels, 1)
        self.out_proj = nn.Conv2d(hidden_channels, in_channels, 1)
        self.spatial_attn = SpatialAttention()

        # 强化学习参数
        self.register_buffer('reward_score', torch.tensor(0.0))
        self.register_buffer('alpha', torch.tensor(1.0))

    def compute_alpha(self, reward_score):
        """计算注意力权重
        Args:
            reward_score (torch.Tensor): 奖励分数
        Returns:
            torch.Tensor: 注意力权重
        """
        return torch.sigmoid(reward_score / 10)

    def update_feedback(self, correct, confidence):
        """更新强化学习反馈
        Args:
            correct (bool): 是否正确
            confidence (float): 置信度
        """
        if self.round_counter > self.warmup_rounds:
            reward = (1 if correct else -10) * confidence
            self.reward_score = 0.9 * self.reward_score + 0.1 * reward
            self.alpha = self.compute_alpha(self.reward_score)

    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
        Returns:
            torch.Tensor: 注意力增强后的特征图 [B, C, H, W]
        """
        self.round_counter += 1
        B, C, H, W = x.size()

        # 计算Q、K、V
        Q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        K = self.key(x).view(B, -1, H * W)
        V = self.value(x).view(B, -1, H * W).permute(0, 2, 1)

        # 计算注意力
        attn = torch.bmm(Q, K) / (K.size(1) ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, V).permute(0, 2, 1).view(B, -1, H, W)
        out = self.out_proj(out)

        # 应用自适应融合
        out = self.alpha * out + (1 - self.alpha) * x
        # 应用空间注意力
        out = self.spatial_attn(out)
        
        return out 