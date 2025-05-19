import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from .spatial_attention import SpatialAttention
import matplotlib.pyplot as plt
import numpy as np

# V2.1
class EnhancedReinforceAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, warmup_rounds=5, num_heads=8):
        """增强版强化学习注意力模块
        Args:
            in_channels (int): 输入通道数
            hidden_channels (int): 隐藏层通道数
            warmup_rounds (int): 预热轮数
            num_heads (int): 注意力头数
        """
        super().__init__()
        self.warmup_rounds = warmup_rounds
        self.round_counter = 0
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.debug_mode = False

        # 确保hidden_channels能被num_heads整除
        assert hidden_channels % num_heads == 0, f"hidden_channels ({hidden_channels}) must be divisible by num_heads ({num_heads})"

        # 多头注意力层
        self.query = nn.Conv2d(in_channels, hidden_channels, 1)
        self.key = nn.Conv2d(in_channels, hidden_channels, 1)
        self.value = nn.Conv2d(in_channels, hidden_channels, 1)
        self.out_proj = nn.Conv2d(hidden_channels, in_channels, 1)
        
        # 修改层归一化实现
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, in_channels)
        
        # 空间注意力
        self.spatial_attn = SpatialAttention()
        
        # 强化学习参数
        self.register_buffer('reward_score', torch.tensor(0.0))
        self.register_buffer('alpha', torch.tensor(1.0))
        self.temperature = nn.Parameter(torch.ones(1))
        
        # 历史记录跟踪
        self.reward_history = []
        self.confidence_history = []
        self.alpha_history = []
        
        # 学习参数
        self.learning_rate = 0.1
        self.min_alpha = 0.1
        self.max_alpha = 0.9
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _attention_forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """注意力计算
        Args:
            Q: Query张量 [B, num_heads, head_dim, H*W]
            K: Key张量 [B, num_heads, head_dim, H*W]
            V: Value张量 [B, num_heads, head_dim, H*W]
            scale: 缩放因子 (Tensor)
        Returns:
            torch.Tensor: 注意力输出 [B, num_heads, head_dim, H*W]
        """
        # 调整维度以进行批量矩阵乘法
        Q = Q.transpose(-2, -1)  # [B, num_heads, H*W, head_dim]
        K = K  # [B, num_heads, head_dim, H*W]
        V = V  # [B, num_heads, head_dim, H*W]

        # 计算注意力分数
        attn = torch.matmul(Q, K) * scale  # [B, num_heads, H*W, H*W]
        attn = F.softmax(attn, dim=-1)
        
        # 计算输出
        out = torch.matmul(attn, V.transpose(-2, -1))  # [B, num_heads, H*W, head_dim]
        out = out.transpose(-2, -1)  # [B, num_heads, head_dim, H*W]
        
        return out

    def compute_alpha(self, reward_score):
        """改进的alpha计算
        Args:
            reward_score (torch.Tensor): 奖励分数
        Returns:
            torch.Tensor: 注意力权重
        """
        base_alpha = torch.sigmoid(reward_score * self.learning_rate)
        return base_alpha.clamp(self.min_alpha, self.max_alpha)

    def update_feedback(self, correct, confidence):
        """增强的反馈更新机制
        Args:
            correct (bool): 是否正确
            confidence (float): 置信度
        """
        if self.round_counter > self.warmup_rounds:
            # 计算动态奖励
            base_reward = 1 if correct else -10
            confidence_factor = min(max(confidence, 0.1), 0.9)
            reward = base_reward * confidence_factor
            
            # 指数移动平均更新
            beta = 0.9
            self.reward_score = beta * self.reward_score + (1 - beta) * reward
            
            # 记录历史
            self.reward_history.append(reward)
            self.confidence_history.append(confidence)
            
            # 自适应alpha更新
            new_alpha = self.compute_alpha(self.reward_score)
            self.alpha = torch.clamp(new_alpha, self.min_alpha, self.max_alpha)
            self.alpha_history.append(self.alpha.item())
            
            # 动态调整学习率
            if len(self.reward_history) > 10:
                recent_rewards = self.reward_history[-10:]
                if all(r > 0 for r in recent_rewards):
                    self.learning_rate *= 0.95  # 降低学习率
                elif all(r < 0 for r in recent_rewards):
                    self.learning_rate *= 1.05  # 提高学习率
            
            # 自适应预热
            self.adaptive_warmup()

    def adaptive_warmup(self):
        """自适应调整预热轮数"""
        if len(self.reward_history) > self.warmup_rounds:
            recent_rewards = self.reward_history[-self.warmup_rounds:]
            if all(r > 0 for r in recent_rewards):
                self.warmup_rounds = max(self.warmup_rounds - 1, 3)
            elif all(r < 0 for r in recent_rewards):
                self.warmup_rounds += 1

    def _forward_impl(self, x):
        """实际的前向传播实现"""
        B, C, H, W = x.size()
        
        # 应用层归一化
        x = self.norm1(x)
        
        # 多头注意力计算
        head_dim = self.hidden_channels // self.num_heads
        Q = self.query(x).view(B, self.num_heads, head_dim, H * W)
        K = self.key(x).view(B, self.num_heads, head_dim, H * W)
        V = self.value(x).view(B, self.num_heads, head_dim, H * W)
        
        # 缩放点积注意力
        scale = torch.tensor(1.0 / math.sqrt(head_dim), device=x.device, dtype=x.dtype)
        out = self._attention_forward(Q, K, V, scale)
        
        # 重塑并投影
        out = out.reshape(B, self.hidden_channels, H, W)
        out = self.out_proj(out)
        
        # 残差连接
        out = out + x
        out = self.norm2(out)
        
        # 自适应融合
        out = self.alpha * out + (1 - self.alpha) * x
        out = self.spatial_attn(out)
        
        return out

    def forward(self, x):
        """前向传播
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
        Returns:
            torch.Tensor: 注意力增强后的特征图 [B, C, H, W]
        """
        self.round_counter += 1
        
        if self.debug_mode:
            self._debug_print(x)
            
        # 使用梯度检查点减少内存使用
        return checkpoint(self._forward_impl, x, use_reentrant=False)

    def _debug_print(self, x):
        """调试信息打印"""
        print(f"Debug Info:")
        print(f"Input shape: {x.shape}")
        print(f"Current alpha: {self.alpha.item():.3f}")
        print(f"Current reward score: {self.reward_score.item():.3f}")
        print(f"Learning rate: {self.learning_rate:.3f}")
        print(f"Warmup rounds: {self.warmup_rounds}")

    def get_attention_stats(self):
        """获取注意力统计信息"""
        return {
            'reward_score': self.reward_score.item(),
            'alpha': self.alpha.item(),
            'round_counter': self.round_counter,
            'learning_rate': self.learning_rate,
            'warmup_rounds': self.warmup_rounds,
            'reward_history': self.reward_history[-10:],
            'confidence_history': self.confidence_history[-10:],
            'alpha_history': self.alpha_history[-10:]
        }

    def visualize_attention(self, x):
        """可视化注意力图
        Args:
            x (torch.Tensor): 输入特征图
        Returns:
            np.ndarray: 注意力热力图
        """
        with torch.no_grad():
            B, C, H, W = x.size()
            Q = self.query(x).view(B, -1, H * W)
            K = self.key(x).view(B, -1, H * W)
            attn = torch.bmm(Q, K.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            attention_map = attn.view(B, -1, H, W)
            
            # 转换为热力图
            attention_map = attention_map.mean(1)  # 平均所有头
            attention_map = attention_map.cpu().numpy()
            
            return attention_map

    def plot_training_history(self, save_path=None):
        """绘制训练历史
        Args:
            save_path (str, optional): 保存路径
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # 绘制奖励历史
        ax1.plot(self.reward_history)
        ax1.set_title('Reward History')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # 绘制置信度历史
        ax2.plot(self.confidence_history)
        ax2.set_title('Confidence History')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Confidence')
        ax2.grid(True)
        
        # 绘制alpha历史
        ax3.plot(self.alpha_history)
        ax3.set_title('Alpha History')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Alpha')
        ax3.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

def test_attention():
    """测试注意力模块"""
    # 创建测试数据
    batch_size = 4
    channels = 64
    height = width = 32
    x = torch.randn(batch_size, channels, height, width)
    
    # 创建注意力模块
    attention = EnhancedReinforceAttention(
        in_channels=channels,
        hidden_channels=128,
        warmup_rounds=5,
        num_heads=8
    )
    
    # 测试前向传播
    out = attention(x)
    print(f"Output shape: {out.shape}")
    
    # 测试反馈更新
    attention.update_feedback(correct=True, confidence=0.8)
    
    # 测试注意力统计
    stats = attention.get_attention_stats()
    print("\nAttention stats:")
    for k, v in stats.items():
        print(f"{k}: {v}")
    
    # 测试可视化
    attention_map = attention.visualize_attention(x)
    print(f"\nAttention map shape: {attention_map.shape}")
    
    return attention

if __name__ == '__main__':
    test_attention() 