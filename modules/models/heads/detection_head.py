"""
目标检测头模块
"""
import torch
import torch.nn as nn
import math

class DetectionHead(nn.Module):
    """目标检测头"""
    def __init__(self, in_channels, num_classes):
        """
        Args:
            in_channels (int): 输入通道数
            num_classes (int): 类别数
        """
        super().__init__()
        
        # 分类子网络
        self.class_subnet = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, num_classes + 1)  # +1 for background
        )
        
        # 边界框回归子网络
        self.bbox_subnet = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, 4)  # 4 for [cx, cy, w, h]
        )
        
        # 初始化
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_subnet[-1].bias, bias_value)
        
        # 边界框预测的缩放因子
        self.bbox_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [batch_size, num_queries, in_channels]
                Transformer解码器的输出
        
        Returns:
            tuple: (class_logits, bbox_pred)
                - class_logits: [batch_size, num_queries, num_classes + 1]
                - bbox_pred: [batch_size, num_queries, 4] (normalized cx,cy,w,h)
        """
        # 类别预测
        class_logits = self.class_subnet(x)
        
        # 边界框预测
        bbox_pred = self.bbox_subnet(x)
        bbox_pred = bbox_pred * self.bbox_scale.exp()
        bbox_pred = bbox_pred.sigmoid()  # 归一化到[0,1]
        
        return class_logits, bbox_pred 