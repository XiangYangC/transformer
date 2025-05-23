"""
特征金字塔网络模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import conv1x1, conv3x3

class FPN(nn.Module):
    """特征金字塔网络"""
    def __init__(self, in_channels_list, out_channels):
        """
        Args:
            in_channels_list (list): 输入特征图的通道数列表
            out_channels (int): 输出特征图的通道数
        """
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        # 创建横向连接和FPN卷积
        for in_channels in in_channels_list:
            self.lateral_convs.append(conv1x1(in_channels, out_channels))
            self.fpn_convs.append(conv3x3(out_channels, out_channels))

    def forward(self, features):
        """
        Args:
            features (list): 主干网络输出的特征图列表 [P2, P3, P4, P5]
        
        Returns:
            list: FPN处理后的特征图列表 [P2_fpn, P3_fpn, P4_fpn, P5_fpn]
        """
        # 应用横向连接
        laterals = [lateral_conv(f) 
                   for lateral_conv, f in zip(self.lateral_convs, features)]

        # 自顶向下的路径
        for i in range(len(laterals) - 1, 0, -1):
            # 确保尺寸匹配以进行元素级加法
            target_size = laterals[i-1].shape[-2:]
            laterals[i-1] += F.interpolate(laterals[i], 
                                         size=target_size, 
                                         mode='nearest')

        # 对每个横向连接应用3x3卷积
        fpn_outputs = [fpn_conv(lateral) 
                      for fpn_conv, lateral in zip(self.fpn_convs, laterals)]

        return fpn_outputs 