"""
主干网络模块
"""
import torch
import torch.nn as nn

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 卷积"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 卷积"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation)

class Bottleneck(nn.Module):
    """ResNet风格的Bottleneck块"""
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion
        self.conv1 = conv1x1(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = conv3x3(mid_channels, mid_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = conv1x1(mid_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class Backbone(nn.Module):
    """特征提取主干网络"""
    def __init__(self):
        super().__init__()
        # 初始层
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ) # 输出P1 (相对于输入1/4)

        # 主干层
        self.layer2 = nn.Sequential(
            Bottleneck(64, 64, stride=1),
            Bottleneck(64, 64, stride=1),
            Bottleneck(64, 64, stride=1)
        ) # 输出P2 (相对于输入1/4)

        self.layer3 = nn.Sequential(
            Bottleneck(64, 128, stride=2),
            Bottleneck(128, 128, stride=1),
            Bottleneck(128, 128, stride=1),
            Bottleneck(128, 128, stride=1)
        ) # 输出P3 (相对于输入1/8)

        self.layer4 = nn.Sequential(
            Bottleneck(128, 256, stride=2),
            Bottleneck(256, 256, stride=1),
            Bottleneck(256, 256, stride=1),
            Bottleneck(256, 256, stride=1),
            Bottleneck(256, 256, stride=1),
            Bottleneck(256, 256, stride=1)
        ) # 输出P4 (相对于输入1/16)

        self.layer5 = nn.Sequential(
            Bottleneck(256, 512, stride=2),
            Bottleneck(512, 512, stride=1),
            Bottleneck(512, 512, stride=1)
        ) # 输出P5 (相对于输入1/32)

    def forward(self, x):
        c1 = self.layer1(x)     # P1特征, 1/4空间尺寸
        c2 = self.layer2(c1)    # P2特征, 1/4空间尺寸
        c3 = self.layer3(c2)    # P3特征, 1/8空间尺寸
        c4 = self.layer4(c3)    # P4特征, 1/16空间尺寸
        c5 = self.layer5(c4)    # P5特征, 1/32空间尺寸
        return [c2, c3, c4, c5] # 返回P2-P5用于FPN 