import torch
import torch.nn as nn
from ..attention import EnhancedReinforceAttention

class WindTurbineNet(nn.Module):
    def __init__(self, num_classes=7, input_size=640):
        """风力发电机组件检测的轻量级网络
        
        Args:
            num_classes (int): 故障类别数量（默认7类：burning, crack, deformity, dirt, oil, peeling, rusty）
            input_size (int): 输入图像大小
        """
        super().__init__()
        
        # 特征提取主干网络
        self.backbone = nn.Sequential(
            # 第一层：标准卷积层
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 320x320
            
            # 第二层：深度可分离卷积（降低参数量）
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 160x160
            
            # 第三层：注意力增强的特征提取
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 80x80
            
            # 第四层：多尺度特征融合
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 40x40
        )
        
        # 使用增强版注意力机制
        self.reinforce_attn = EnhancedReinforceAttention(
            in_channels=256,
            hidden_channels=256,  # 修改为与输入通道数相同
            num_heads=8,  # 使用8个头以确保每个头的维度为32
            warmup_rounds=5
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 故障检测头
        self.detection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入图像 [B, 3, H, W]
            
        Returns:
            torch.Tensor: 故障类别预测 [B, num_classes]
        """
        # 特征提取
        features = self.backbone(x)
        
        # 应用注意力机制
        features = self.reinforce_attn(features)
        
        # 全局池化
        features = self.global_pool(features)
        
        # 故障检测
        output = self.detection_head(features)
        
        return output
    
    def compute_loss(self, predictions, targets):
        """计算损失函数
        
        Args:
            predictions (torch.Tensor): 模型预测输出 [B, num_classes]
            targets (torch.Tensor): 目标标签 [B]
            
        Returns:
            torch.Tensor: 损失值
        """
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predictions, targets)
        
        # 更新强化学习反馈
        with torch.no_grad():
            accuracy = (predictions.argmax(dim=1) == targets).float().mean()
            self.reinforce_attn.update_feedback(
                correct=(accuracy > 0.7),
                confidence=torch.softmax(predictions, dim=1).max(dim=1)[0].mean().item()
            )
        
        return loss 