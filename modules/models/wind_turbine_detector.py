"""
风力发电机组件检测模型
"""
import torch
import torch.nn as nn
from .components.backbone import Backbone
from .components.fpn import FPN
from .components.transformer import Transformer, PositionEmbeddingSine
from .heads.detection_head import DetectionHead
from .utils import box_ops, losses

class WindTurbineDetector(nn.Module):
    """风力发电机组件检测模型"""
    def __init__(self, num_classes=7, input_size=640, fpn_channels=256, num_queries=100):
        """
        Args:
            num_classes (int): 故障类别数量
            input_size (int): 输入图像大小
            fpn_channels (int): 特征金字塔网络的输出通道数
            num_queries (int): 目标查询的数量
        """
        super().__init__()
        
        # 特征提取
        self.backbone = Backbone()
        self.fpn = FPN(
            in_channels_list=[64, 128, 256, 512],  # 基于backbone的输出通道
            out_channels=fpn_channels
        )
        
        # 位置编码
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=fpn_channels // 2,
            normalize=True
        )
        
        # Transformer
        self.transformer = Transformer(
            d_model=fpn_channels,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        
        # 目标查询
        self.object_queries = nn.Embedding(num_queries, fpn_channels)
        
        # 检测头
        self.detection_head = DetectionHead(
            in_channels=fpn_channels,
            num_classes=num_classes
        )
        
        # 保存配置
        self.num_queries = num_queries
        self.input_size = input_size
        self.fpn_channels = fpn_channels

    def forward(self, samples):
        """
        Args:
            samples (tuple): (images, targets)
                - images (torch.Tensor): [B, C, H, W]
                - targets (list[dict]): 每个图像的标注信息
                    - boxes (torch.Tensor): [num_objects, 4]
                    - labels (torch.Tensor): [num_objects]
        
        Returns:
            dict: 
                训练模式下返回损失字典
                推理模式下返回预测结果
        """
        if isinstance(samples, tuple):
            images, targets = samples
        else:
            images = samples
            targets = None
        
        # 特征提取
        features = self.backbone(images)
        fpn_features = self.fpn(features)
        
        # 使用最后一层特征进行目标检测
        src = fpn_features[-1]
        
        # 位置编码
        pos_embed = self.position_embedding(src)
        
        # 准备Transformer输入
        batch_size = src.shape[0]
        src_flatten = src.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        pos_embed_flatten = pos_embed.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        
        # 准备目标查询
        query_embed = self.object_queries.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Transformer处理
        hs = self.transformer(src_flatten, query_embed, pos_embed_flatten)
        
        # 检测头预测
        class_logits, pred_boxes = self.detection_head(hs)
        
        # 准备输出
        outputs = {
            'pred_logits': class_logits,
            'pred_boxes': pred_boxes
        }
        
        if self.training:
            # 训练模式：计算损失
            assert targets is not None
            loss_dict = losses.compute_loss(outputs, targets, box_ops)
            return loss_dict
        else:
            # 推理模式：返回预测结果
            return outputs

    @torch.no_grad()
    def inference(self, images):
        """模型推理
        
        Args:
            images (torch.Tensor): [B, C, H, W] 输入图像
            
        Returns:
            list[dict]: 每张图像的检测结果
                - boxes (torch.Tensor): [num_det, 4] 检测框
                - scores (torch.Tensor): [num_det] 置信度
                - labels (torch.Tensor): [num_det] 类别
        """
        # 获取原始图像尺寸
        orig_sizes = torch.tensor([[img.shape[-2], img.shape[-1]] 
                                 for img in images], device=images.device)
        
        # 模型前向传播
        outputs = self.forward(images)
        
        # 后处理
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        # 将预测转换为概率
        prob = out_logits.sigmoid()
        
        # 对每个图像处理结果
        results = []
        for scores, boxes in zip(prob, out_bbox):
            # 获取最高分及其索引
            scores_per_image, labels_per_image = scores.max(dim=1)
            
            # 应用阈值
            keep = scores_per_image > 0.5  # 可配置的阈值
            
            # 保留高于阈值的预测
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            boxes_per_image = boxes[keep]
            
            results.append({
                'scores': scores_per_image,
                'labels': labels_per_image,
                'boxes': boxes_per_image
            })
        
        return results 