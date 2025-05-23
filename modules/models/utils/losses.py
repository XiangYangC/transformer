"""
损失函数相关的工具
"""
import torch
import torch.nn.functional as F

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """Focal Loss for classification
    
    Args:
        inputs (torch.Tensor): [N, num_classes] 预测logits
        targets (torch.Tensor): [N, num_classes] one-hot标签
        num_boxes (int): 用于归一化的框数量
        alpha (float): 权重因子
        gamma (float): 聚焦参数
    
    Returns:
        torch.Tensor: 标量损失值
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def compute_loss(outputs, targets, box_ops):
    """计算目标检测的总损失
    
    Args:
        outputs (dict): 模型输出
            - pred_logits: [batch_size, num_queries, num_classes + 1]
            - pred_boxes: [batch_size, num_queries, 4]
        targets (list[dict]): 每个图像的目标
            - boxes: [num_objects, 4]
            - labels: [num_objects]
        box_ops: 边界框操作模块
    
    Returns:
        dict: 损失字典
    """
    pred_logits = outputs['pred_logits']
    pred_boxes = outputs['pred_boxes']
    batch_size = len(targets)
    losses = {}
    
    for batch_idx in range(batch_size):
        sample_logits = pred_logits[batch_idx]
        sample_boxes = pred_boxes[batch_idx]
        target = targets[batch_idx]
        
        target_boxes = target['boxes']
        target_labels = target['labels']
        
        if len(target_boxes) == 0:
            continue
            
        pred_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(sample_boxes)
        
        # 计算成本矩阵
        cost_class = -sample_logits[:, target_labels]
        cost_bbox = torch.cdist(pred_boxes_xyxy, target_boxes, p=1)
        cost_giou = -box_ops.generalized_box_iou(pred_boxes_xyxy, target_boxes)
        
        # 组合成本
        C = cost_class + 5 * cost_bbox + 2 * cost_giou
        
        # 匹配
        matched_indices = torch.min(C, dim=0)[1]
        
        # 分类损失
        target_classes = torch.full(sample_logits.shape[:1], 
                                 pred_logits.shape[-1]-1,
                                 dtype=torch.int64, 
                                 device=pred_logits.device)
        target_classes[matched_indices] = target_labels
        
        target_classes_onehot = torch.zeros_like(sample_logits)
        target_classes_onehot.scatter_(1, target_classes.unsqueeze(-1), 1)
        
        num_boxes = len(target_boxes)
        class_loss = sigmoid_focal_loss(sample_logits, target_classes_onehot, num_boxes)
        
        # 边界框损失
        src_boxes = sample_boxes[matched_indices]
        target_boxes_cxcywh = box_ops.box_xyxy_to_cxcywh(target_boxes)
        
        # L1 loss
        box_loss = F.l1_loss(src_boxes, target_boxes_cxcywh, reduction='none').sum(1).mean()
        
        # GIoU loss
        giou_loss = (1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            target_boxes))).mean()
        
        # 累加损失
        losses.setdefault('loss_ce', 0)
        losses.setdefault('loss_bbox', 0)
        losses.setdefault('loss_giou', 0)
        
        losses['loss_ce'] += class_loss
        losses['loss_bbox'] += box_loss
        losses['loss_giou'] += giou_loss
    
    # 计算平均损失
    for k in losses.keys():
        losses[k] = losses[k] / batch_size
    
    # 计算总损失
    losses['total_loss'] = (
        losses['loss_ce'] * 2 +
        losses['loss_bbox'] * 5 +
        losses['loss_giou'] * 2
    )
    
    return losses 