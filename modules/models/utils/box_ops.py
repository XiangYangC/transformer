"""
边界框操作相关的工具函数
"""
import torch
import torch.nn.functional as F

def box_cxcywh_to_xyxy(x):
    """将边界框从(center_x, center_y, width, height)格式转换为(x1, y1, x2, y2)格式
    
    Args:
        x (torch.Tensor): [..., 4] 格式为(cx, cy, w, h)的边界框
    
    Returns:
        torch.Tensor: [..., 4] 格式为(x1, y1, x2, y2)的边界框
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    """将边界框从(x1, y1, x2, y2)格式转换为(center_x, center_y, width, height)格式
    
    Args:
        x (torch.Tensor): [..., 4] 格式为(x1, y1, x2, y2)的边界框
    
    Returns:
        torch.Tensor: [..., 4] 格式为(cx, cy, w, h)的边界框
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_area(boxes):
    """计算边界框的面积
    
    Args:
        boxes (torch.Tensor): [..., 4] 格式为(x1, y1, x2, y2)的边界框
    
    Returns:
        torch.Tensor: [...] 边界框的面积
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])

def box_iou(boxes1, boxes2):
    """计算两组边界框之间的IoU
    
    Args:
        boxes1 (torch.Tensor): [N, 4] 第一组边界框
        boxes2 (torch.Tensor): [M, 4] 第二组边界框
    
    Returns:
        tuple: (iou, union)
            - iou (torch.Tensor): [N, M] IoU矩阵
            - union (torch.Tensor): [N, M] 并集面积矩阵
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """计算两组边界框之间的广义IoU
    
    Args:
        boxes1 (torch.Tensor): [N, 4] 第一组边界框
        boxes2 (torch.Tensor): [M, 4] 第二组边界框
    
    Returns:
        torch.Tensor: [N, M] 广义IoU矩阵
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area 