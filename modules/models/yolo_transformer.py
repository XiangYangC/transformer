import torch
import torch.nn as nn
import torch.nn.functional as F
from ..attention import EnhancedReinforceAttention
from torchvision.models.detection.image_list import ImageList
import copy
import math

# Helper function for 1x1 convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

# Helper function for 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

# Helper functions for loss computation
def box_cxcywh_to_xyxy(x):
    """Convert bbox coordinates from (center_x, center_y, width, height) to (x1, y1, x2, y2)"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (center_x, center_y, width, height)"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def box_area(boxes):
    """Compute area of boxes (x1, y1, x2, y2)"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def generalized_box_iou(boxes1, boxes2):
    """
    Compute the generalized IoU between two sets of boxes.
    Args:
        boxes1: (N, 4) (x1, y1, x2, y2)
        boxes2: (M, 4) (x1, y1, x2, y2)
    Returns:
        giou: (N, M)
    """
    # compute IoU
    iou, union = box_iou(boxes1, boxes2)

    # compute enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Focal loss for classification
    Args:
        inputs: (N, num_classes)
        targets: (N, )
        num_boxes: int, normalization factor
        alpha: float, weighting factor for rare class
        gamma: float, focusing parameter
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class Bottleneck(nn.Module):
    # Standard bottleneck block (similar to ResNet)
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
    # Simple CNN backbone to produce multi-scale features
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ) # Output P1 (relative to input 1/4)

        self.layer2 = nn.Sequential(
            Bottleneck(64, 64, stride=1),
            Bottleneck(64, 64, stride=1),
            Bottleneck(64, 64, stride=1)
        ) # Output P2 (relative to input 1/4)

        self.layer3 = nn.Sequential(
            Bottleneck(64, 128, stride=2),
            Bottleneck(128, 128, stride=1),
            Bottleneck(128, 128, stride=1),
            Bottleneck(128, 128, stride=1)
        ) # Output P3 (relative to input 1/8)

        self.layer4 = nn.Sequential(
            Bottleneck(128, 256, stride=2),
            Bottleneck(256, 256, stride=1),
            Bottleneck(256, 256, stride=1),
            Bottleneck(256, 256, stride=1),
            Bottleneck(256, 256, stride=1),
            Bottleneck(256, 256, stride=1)
        ) # Output P4 (relative to input 1/16)

        self.layer5 = nn.Sequential(
            Bottleneck(256, 512, stride=2),
            Bottleneck(512, 512, stride=1),
            Bottleneck(512, 512, stride=1)
        ) # Output P5 (relative to input 1/32)

    def forward(self, x):
        c1 = self.layer1(x) # P1 feature, 1/4 spatial size
        c2 = self.layer2(c1) # P2 feature, 1/4 spatial size
        c3 = self.layer3(c2) # P3 feature, 1/8 spatial size
        c4 = self.layer4(c3) # P4 feature, 1/16 spatial size
        c5 = self.layer5(c4) # P5 feature, 1/32 spatial size
        return [c2, c3, c4, c5] # Return P2, P3, P4, P5 for FPN

class FPN(nn.Module):
    # Feature Pyramid Network
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        for in_channels in in_channels_list:
            self.lateral_convs.append(conv1x1(in_channels, out_channels))
            self.fpn_convs.append(conv3x3(out_channels, out_channels))

    def forward(self, features):
        # features are [P2, P3, P4, P5]
        laterals = [lateral_conv(f) for lateral_conv, f in zip(self.lateral_convs, features)]

        # Build top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            # Ensure the sizes match for element-wise addition
            target_size = laterals[i-1].shape[-2:]
            laterals[i-1] += F.interpolate(laterals[i], size=target_size, mode='nearest')

        # Apply 3x3 convolution to each lateral connection
        fpn_outputs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]

        # Optionally add P6 and P7 if needed (e.g., for larger objects or different FPN variants)
        # P6 = MaxPool(P5, 1)
        # P7 = MaxPool(P6, 1)
        # fpn_outputs.append(P6)
        # fpn_outputs.append(P7)

        return fpn_outputs # Returns [P2_fpn, P3_fpn, P4_fpn, P5_fpn]

class Transformer(nn.Module):
    # Basic Transformer Encoder-Decoder (Simplified DETR-like)
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        # src: flattened FPN features [Batch_size, H*W, d_model]
        # query_embed: learnable object queries [num_queries, d_model]
        # pos_embed: positional encoding for src [Batch_size, H*W, d_model]

        # Add positional encoding to src
        src = src + pos_embed

        # Expand query_embed to match batch size
        query_embed = query_embed.unsqueeze(0).repeat(src.size(0), 1, 1)
        
        # Transformer Encoder
        memory = self.transformer_encoder(src)

        # Transformer Decoder
        # The query_embed acts as the initial input to the decoder
        # query_pos is added to the query_embed within the decoder layer (typical in DETR)
        hs = self.transformer_decoder(query_embed, memory) # hs: [Batch_size, num_queries, d_model]

        return hs

class PositionEmbeddingSine(nn.Module):
    """Position embedding from DETR"""
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None): # x: [B, C, H, W], mask: [B, H, W] (True for padding, False otherwise)
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos # [B, d_model, H, W]

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # 多层预测网络
        self.class_subnet = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, num_classes + 1)
        )
        
        # 边界框预测子网络
        self.bbox_subnet = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, 4)
        )
        
        # 初始化最后一层
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_subnet[-1].bias, bias_value)
        
        # 边界框预测的缩放因子
        self.bbox_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, transformer_outputs):
        # 类别预测
        class_logits = self.class_subnet(transformer_outputs)
        
        # 边界框预测（使用缩放因子）
        bbox_pred = self.bbox_subnet(transformer_outputs)
        bbox_pred = bbox_pred * self.bbox_scale.exp()
        bboxes = bbox_pred.sigmoid()
        
        return class_logits, bboxes

class WindTurbineDetector(nn.Module):
    def __init__(self, num_classes=7, input_size=640, fpn_channels=256, num_queries=100):
        """风力发电机组件检测的轻量级网络
        
        Args:
            num_classes (int): 故障类别数量（默认7类：burning, crack, deformity, dirt, oil, peeling, rusty）
            input_size (int): 输入图像大小
            fpn_channels (int): 特征金字塔网络的输出通道数
            num_queries (int): 对象查询的数量
        """
        super().__init__()
        
        self.num_queries = num_queries
        self.fpn_channels = fpn_channels # Store fpn_channels

        self.backbone = Backbone()
        # Adjust in_channels_list based on the actual output channels of the Backbone's layers
        # Based on the Backbone definition, the output channels are 64 (P2), 128 (P3), 256 (P4), 512 (P5)
        self.fpn = FPN(in_channels_list=[64, 128, 256, 512], out_channels=fpn_channels)

        # Positional encoding for FPN features
        # Use the same number of positional features as FPN output channels
        self.position_embedding = PositionEmbeddingSine(num_pos_feats=fpn_channels // 2, normalize=True)

        # Transformer
        self.transformer = Transformer(d_model=fpn_channels, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

        # Learnable object queries and positional queries
        self.object_queries = nn.Embedding(num_queries, fpn_channels)
        # The query positional embedding is typically added inside the Transformer Decoder layer
        # self.query_pos = nn.Embedding(num_queries, fpn_channels) # Not needed as separate layer if added inside decoder
        
        # Detection Head
        self.detection_head = DetectionHead(in_channels=fpn_channels, num_classes=num_classes)

        # 添加多尺度特征融合
        self.multi_scale_attention = nn.ModuleList([
            nn.MultiheadAttention(fpn_channels, 8, dropout=0.1)
            for _ in range(4)  # 对应P2-P5四个尺度
        ])
        
        # 添加特征融合后的卷积
        self.fusion_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])

    def forward(self, samples): # samples is expected to be an ImageList (from torchvision)
        # Assuming samples is a tuple (images, targets) from the DataLoader with collate_fn
        # We need to adapt this to handle the ImageList format if using torchvision components
        
        # For now, let's assume samples is a batch of images [B, C, H, W]
        # and targets is a list of dictionaries (as produced by our collate_fn)
        # We need to adapt the forward pass to take images and optionally targets (during training)
        
        images, targets = samples # Assuming samples is (images, targets) tuple

        # Backbone to get multi-scale features
        features = self.backbone(images) # Returns [P2, P3, P4, P5]

        # FPN to fuse features
        fpn_features = self.fpn(features) # Returns [P2_fpn, P3_fpn, P4_fpn, P5_fpn]

        # 多尺度特征融合
        batch_size = fpn_features[0].shape[0]
        fused_features = []
        
        for idx, feat in enumerate(fpn_features):
            # 将特征展平
            hw = feat.shape[-2] * feat.shape[-1]
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # [B, HW, C]
            
            # 对每个尺度进行自注意力
            attn_output, _ = self.multi_scale_attention[idx](
                feat_flat, feat_flat, feat_flat
            )
            
            # 重塑回特征图形状
            attn_output = attn_output.permute(0, 2, 1).reshape(
                batch_size, -1, feat.shape[-2], feat.shape[-1]
            )
            
            # 通过融合卷积
            fused_feat = self.fusion_conv[idx](attn_output)
            fused_features.append(fused_feat)
        
        # 选择最合适的特征级别或组合多个级别
        src = fused_features[-1]  # 使用P5特征，或者可以组合多个级别

        # Generate positional encoding for the source features
        # Needs a mask if there's padding (not handled yet)
        pos_embed = self.position_embedding(src)

        # Flatten the source features and positional encoding for the Transformer
        # src_flattened: [Batch_size, H*W, d_model]
        # pos_embed_flattened: [Batch_size, H*W, d_model]
        batch_size, channels, h, w = src.shape
        src_flattened = src.flatten(2).permute(0, 2, 1) # [B, H*W, C]
        pos_embed_flattened = pos_embed.flatten(2).permute(0, 2, 1) # [B, H*W, C]

        # Get object queries (query_embed) and positional queries (query_pos)
        query_embed = self.object_queries.weight # [num_queries, d_model]
        # query_pos is typically added inside the decoder, but here we pass it explicitly for clarity
        # If the Transformer module adds query_pos internally, we don't need a separate query_pos embedding here.
        # Let's assume the Transformer module handles adding query_pos internally based on query_embed.

        # Pass features and queries to the Transformer
        # hs: [Batch_size, num_queries, d_model]
        hs = self.transformer(src_flattened, query_embed, pos_embed_flattened) # Assuming transformer takes src, query_embed, pos_embed

        # Pass Transformer outputs to the Detection Head
        # class_logits: [Batch_size, num_queries, num_classes + 1]
        # bboxes: [Batch_size, num_queries, 4] (normalized cx, cy, w, h)
        class_logits, bboxes = self.detection_head(hs)

        # For training, return class_logits, bboxes, and targets
        # For inference, post-process class_logits and bboxes to get final detections

        # During training, we need to compute loss
        # During inference, we need to post-process the outputs

        outputs = {
            'pred_logits': class_logits,
            'pred_boxes': bboxes,
            'multi_scale_features': fused_features  # 保存多尺度特征用于后续处理
        }

        if self.training:
            # In training mode, compute and return loss
            # The targets from the DataLoader are a list of dictionaries
            loss_dict = self.compute_loss(outputs, targets) # Need to implement compute_loss
            return loss_dict
        else:
            # In inference mode, return the raw outputs (or post-processed detections)
            # Post-processing would involve converting bboxes to absolute coordinates,
            # applying confidence threshold, and non-maximum suppression (if needed, though DETR-like models aim to reduce duplicates).
            return outputs

    def compute_loss(self, outputs, targets):
        """计算目标检测的损失函数
        
        Args:
            outputs (dict): 模型输出，包含 'pred_logits' 和 'pred_boxes'
                - pred_logits: [batch_size, num_queries, num_classes + 1]
                - pred_boxes: [batch_size, num_queries, 4] (normalized cx,cy,w,h)
            targets (list): 每个图像的目标，每个目标是一个字典，包含：
                - boxes: [num_objects, 4] (x1,y1,x2,y2) in absolute coordinates
                - labels: [num_objects] class indices
                
        Returns:
            dict: 包含各个损失项的字典
        """
        # 提取预测结果
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        # 获取批次大小
        batch_size = len(targets)
        
        # 初始化损失字典
        losses = {}
        
        # 对批次中的每个样本分别计算损失
        for batch_idx in range(batch_size):
            # 获取当前样本的预测和目标
            sample_logits = pred_logits[batch_idx]  # [num_queries, num_classes + 1]
            sample_boxes = pred_boxes[batch_idx]     # [num_queries, 4]
            target = targets[batch_idx]
            
            target_boxes = target['boxes']      # [num_objects, 4]
            target_labels = target['labels']    # [num_objects]
            
            # 如果当前图像没有目标物体，跳过损失计算
            if len(target_boxes) == 0:
                continue
                
            # 将预测框从 (cx,cy,w,h) 转换为 (x1,y1,x2,y2) 格式
            pred_boxes_xyxy = box_cxcywh_to_xyxy(sample_boxes)
            
            # 计算预测框和真实框之间的成本矩阵
            # 分类成本：使用交叉熵
            cost_class = -sample_logits[:, target_labels]
            
            # 边界框回归成本：L1距离 + GIoU
            cost_bbox = torch.cdist(pred_boxes_xyxy, target_boxes, p=1)
            cost_giou = -generalized_box_iou(pred_boxes_xyxy, target_boxes)
            
            # 组合成本
            C = cost_class + 5 * cost_bbox + 2 * cost_giou
            
            # 使用匈牙利算法进行匹配（这里简化为贪婪匹配）
            # 对每个GT框选择成本最小的预测框
            matched_indices = torch.min(C, dim=0)[1]  # [num_objects]
            
            # 计算分类损失（Focal Loss）
            target_classes = torch.full(sample_logits.shape[:1], pred_logits.shape[-1]-1,
                                     dtype=torch.int64, device=pred_logits.device)
            target_classes[matched_indices] = target_labels
            
            target_classes_onehot = torch.zeros_like(sample_logits)
            target_classes_onehot.scatter_(1, target_classes.unsqueeze(-1), 1)
            
            num_boxes = len(target_boxes)
            class_loss = sigmoid_focal_loss(sample_logits, target_classes_onehot, num_boxes)
            
            # 计算边界框损失
            src_boxes = sample_boxes[matched_indices]
            target_boxes_cxcywh = box_xyxy_to_cxcywh(target_boxes)
            
            # L1 loss
            box_loss = F.l1_loss(src_boxes, target_boxes_cxcywh, reduction='none').sum(1).mean()
            
            # GIoU loss
            giou_loss = (1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                target_boxes))).mean()
            
            # 累加到总损失
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

    @torch.no_grad()
    def post_process(self, outputs, target_sizes):
        """后处理函数，将预测结果转换为绝对坐标
        
        Args:
            outputs (dict): 模型输出
            target_sizes (torch.Tensor): 原始图像尺寸 [batch_size, 2]
            
        Returns:
            list[dict]: 每张图像的检测结果，包含：
                - boxes: [num_boxes, 4] 绝对坐标
                - scores: [num_boxes]
                - labels: [num_boxes]
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        
        # 从相对坐标转换为绝对坐标
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        
        results = []
        for s, l, b in zip(scores, labels, boxes):
            # 应用阈值
            keep = s > 0.5  # 可配置的阈值
            results.append({
                'scores': s[keep],
                'labels': l[keep],
                'boxes': b[keep]
            })
        
        return results

    def inference(self, images):
        """推理函数
        
        Args:
            images (torch.Tensor): [B, C, H, W]
            
        Returns:
            list[dict]: 检测结果
        """
        # 获取原始图像尺寸
        orig_sizes = torch.tensor([[img.shape[-2], img.shape[-1]] for img in images],
                                device=images.device)
        
        # 模型前向传播
        outputs = self.forward((images, None))
        
        # 后处理
        results = self.post_process(outputs, orig_sizes)
        
        return results

# The old WindTurbineNet class definition is removed 