"""
Transformer编解码器模块
"""
import torch
import torch.nn as nn
import math

class PositionEmbeddingSine(nn.Module):
    """正弦位置编码"""
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """
        Args:
            num_pos_feats (int): 位置特征的维度的一半
            temperature (float): 缩放因子
            normalize (bool): 是否归一化
            scale (float): 归一化时的缩放因子
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): [B, C, H, W] 输入特征图
            mask (torch.Tensor): [B, H, W] 掩码（True表示padding）
        
        Returns:
            torch.Tensor: [B, d_model, H, W] 位置编码
        """
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), 
                             dtype=torch.bool, device=x.device)
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
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), 
                           pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), 
                           pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class Transformer(nn.Module):
    """Transformer编解码器"""
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        Args:
            d_model (int): 模型维度
            nhead (int): 注意力头数
            num_encoder_layers (int): 编码器层数
            num_decoder_layers (int): 解码器层数
            dim_feedforward (int): 前馈网络维度
            dropout (float): Dropout率
        """
        super().__init__()
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_encoder_layers
        )
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_decoder_layers
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        """
        Args:
            src (torch.Tensor): [B, HW, d_model] 展平的特征图
            query_embed (torch.Tensor): [B, num_queries, d_model] 目标查询
            pos_embed (torch.Tensor): [B, HW, d_model] 位置编码
        
        Returns:
            torch.Tensor: [B, num_queries, d_model] 解码器输出
        """
        # 添加位置编码
        src = src + pos_embed
        
        # Transformer编码器
        memory = self.transformer_encoder(src)
        
        # Transformer解码器
        # query_embed作为解码器的初始输入
        hs = self.transformer_decoder(query_embed, memory)
        
        return hs 