def forward(self, x):
    # 添加多尺度特征提取
    scales = [1.0, 0.75, 0.5]  # 多尺度分析
    outputs = []
    for scale in scales:
        # 在不同尺度下提取特征
        scaled_x = F.interpolate(x, scale_factor=scale)
        out = self._forward_impl(scaled_x)
        outputs.append(F.interpolate(out, size=x.shape[2:]))
    return sum(outputs) / len(outputs)

def _attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: torch.Tensor) -> torch.Tensor: 