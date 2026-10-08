import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("RoPE 的 head_dim 必须是正偶数")
        if base <= 0:
            raise ValueError("base 必须大于 0")
        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, offset=0):
        if offset < 0 or x.size(-1) != self.inv_freq.numel() * 2:
            raise ValueError("RoPE 的 offset 或输入维度不合法")
        positions = torch.arange(offset, offset + x.size(-2), device=x.device)
        angles = positions.float()[:, None] * self.inv_freq.float()[None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
