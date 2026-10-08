import math

import torch
from torch import nn


def scaled_dot_product_attention(Q, K, V, mask=None, *, return_weights=False):

    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("mask 必须是布尔张量，True 表示屏蔽")
        scores = scores.masked_fill(mask, float("-inf"))
        # 全屏蔽行先用有限值计算softmax，再将该行权重全部置零。
        all_masked = mask.all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_masked, 0.0)
    # 沿key归一化
    weights = torch.softmax(scores, dim=-1)
    if mask is not None:
        weights = weights.masked_fill(mask, 0.0)
    output = weights @ V
    return (output, weights) if return_weights else output


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model 和 n_heads 必须为正数，且 d_model 能被 n_heads 整除")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, *, return_weights=False):
        batch, length, d_model = x.shape

        def split_heads(tensor):
            # (B, T, d_model) - (B, T, H, d_k) - (B, H, T, d_k)
            return tensor.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q_proj(x))
        K = split_heads(self.k_proj(x))
        V = split_heads(self.v_proj(x))
        context, weights = scaled_dot_product_attention(Q, K, V, mask, return_weights=True)
        # 合并 head。
        context = context.transpose(1, 2).contiguous().view(batch, length, d_model)
        output = self.out_proj(context)
        return (output, weights) if return_weights else output
