import math

import torch
from torch import nn

from .rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, rope_base=10000.0):
        super().__init__()
        if n_heads <= 0 or d_model <= 0 or d_model % n_heads:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim, rope_base)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None, return_cache=False):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        past_length = 0
        if kv_cache is not None:
            if not isinstance(kv_cache, (tuple, list)) or len(kv_cache) != 2:
                raise ValueError("每层 cache 必须是 (K, V)")
            past_k, past_v = kv_cache
            expected = (batch, self.n_heads)
            if (past_k.ndim != 4 or past_k.shape != past_v.shape
                    or past_k.shape[:2] != expected or past_k.size(-1) != self.head_dim):
                raise ValueError("cache 形状必须为 [batch, heads, past_length, head_dim]")
            if any(t.device != k.device or t.dtype != k.dtype for t in kv_cache):
                raise ValueError("cache 的 device/dtype 必须与当前 K/V 一致")
            past_length = past_k.size(2)
        q, k = self.rope(q, past_length), self.rope(k, past_length)
        if kv_cache is not None:
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        query_positions = past_length + torch.arange(length, device=x.device)
        key_positions = torch.arange(k.size(2), device=x.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
        out = (self.attn_dropout(weights) @ v).transpose(1, 2).contiguous().view(batch, length, width)
        out = self.resid_dropout(self.proj(out))
        return (out, (k, v)) if return_cache else out
