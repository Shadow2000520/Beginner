from torch import nn

from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """先规范化再计算子层；两次残差连接保持 (B, T, d_model) 形状。

    本模块接收词向量；词嵌入、位置编码及分类头由后续完整模型负责。
    """

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model if d_ff is None else d_ff),
            nn.GELU(),
            nn.Linear(4 * d_model if d_ff is None else d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, *, return_weights=False):
        attention_output, weights = self.attention(self.norm1(x), mask, return_weights=True)
        x = x + self.dropout(attention_output)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return (x, weights) if return_weights else x
