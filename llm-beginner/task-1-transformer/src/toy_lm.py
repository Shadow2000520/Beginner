import torch
from torch import nn

from .block import TransformerBlock


class ToyLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 block_size=64, dropout=0.1):
        super().__init__()
        if block_size < 1 or n_layers < 1:
            raise ValueError("block_size 和 n_layers 必须为正数")
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        # True = 不可见；对角线保留，因此位置 t 可以看到自身。
        self.register_buffer("causal_mask", torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1), persistent=False)

    def forward(self, ids):
        length = ids.size(1)
        if not 1 <= length <= self.block_size:
            raise ValueError(f"输入长度必须在 1 到 {self.block_size} 之间")
        positions = torch.arange(length, device=ids.device)
        x = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        mask = self.causal_mask[:length, :length] | ids.eq(0)[:, None, None, :]
        for block in self.blocks:
            x = block(x, mask)
        # 不做平均池化：每个位置都预测下一个字符，输出 (B, T, vocab_size)。
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=80, temperature=0.8):
        if temperature < 0 or max_new_tokens < 0 or ids.size(1) == 0:
            raise ValueError("temperature/max_new_tokens 不能为负，prompt 不能为空")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # toy 模型重算最近的上下文；KV cache 留到任务二实现。
                logits = self(ids[:, -self.block_size:])[:, -1].clone()
                logits[:, :2] = -torch.inf  # 不生成 PAD / UNK
                if temperature == 0:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    next_id = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
                ids = torch.cat([ids, next_id], dim=1)
            return ids
        finally:
            self.train(was_training)


@torch.no_grad()
def causal_leakage(model, ids):
    if ids.size(1) < 2:
        raise ValueError("验证因果性至少需要两个字符")
    was_training = model.training
    model.eval()
    try:
        split = ids.size(1) // 2
        changed = ids.clone()
        vocab_size = model.token_embedding.num_embeddings
        changed[:, split:] = (changed[:, split:] - 2 + 1) % (vocab_size - 2) + 2
        return (model(ids)[:, :split] - model(changed)[:, :split]).abs().max().item()
    finally:
        model.train(was_training)
