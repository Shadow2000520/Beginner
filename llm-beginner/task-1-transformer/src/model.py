import json
from pathlib import Path

import torch
from torch import nn

from .block import TransformerBlock
from .data import CharTokenizer


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 max_length=256, dropout=0.1, num_classes=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, ids, *, return_weights=False):
        ids = ids.to(self.token_embedding.weight.device)
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        valid = ids.ne(0)
        mask = ~valid[:, None, None, :]  # True = PAD key，所有 query 都不能关注它
        attention_maps = []
        for block in self.blocks:
            if return_weights:
                x, weights = block(x, mask, return_weights=True)
                attention_maps.append(weights)
            else:
                x = block(x, mask)
        x = self.norm(x)
        # PAD query 虽然仍有输出，但不能参与最终平均池化。
        pooled = (x * valid.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / valid.sum(dim=1, keepdim=True).clamp_min(1)
        logits = self.classifier(pooled)
        return (logits, attention_maps) if return_weights else logits


def load_for_eval(ckpt_path):
    #best.pt 是纯 state_dict；同目录 metadata.json 保存架构和训练词表。
    path = Path(ckpt_path)
    metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
    model = TransformerClassifier(**metadata["model_config"])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    tokenizer = CharTokenizer(metadata["vocab"], metadata["model_config"]["max_length"])
    return model, tokenizer
