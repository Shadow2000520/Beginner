from pathlib import Path

import torch
from torch import nn

from .attention import CausalSelfAttention
from .tokenizer import BPETokenizer
from .sampling import sample_next_token, validate_sampling


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, rope_base):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, rope_base)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x, kv_cache=None, return_cache=False):
        result = self.attn(self.ln1(x), kv_cache, return_cache)
        attention, cache = result if return_cache else (result, None)
        x = x + attention
        x = x + self.ffn(self.ln2(x))
        return (x, cache) if return_cache else x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 block_size=256, d_ff=None, dropout=0.1, rope_base=10000.0,
                 tie_weights=False):
        super().__init__()
        d_ff = 4 * d_model if d_ff is None else d_ff
        if min(vocab_size, n_layers, block_size, d_ff) <= 0:
            raise ValueError("词表、层数、上下文长度及 d_ff 必须为正数")
        self.config = dict(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                           n_layers=n_layers, block_size=block_size, d_ff=d_ff,
                           dropout=dropout, rope_base=rope_base, tie_weights=tie_weights)
        self.block_size = self.max_seq_len = block_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, rope_base)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        if tie_weights:
            self.lm_head.weight = self.embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, kv_cache=None, return_cache=False):
        if ids.ndim != 2 or ids.size(1) == 0:
            raise ValueError("ids 必须为非空的 [batch, length] 张量")
        caches = [None] * len(self.blocks) if kv_cache is None else kv_cache
        if len(caches) != len(self.blocks):
            raise ValueError("cache 层数与模型不一致")
        lengths = []
        for cache in caches:
            if cache is None:
                lengths.append(0)
            elif (isinstance(cache, (tuple, list)) and len(cache) == 2
                  and all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in cache)):
                lengths.append(cache[0].size(2))
            else:
                raise ValueError("每层 cache 必须包含两个四维张量")
        if len(set(lengths)) != 1:
            raise ValueError("各层 cache 的历史长度必须一致")
        if lengths[0] + ids.size(1) > self.block_size:
            raise ValueError("输入加 cache 的长度超过 block_size")
        x = self.dropout(self.embedding(ids))
        new_cache = []
        for block, cache in zip(self.blocks, caches):
            if return_cache:
                x, updated = block(x, cache, True)
                new_cache.append(updated)
            else:
                x = block(x, cache)
        logits = self.lm_head(self.ln_f(x))
        return (logits, new_cache) if return_cache else logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=50, top_k=0, top_p=1.0,
                 temperature=0.0, *, use_cache=True, generator=None, forbidden_ids=(),
                 logits_processor=None):
        """超过上下文窗口时，用最近的 token 重建 cache；返回包含 prompt 的序列。"""
        validate_sampling(top_k, top_p, temperature)
        if type(max_new_tokens) is not int:
            raise ValueError("max_new_tokens 必须是整数")
        if prompt_ids.ndim != 2 or prompt_ids.size(1) == 0 or max_new_tokens < 0:
            raise ValueError("prompt_ids 必须是非空二维张量，max_new_tokens 不能为负")
        forbidden_ids = list(forbidden_ids)
        if any(type(i) is not int or not 0 <= i < self.config["vocab_size"] for i in forbidden_ids):
            raise ValueError("forbidden_ids 包含无效 token ID")
        if len(set(forbidden_ids)) >= self.config["vocab_size"]:
            raise ValueError("不能禁止所有 token")
        was_training = self.training
        self.eval()
        ids, cache = prompt_ids, None
        try:
            for step in range(max_new_tokens):
                if use_cache:
                    if cache is None or cache[0][0].size(2) >= self.block_size:
                        logits, cache = self(ids[:, -self.block_size:], return_cache=True)
                    else:
                        logits, cache = self(ids[:, -1:], kv_cache=cache, return_cache=True)
                else:
                    logits = self(ids[:, -self.block_size:])
                scores = logits[:, -1].clone()
                scores[:, forbidden_ids] = float("-inf")
                if logits_processor is not None:
                    scores = logits_processor(scores, ids, max_new_tokens - step - 1)
                token = sample_next_token(scores, top_k, top_p, temperature, generator=generator)
                ids = torch.cat((ids, token), dim=1)
        finally:
            self.train(was_training)
        return ids


def load_for_eval(ckpt_path):
    ckpt_path = Path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    tokenizer = BPETokenizer.from_pretrained(ckpt_path.parent / "tokenizer.json")
    model = MiniGPT(**checkpoint["config"])
    if model.config["vocab_size"] != tokenizer.vocab_size:
        raise ValueError("模型词表大小与 tokenizer 不一致")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval(), tokenizer
