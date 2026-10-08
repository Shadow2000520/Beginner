import math

import torch


def validate_sampling(top_k, top_p, temperature):
    if type(top_k) is not int or top_k < 0:
        raise ValueError("top_k 必须是非负整数，0 表示不筛选")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p 必须在 (0, 1]")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature 必须是有限非负数")


def sampling_probs(logits, top_k=0, top_p=1.0, temperature=1.0):
    validate_sampling(top_k, top_p, temperature)
    if logits.ndim != 2 or logits.size(-1) == 0 or not logits.is_floating_point():
        raise ValueError("logits 必须是浮点 [batch, vocab_size] 张量")
    if (torch.isnan(logits).any() or torch.isposinf(logits).any()
            or not torch.isfinite(logits).any(dim=-1).all()):
        raise ValueError("每行至少需要一个有限 logit，且不能包含 NaN/+inf")
    if temperature == 0:
        return torch.zeros_like(logits).scatter(-1, logits.argmax(-1, keepdim=True), 1.0)
    scores = logits.float()
    scores = (scores - scores.max(dim=-1, keepdim=True).values) / max(temperature, torch.finfo(scores.dtype).tiny)
    if top_k:
        values, indices = scores.topk(min(top_k, scores.size(-1)), dim=-1)
        scores = torch.full_like(scores, float("-inf")).scatter(-1, indices, values)
    if top_p < 1:
        values, indices = scores.sort(dim=-1, descending=True)
        probs = values.softmax(dim=-1)
        remove = probs.cumsum(dim=-1) - probs >= top_p
        remove[:, 0] = False
        values = values.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(-1, indices, values)
    return scores.softmax(dim=-1)


def sample_next_token(logits, top_k=0, top_p=1.0, temperature=1.0, *, generator=None):
    probs = sampling_probs(logits, top_k, top_p, temperature)
    if temperature == 0:
        return probs.argmax(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1, generator=generator)
