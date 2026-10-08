import codecs
import math
from functools import lru_cache

import torch


@lru_cache(maxsize=8192)
def utf8_transition(pending, piece):
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        text = decoder.decode(pending + piece, final=False)
    except UnicodeDecodeError:
        return None
    return text, decoder.getstate()[0]


class TextConstraints:
    def __init__(self, tokenizer, repetition_penalty=1.1, no_repeat_ngram_size=4):
        if not math.isfinite(repetition_penalty) or repetition_penalty < 1:
            raise ValueError("repetition_penalty 必须为大于等于 1 的有限数")
        if type(no_repeat_ngram_size) is not int or no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size 必须为非负整数")
        self.pieces = [tokenizer.token_bytes(i) for i in range(tokenizer.vocab_size)]
        self.penalty = repetition_penalty
        self.ngram_size = no_repeat_ngram_size

    def __call__(self, scores, ids, remaining_tokens):
        result = scores.clone()
        for row, history in enumerate(ids.tolist()):
            raw = b"".join(self.pieces[i] for i in history)
            decoded = utf8_transition(b"", raw)
            if decoded is None:
                raise ValueError("输入历史包含无效 UTF-8")
            text, pending = decoded
            seen_chars = set(text)
            n = self.ngram_size
            seen_ngrams = {text[i:i+n] for i in range(len(text)-n+1)} if n else set()
            valid, repeats = [], []
            for piece in self.pieces:
                transition = utf8_transition(pending, piece) if piece else None
                if transition is None:
                    valid.append(False)
                    repeats.append(False)
                    continue
                added, tail = transition
                needed = 0
                if tail:
                    total = 2 if tail[0] < 0xE0 else 3 if tail[0] < 0xF0 else 4
                    needed = total - len(tail)
                valid.append(needed <= remaining_tokens)
                duplicate = False
                if n:
                    joined = text + added
                    grams = set(seen_ngrams)
                    for i in range(max(0, len(text)-n+1), len(joined)-n+1):
                        gram = joined[i:i+n]
                        duplicate |= gram in grams
                        grams.add(gram)
                repeats.append(duplicate)
                if any(c in seen_chars and c.isalnum() for c in added):
                    index = len(valid)-1
                    value = result[row, index]
                    result[row, index] = torch.where(value < 0, value * self.penalty, value / self.penalty)
            valid = torch.tensor(valid, device=scores.device)
            repeats = torch.tensor(repeats, device=scores.device)
            allowed = valid & ~repeats & torch.isfinite(scores[row])
            if not allowed.any():
                raise RuntimeError("没有满足 UTF-8 和重复约束的候选，请减小 no-repeat-ngram-size")
            result[row] = result[row].masked_fill(~allowed, float("-inf"))
        return result
