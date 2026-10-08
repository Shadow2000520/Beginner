from collections import Counter
import json
from pathlib import Path


def merge_pair(ids, pair, new_id):
    merged, index = [], 0
    while index < len(ids):
        if index + 1 < len(ids) and (ids[index], ids[index + 1]) == pair:
            merged.append(new_id)
            index += 2
        else:
            merged.append(ids[index])
            index += 1
    return merged


class BPETokenizer:
    pad_id, bos_id, eos_id = 256, 257, 258
    special_tokens = {"<pad>": 256, "<bos>": 257, "<eos>": 258}

    def __init__(self, merges=None):
        self.merges = []
        self._bytes = {i: bytes([i]) for i in range(256)}
        seen = set()
        for raw_pair in merges or []:
            if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
                raise ValueError("每条 merge 必须包含两个 token ID")
            pair = tuple(raw_pair)
            if any(type(i) is not int or i not in self._bytes for i in pair) or pair in seen:
                raise ValueError("merge 引用了未定义/特殊 token，或存在重复规则")
            new_id = 259 + len(self.merges)
            self._bytes[new_id] = self._bytes[pair[0]] + self._bytes[pair[1]]
            self.merges.append(pair)
            seen.add(pair)

    @property
    def vocab_size(self):
        return 259 + len(self.merges)

    def token_bytes(self, token_id):
        if token_id in self.special_tokens.values():
            return b""
        return self._bytes[token_id]

    @classmethod
    def train(cls, text, vocab_size=512, min_frequency=2, verbose=False):
        if vocab_size < 259 or min_frequency < 2:
            raise ValueError("vocab_size 至少为 259，min_frequency 至少为 2")
        if not text:
            raise ValueError("训练语料不能为空")
        ids = list(text.encode("utf-8"))
        merges = []
        for new_id in range(259, vocab_size):
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            if counts[pair] < min_frequency:
                break
            ids = merge_pair(ids, pair, new_id)
            merges.append(pair)
            if verbose and (len(merges) % 50 == 0 or new_id == vocab_size - 1):
                print(f"merges={len(merges)}, vocab={new_id+1}, tokens={len(ids)}", flush=True)
        return cls(merges)

    def encode(self, text, *, add_bos=False, add_eos=False):
        ids = list(text.encode("utf-8"))
        # 按训练顺序应用合并规则。
        for rank, pair in enumerate(self.merges):
            ids = merge_pair(ids, pair, 259 + rank)
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids, *, skip_special_tokens=True, errors="strict"):
        pieces = []
        specials = {value: key for key, value in self.special_tokens.items()}
        for token_id in ids:
            if token_id in specials:
                if not skip_special_tokens:
                    pieces.append(specials[token_id].encode("utf-8"))
            elif token_id in self._bytes:
                pieces.append(self._bytes[token_id])
            else:
                raise ValueError(f"未知 token ID: {token_id}")
        return b"".join(pieces).decode("utf-8", errors=errors)

    def save_pretrained(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format="llm-beginner-byte-bpe", version=1, vocab_size=self.vocab_size,
                       special_tokens=self.special_tokens, merges=self.merges)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "llm-beginner-byte-bpe" or payload.get("version") != 1:
            raise ValueError("不支持的 tokenizer 文件格式")
        if payload.get("special_tokens") != cls.special_tokens:
            raise ValueError("特殊 token 定义不匹配")
        tokenizer = cls(payload["merges"])
        if payload.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError("vocab_size 与 merge 表不匹配")
        return tokenizer
