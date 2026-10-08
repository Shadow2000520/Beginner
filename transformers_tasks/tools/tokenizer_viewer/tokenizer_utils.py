"""Reusable, side-effect-free helpers for the tokenizer viewer."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


def load_tokenizer(source: str):
    """Load a local directory or Hugging Face model ID without remote code."""
    source = source.strip()
    if not source:
        raise ValueError("Tokenizer 路径或模型 ID 不能为空。")
    local = Path(source).expanduser()
    if local.exists():
        if not local.is_dir():
            raise ValueError(f"Tokenizer 路径不是目录：{local}")
        return AutoTokenizer.from_pretrained(str(local), local_files_only=True, trust_remote_code=False)
    return AutoTokenizer.from_pretrained(source, trust_remote_code=False)


def parse_token_ids(value: str) -> list[int]:
    """Accept only a JSON array of non-negative integer token IDs."""
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("请输入 JSON 整数数组，例如 [0, 1, 2]。") from exc
    if not isinstance(result, list) or any(type(item) is not int or item < 0 for item in result):
        raise ValueError("Token IDs 必须是非负整数构成的 JSON 数组。")
    return result


def vocab_stats(tokenizer) -> dict[str, int]:
    vocab = tokenizer.get_vocab()
    added = tokenizer.get_added_vocab()
    added_count = len(set(vocab).intersection(added))
    return {"base_vocab": len(vocab) - added_count, "added_vocab": added_count, "total_vocab": len(vocab)}


def sorted_vocab(vocab: dict[str, int]) -> list[dict[str, object]]:
    return [{"id": index, "token": token} for token, index in sorted(vocab.items(), key=lambda pair: (pair[1], pair[0]))]


def compare_tokenizers(first, second) -> dict[str, object]:
    left, right = first.get_vocab(), second.get_vocab()
    only_left = {token: index for token, index in left.items() if token not in right}
    only_right = {token: index for token, index in right.items() if token not in left}
    changed = [
        {"token": token, "first_id": left[token], "second_id": right[token]}
        for token in left.keys() & right.keys()
        if left[token] != right[token]
    ]
    changed.sort(key=lambda row: (row["first_id"], row["token"]))
    return {
        "first_stats": vocab_stats(first),
        "second_stats": vocab_stats(second),
        "only_first": sorted_vocab(only_left),
        "only_second": sorted_vocab(only_right),
        "different_ids": changed,
    }
