"""独立运行 M1 tokenizer round-trip 自检。

用法：在 task-2-mini-gpt 目录执行 python eval/test_tokenizer_roundtrip.py。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tokenizer import BPETokenizer


def test_tokenizer_roundtrip():
    path = ROOT / "ckpt" / "tokenizer.json"
    if not path.exists():
        return {"test": "tokenizer_roundtrip", "pass": None,
                "skip": "ckpt/tokenizer.json 不存在；先运行 train_tokenizer.py"}

    try:
        tokenizer = BPETokenizer.from_pretrained(path)
        samples = [
            "床前明月光",
            "Hello, world!",
            "深度学习需要数学基础",
            "𠮷🙂 café\n\t",
            "",
            "<bos><eos><pad>",
        ]
        failures = []
        invalid_ids = []
        for text in samples:
            ids = tokenizer.encode(text)
            invalid_ids.extend(i for i in ids if not 0 <= i < tokenizer.vocab_size)
            decoded = tokenizer.decode(ids)
            if decoded != text:
                failures.append({"input": text, "output": decoded, "ids": ids})

        with_specials = tokenizer.encode("中文", add_bos=True, add_eos=True)
        special_roundtrip = tokenizer.decode(with_specials) == "中文"
        return {
            "test": "tokenizer_roundtrip",
            "pass": not failures and not invalid_ids and special_roundtrip,
            "vocab_size": tokenizer.vocab_size,
            "samples": len(samples),
            "failures": failures,
            "invalid_ids": invalid_ids,
            "special_tokens_pass": special_roundtrip,
        }
    except Exception as error:
        return {"test": "tokenizer_roundtrip", "pass": False,
                "error": f"{type(error).__name__}: {error}"}


if __name__ == "__main__":
    result = test_tokenizer_roundtrip()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["pass"] is not False else 1)
