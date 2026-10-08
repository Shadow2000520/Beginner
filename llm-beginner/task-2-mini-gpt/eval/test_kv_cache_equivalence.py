import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import MiniGPT


@torch.no_grad()
def test_kv_cache_equivalence():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = MiniGPT(vocab_size=259, d_model=32, n_heads=4, n_layers=2,
                        block_size=32).eval()
        ids = torch.randint(0, 259, (2, 19))
        full = model(ids)
        errors = {}
        for name, chunks in (("token_by_token", [1] * 19), ("prefill_and_chunks", [7, 3, 1, 8])):
            cache, outputs, offset = None, [], 0
            for length in chunks:
                out, cache = model(ids[:, offset:offset + length], cache, True)
                outputs.append(out)
                offset += length
            errors[name] = (full - torch.cat(outputs, dim=1)).abs().max().item()
    return {"test": "kv_cache_equivalence", "pass": max(errors.values()) < 1e-4,
            "max_abs_diff": max(errors.values()), "details": errors}


if __name__ == "__main__":
    result = test_kv_cache_equivalence()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["pass"] else 1)
