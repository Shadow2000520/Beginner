import argparse
import hashlib
import json
from pathlib import Path
import runpy

from src.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, default=ROOT / "data/train.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "ckpt/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} 已存在，请用 --output 指定新文件，以免与已有模型词表冲突")
    text = args.text.read_text(encoding="utf-8")
    tokenizer = BPETokenizer.train(text, args.vocab_size, verbose=True)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    tokenizer.save_pretrained(args.output)
    loaded = BPETokenizer.from_pretrained(args.output)
    for sample in ["床前明月光", "Hello, world!", "深度学习需要数学基础", "𠮷🙂\n\t", "", "<bos>"]:
        assert loaded.decode(loaded.encode(sample)) == sample
    result = dict(vocab_size=loaded.vocab_size, merges=len(loaded.merges),
                  training_file=str(args.text.resolve()),
                  training_sha256=hashlib.sha256(args.text.read_bytes()).hexdigest(),
                  training_bytes=len(text.encode("utf-8")), training_tokens=len(loaded.encode(text)),
                  roundtrip_pass=True)
    if args.output.resolve() == (ROOT / "ckpt/tokenizer.json").resolve():
        tests = runpy.run_path(str(ROOT / "eval/run.py"))
        result["official_test"] = tests["test_tokenizer_roundtrip"]()
        assert result["official_test"]["pass"]
    args.output.with_suffix(".report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"保存到 {args.output.resolve()}")


if __name__ == "__main__":
    main()
