"""Headless tokenizer inspection and comparison."""

import argparse
import json
import sys

from tokenizer_utils import compare_tokenizers, load_tokenizer, parse_token_ids, sorted_vocab, vocab_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="查看、测试并比较 Hugging Face tokenizer")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="查看词表统计并搜索 token")
    inspect.add_argument("source")
    inspect.add_argument("--token", help="精确搜索 token")
    inspect.add_argument("--limit", type=int, default=10, help="显示前 N 个 token；0 表示不显示")
    encode = commands.add_parser("encode", help="编码文本")
    encode.add_argument("source")
    encode.add_argument("text")
    decode = commands.add_parser("decode", help="解码 JSON token ID 数组")
    decode.add_argument("source")
    decode.add_argument("ids")
    compare = commands.add_parser("compare", help="比较两个词表")
    compare.add_argument("first")
    compare.add_argument("second")
    compare.add_argument("--limit", type=int, default=10, help="每种差异显示前 N 项")
    args = parser.parse_args()

    if getattr(args, "limit", 0) < 0:
        parser.error("--limit 不能为负数")

    if args.command == "compare":
        result = compare_tokenizers(load_tokenizer(args.first), load_tokenizer(args.second))
        for key in ("only_first", "only_second", "different_ids"):
            rows = result[key]
            result[key + "_count"] = len(rows)
            result[key] = rows[: args.limit]
    else:
        tokenizer = load_tokenizer(args.source)
        if args.command == "inspect":
            vocab = tokenizer.get_vocab()
            result = {"source": args.source, **vocab_stats(tokenizer), "tokens": sorted_vocab(vocab)[: args.limit]}
            if args.token is not None:
                result["searched_token"] = args.token
                result["token_id"] = vocab.get(args.token)
        elif args.command == "encode":
            ids = tokenizer.encode(args.text)
            result = {"text": args.text, "ids": ids, "tokens": tokenizer.convert_ids_to_tokens(ids)}
        else:
            ids = parse_token_ids(args.ids)
            result = {"ids": ids, "tokens": tokenizer.convert_ids_to_tokens(ids), "text": tokenizer.decode(ids)}
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
