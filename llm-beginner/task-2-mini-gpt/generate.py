import argparse
import json
import sys
from pathlib import Path

import torch

from src.model import load_for_eval
from src.sampling import validate_sampling
from src.decoding import TextConstraints

ROOT = Path(__file__).resolve().parent


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="mini-GPT 文本生成与采样对比")
    parser.add_argument("--ckpt", type=Path, default=ROOT / "ckpt/best.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prompt", default="春江花月夜，")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--strategy", choices=["greedy", "temperature", "top-k", "top-p"], default="top-p")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare", action="store_true", help="同一 prompt 对比四种策略")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4, help="禁止重复的连续字符数；0 关闭")
    parser.add_argument("--raw-sampling", action="store_true", help="关闭 UTF-8 与字符重复约束，用于原始策略对比")
    parser.add_argument("--output", type=Path, help="指定后保存 JSON；默认只打印")
    args = parser.parse_args()
    if not args.prompt or args.max_new_tokens < 0:
        parser.error("prompt 不能为空，max-new-tokens 不能为负")
    try:
        validate_sampling(args.top_k, args.top_p, args.temperature)
    except ValueError as error:
        parser.error(str(error))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("当前 PyTorch 无可用 CUDA，请使用 --device cpu 或检查 GPU 环境")
    model, tokenizer = load_for_eval(args.ckpt)
    model = model.to(device)
    try:
        constraints = TextConstraints(tokenizer, args.repetition_penalty, args.no_repeat_ngram_size)
    except ValueError as error:
        parser.error(str(error))
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    settings = {
        "greedy": dict(temperature=0.0, top_k=0, top_p=1.0),
        "temperature": dict(temperature=args.temperature, top_k=0, top_p=1.0),
        "top-k": dict(temperature=args.temperature, top_k=args.top_k, top_p=1.0),
        "top-p": dict(temperature=args.temperature, top_k=0, top_p=args.top_p),
    }
    samples = []
    for strategy in settings if args.compare else [args.strategy]:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        ids = model.generate(prompt, args.max_new_tokens, **settings[strategy],
                             use_cache=not args.no_cache, generator=generator,
                             forbidden_ids=tokenizer.special_tokens.values(),
                             logits_processor=None if args.raw_sampling else constraints)[0].tolist()
        try:
            text = tokenizer.decode(ids)
            valid_utf8 = True
        except UnicodeDecodeError:
            text = tokenizer.decode(ids, errors="replace")
            valid_utf8 = False
        row = dict(strategy=strategy, **settings[strategy], text=text,
                   valid_utf8=valid_utf8, token_ids=ids)
        samples.append(row)
        print(f"\n[{strategy}] {settings[strategy]}\n{text}")
        if not valid_utf8:
            print("[提示] 输出含无效或末尾不完整的 UTF-8 字节，显示时以替换符呈现。")
    if args.output:
        payload = dict(checkpoint=str(args.ckpt.resolve()), vocab_size=tokenizer.vocab_size,
                       prompt=args.prompt, seed=args.seed, device=str(device),
                       max_new_tokens=args.max_new_tokens, use_cache=not args.no_cache, samples=samples)
        payload.update(text_constraints=not args.raw_sampling, repetition_penalty=args.repetition_penalty,
                       no_repeat_ngram_size=args.no_repeat_ngram_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
