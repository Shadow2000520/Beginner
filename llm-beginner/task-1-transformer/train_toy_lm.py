import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import CharTokenizer
from src.toy_lm import ToyLanguageModel, causal_leakage

ROOT = Path(__file__).resolve().parent


def make_examples(text, tokenizer, block_size):
    # 整篇编码不调用 tokenizer(text)，避免其 max_length 截断语料。
    ids = torch.tensor([tokenizer.vocab.get(char, 1) for char in text], dtype=torch.long)
    examples = []
    for start in range(0, len(ids) - 1, block_size):
        window = ids[start:start + block_size + 1]
        examples.append((window[:-1], window[1:]))
    return examples


def collate_lm(batch):
    inputs, targets = zip(*batch)
    return (torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0),
            torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100))


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_nll, total_tokens = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                  ignore_index=-100, reduction="sum")
            tokens = targets.ne(-100).sum()
            loss = nll / tokens
            if not torch.isfinite(loss):
                raise RuntimeError("loss 出现非有限值")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_nll += nll.item()
            total_tokens += tokens.item()
    return total_nll / total_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, default=ROOT.parent / "poetryFromTang.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ckpt" / "toy-lm")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="春")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.block_size) < 1 or args.lr <= 0:
        parser.error("epochs/batch-size/block-size/lr 必须为正数")
    if not args.prompt or args.temperature < 0 or args.max_new_tokens < 0:
        parser.error("prompt 不能为空，temperature/max-new-tokens 不能为负数")
    torch.manual_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 PyTorch 或使用 --device cpu")
    text = args.text.read_text(encoding="utf-8")
    if len(text) < 100:
        raise ValueError("语料至少需要 100 个字符")
    # 先切原始文本，再建窗口；训练和验证不共享字符位置。
    split = int(len(text) * 0.9)
    train_text, dev_text = text[:split], text[split:]
    tokenizer = CharTokenizer.build([train_text], max_length=args.block_size)
    if len(tokenizer.vocab) < 4:
        raise ValueError("训练语料至少需要两种不同字符")
    loaders = [DataLoader(make_examples(part, tokenizer, args.block_size),
                          batch_size=args.batch_size, shuffle=index == 0, collate_fn=collate_lm)
               for index, part in enumerate((train_text, dev_text))]
    config = dict(vocab_size=len(tokenizer.vocab), d_model=args.d_model, n_heads=args.n_heads,
                  n_layers=args.n_layers, block_size=args.block_size)
    model = ToyLanguageModel(**config).to(device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists():
        raise FileExistsError("输出目录已有 best.pt，请用 --output-dir 选择新目录")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    metadata = dict(model_config=config, vocab=tokenizer.vocab,
                    train_chars=len(train_text), dev_chars=len(dev_text),
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"device={device}, vocab={len(tokenizer.vocab)}, train_chars={len(train_text)}, dev_chars={len(dev_text)}", flush=True)
    history, best = [], math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders[0], device, optimizer)
        dev_loss = run_epoch(model, loaders[1], device)
        history.append(dict(epoch=epoch, train_loss=train_loss, dev_loss=dev_loss,
                            dev_perplexity=math.exp(min(dev_loss, 700))))
        if dev_loss < best:
            best = dev_loss
            torch.save(model.state_dict(), output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} "
              f"dev_ppl={history[-1]['dev_perplexity']:.2f}", flush=True)
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
    probe = torch.tensor([[2, 3] * min(4, args.block_size // 2)], device=device)
    leakage = causal_leakage(model, probe) if args.block_size >= 2 else None
    prompt = torch.tensor([[tokenizer.vocab.get(char, 1) for char in args.prompt]], device=device)
    generated = model.generate(prompt, args.max_new_tokens, args.temperature)[0].tolist()
    inverse = {idx: char for char, idx in tokenizer.vocab.items()}
    sample = args.prompt + "".join(inverse[idx] for idx in generated[len(args.prompt):])
    (output / "sample.txt").write_text(sample, encoding="utf-8")
    result = dict(causal_leaked_diff=leakage, causal_pass=leakage < 1e-6 if leakage is not None else None,
                  best_dev_loss=best, best_dev_perplexity=math.exp(min(best, 700)))
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"causal_leaked_diff={leakage}\n{sample}\n产物：{output}", flush=True)


if __name__ == "__main__":
    main()
