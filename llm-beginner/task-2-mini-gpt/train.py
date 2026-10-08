import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.model import MiniGPT
from src.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="mini-GPT next-token 预训练")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-file", type=Path, default=ROOT / "data/train.txt")
    parser.add_argument("--dev-file", type=Path, default=ROOT / "data/dev.txt")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "ckpt/tokenizer.json")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=None,
                        help="仅用 train-file 重训 BPE，并与新模型一起保存；省略则使用已有词表")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ckpt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tie-weights", action="store_true", help="共享输入 embedding 和输出投影权重")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.tokenizer_vocab_size is not None and args.tokenizer_vocab_size < 259:
        parser.error("tokenizer-vocab-size 至少为 259")
    if min(args.epochs, args.steps_per_epoch, args.batch_size, args.block_size,
           args.d_model, args.n_heads, args.n_layers, args.patience) <= 0:
        parser.error("轮数、步数、batch、模型维度及 patience 必须为正数")
    if args.d_model % args.n_heads or (args.d_model // args.n_heads) % 2:
        parser.error("d-model 必须能被 n-heads 整除，且每个头的维度为偶数")
    if not 0 <= args.dropout < 1 or not 0 <= args.warmup_ratio < 1:
        parser.error("dropout 和 warmup-ratio 必须在 [0, 1)")
    if (not 0 < args.lr < math.inf or not 0 < args.grad_clip < math.inf
            or not 0 <= args.weight_decay < math.inf or not 0 <= args.min_lr_ratio <= 1):
        parser.error("学习率和梯度裁剪值必须为有限正数，weight-decay 非负，min-lr-ratio 在 [0, 1]")
    return args


def sample_batch(tokens, batch_size, block_size, generator):
    starts = torch.randint(len(tokens) - block_size, (batch_size,), generator=generator)
    positions = starts[:, None] + torch.arange(block_size)[None, :]
    return tokens[positions], tokens[positions + 1]


def learning_rate(step, total_steps, warmup_steps, peak_lr, min_ratio):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return peak_lr * (min_ratio + (1 - min_ratio) * cosine)


@torch.no_grad()
def evaluate(model, tokens, device):
    model.eval()
    nll, count = 0.0, 0
    for start in range(0, len(tokens) - 1, model.block_size):
        window = tokens[start:start + model.block_size + 1].unsqueeze(0).to(device)
        logits = model(window[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               window[:, 1:].reshape(-1), reduction="sum")
        nll += loss.item()
        count += window.size(1) - 1
    if count == 0:
        raise ValueError("验证集至少需要两个 token")
    result = nll / count
    if not math.isfinite(result):
        raise RuntimeError("验证集 loss 非有限值")
    return result


def perplexity(loss):
    return math.exp(loss) if loss < 709 else float("inf")


def save_curves(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["dev_loss"] for r in history], label="dev")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, [r["dev_ppl"] for r in history], label="dev")
    axes[1].set(xlabel="Epoch", ylabel="Perplexity", yscale="log")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无可用 CUDA，请检查 GPU 环境或使用 --device cpu")
    output = args.output_dir.resolve()
    if args.train_file.resolve() == args.dev_file.resolve():
        raise ValueError("训练集和验证集不能使用同一个文件")
    train_text = args.train_file.read_text(encoding="utf-8")
    dev_text = args.dev_file.read_text(encoding="utf-8")
    if args.tokenizer_vocab_size is not None:
        print("仅用训练集重建 BPE 词表……", flush=True)
        tokenizer = BPETokenizer.train(train_text, args.tokenizer_vocab_size, verbose=True)
    else:
        tokenizer = BPETokenizer.from_pretrained(args.tokenizer)
    target_tokenizer = output / "tokenizer.json"
    if target_tokenizer.exists() and args.tokenizer_vocab_size is None:
        existing = BPETokenizer.from_pretrained(target_tokenizer)
        if existing.merges != tokenizer.merges:
            raise ValueError("输出目录内已有不同词表，请指定新的实验目录")
    print("正在编码训练集和验证集……", flush=True)
    train_tokens = torch.tensor(tokenizer.encode(train_text))
    dev_tokens = torch.tensor(tokenizer.encode(dev_text))
    if len(train_tokens) <= args.block_size or len(dev_tokens) < 2:
        raise ValueError("训练集 token 数须大于 block-size，验证集至少需要两个 token")
    model = MiniGPT(tokenizer.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                    n_layers=args.n_layers, block_size=args.block_size, dropout=args.dropout,
                    tie_weights=args.tie_weights).to(device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "history.json", "metadata.json", "training_curves.png", "eval/result.json"):
        (output / name).unlink(missing_ok=True)
    if output == (ROOT / "ckpt").resolve():
        (ROOT / "eval/result.json").unlink(missing_ok=True)
    if args.tokenizer_vocab_size is not None or not target_tokenizer.exists():
        tokenizer.save_pretrained(target_tokenizer)
        target_tokenizer.with_suffix(".report.json").unlink(missing_ok=True)
    metadata = {"model_config": model.config,
                "train_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "train_tokens": len(train_tokens), "dev_tokens": len(dev_tokens),
                "epoch_definition": "steps_per_epoch random batches, sampled with replacement"}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"device={device}, vocab={tokenizer.vocab_size}, train_tokens={len(train_tokens)}, "
          f"dev_tokens={len(dev_tokens)}, parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"每轮随机采样 {args.steps_per_epoch} 个 batch；验证使用完整 dev 集。", flush=True)
    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    best_loss, best_epoch, stale, step = float("inf"), 0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(args.steps_per_epoch):
            lr = learning_rate(step, total_steps, warmup_steps, args.lr, args.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = sample_batch(train_tokens, args.batch_size, args.block_size, generator)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("训练 loss 非有限值，请检查学习率和数据")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item()
            step += 1
        dev_loss = evaluate(model, dev_tokens, device)
        row = dict(epoch=epoch, step=step, train_loss=loss_sum / args.steps_per_epoch,
                   dev_loss=dev_loss, dev_ppl=perplexity(dev_loss), lr=lr)
        history.append(row)
        if dev_loss < best_loss:
            best_loss, best_epoch, stale = dev_loss, epoch, 0
            torch.save({"config": model.config, "model_state_dict": model.state_dict(),
                        "epoch": epoch, "dev_loss": dev_loss}, output / "best.pt")
        else:
            stale += 1
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} dev_loss={dev_loss:.4f} "
              f"dev_ppl={row['dev_ppl']:.2f} best_ppl={perplexity(best_loss):.2f} lr={lr:.2e}", flush=True)
        if stale >= args.patience:
            print("验证集 loss 连续未下降，提前停止。", flush=True)
            break
    save_curves(history, output)
    print(f"训练结束，best_epoch={best_epoch}, best_dev_ppl={perplexity(best_loss):.2f}；产物：{output}")


if __name__ == "__main__":
    main()
