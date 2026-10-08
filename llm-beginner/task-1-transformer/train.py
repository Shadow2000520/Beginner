import argparse
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import CharTokenizer, collate_batch
from src.model import TransformerClassifier

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ckpt")
    parser.add_argument("--train-limit", type=int, default=0, help="仅用于调试；0 表示全量")
    parser.add_argument("--dev-limit", type=int, default=0, help="仅用于调试；0 表示全量")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_length, args.n_layers, args.patience) < 1:
        parser.error("轮数、batch、序列长度、层数和 patience 必须为正数")
    if not 0 <= args.warmup_ratio < 1 or min(args.train_limit, args.dev_limit) < 0:
        parser.error("warmup-ratio 必须在 [0,1)，样本限制不能为负数")
    return args


def make_loader(frame, tokenizer, batch_size, shuffle):
    samples = [(tokenizer(row.text), int(row.label)) for row in frame.itertuples()]
    return DataLoader(samples, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_batch, num_workers=0)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids)
        total_loss += criterion(logits, labels).item() * len(labels)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        count += len(labels)
    return total_loss / count, correct / count


def save_curves(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["dev_loss"] for r in history], label="dev")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(epochs, [r["dev_accuracy"] for r in history])
    axes[1].axhline(0.8, color="gray", linestyle="--", label="M3 target")
    axes[1].set(xlabel="Epoch", ylabel="Dev accuracy", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无可用 CUDA，请改用 --device cpu 或检查 GPU 环境")
    train = pd.read_parquet(ROOT / "data" / "train.parquet")
    dev = pd.read_parquet(ROOT / "data" / "validation.parquet")
    if args.train_limit:
        train = train.sample(n=min(args.train_limit, len(train)), random_state=args.seed)
    if args.dev_limit:
        dev = dev.sample(n=min(args.dev_limit, len(dev)), random_state=args.seed)
    for frame in (train, dev):
        if frame.empty or frame[["text", "label"]].isna().any().any():
            raise ValueError("数据集不能为空，text/label 不能有空值")
        if not frame.label.isin([0, 1]).all():
            raise ValueError("label 必须是 0 或 1")
    tokenizer = CharTokenizer.build(train.text, max_length=args.max_length)
    train_loader = make_loader(train, tokenizer, args.batch_size, True)
    dev_loader = make_loader(dev, tokenizer, args.batch_size, False)
    config = dict(vocab_size=len(tokenizer.vocab), d_model=args.d_model,
                  n_heads=args.n_heads, n_layers=args.n_layers,
                  max_length=args.max_length, dropout=args.dropout, num_classes=2)
    model = TransformerClassifier(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_scale(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    criterion = nn.CrossEntropyLoss()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists():
        raise FileExistsError(f"{output}/best.pt 已存在，请用 --output-dir 指定新实验目录")
    metadata = {"model_config": config, "vocab": tokenizer.vocab,
                "train_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"device={device}, train={len(train)}, dev={len(dev)}, vocab={len(tokenizer.vocab)}", flush=True)
    if args.train_limit or args.dev_limit:
        print("调试模式：子集准确率不代表完整验证集上的 M3 成绩。", flush=True)
    best_acc, stale, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ids), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("训练 loss 非有限值，请检查学习率和数据")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
        dev_loss, accuracy = evaluate(model, dev_loader, criterion, device)
        history.append(dict(epoch=epoch, train_loss=loss_sum / count,
                            dev_loss=dev_loss, dev_accuracy=accuracy,
                            lr=optimizer.param_groups[0]["lr"]))
        if accuracy > best_acc:
            best_acc, stale = accuracy, 0
            torch.save(model.state_dict(), output / "best.pt")
        else:
            stale += 1
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch={epoch:02d} train_loss={loss_sum/count:.4f} dev_loss={dev_loss:.4f} "
              f"dev_acc={accuracy:.4f} best={best_acc:.4f}", flush=True)
        if stale >= args.patience:
            print("验证集准确率连续未提升，提前停止。", flush=True)
            break
    save_curves(history, output)
    print(f"训练结束，best_dev_acc={best_acc:.4f}；产物：{output}", flush=True)


if __name__ == "__main__":
    main()
