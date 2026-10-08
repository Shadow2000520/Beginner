import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from data import DATASETS, TASK_ROOT, load_splits
from models import GraphClassifier


def parser():
    p = argparse.ArgumentParser(description="GNN graph classification")
    p.add_argument("--dataset", choices=DATASETS, default="MUTAG")
    p.add_argument("--model", choices=["gcn", "gat", "sage", "gin"], default="gcn")
    p.add_argument("--pooling", choices=["mean", "max", "min"], default="mean")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--zinc-bins", type=int, default=10)
    p.add_argument("--max-train-graphs", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--data-root", type=Path, default=TASK_ROOT / "data")
    p.add_argument("--output", type=Path, default=TASK_ROOT / "results" / "single")
    return p


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device(name)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def classification_metrics(logits, labels, classes):
    predictions = logits.argmax(dim=-1)
    matrix = torch.bincount(
        labels * classes + predictions,
        minlength=classes * classes,
    ).reshape(classes, classes).float()
    true_positive = matrix.diag()
    f1 = 2 * true_positive / (matrix.sum(0) + matrix.sum(1)).clamp_min(1)
    return {
        "accuracy": float(true_positive.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()),
    }


def evaluate(model, loader, device, classes):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits_list.append(model(batch.x.float(), batch.edge_index, batch.batch))
            labels_list.append(batch.y.view(-1).long())
    return classification_metrics(
        torch.cat(logits_list), torch.cat(labels_list), classes)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run(args):
    if min(args.epochs, args.eval_every, args.layers, args.batch_size, args.threads) < 1:
        raise ValueError("epochs, layers and batch size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.json").exists():
        raise FileExistsError(f"Result exists: {args.output}")

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    save_json(args.output / "config.json", config)
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    train_set, val_set, test_set, features, classes = load_splits(
        args.dataset,
        args.data_root,
        args.seed,
        args.zinc_bins,
        args.max_train_graphs,
    )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = GraphClassifier(
        args.model, args.pooling, features, args.hidden, classes,
        args.layers, args.dropout, args.heads).to(device)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val = -1.0
    best_epoch = 0
    best_state = None
    train_seconds = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        sync(device)
        start = time.perf_counter()
        total_loss = 0.0
        total_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(batch.x.float(), batch.edge_index, batch.batch)
            labels = batch.y.view(-1).long()
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimiser.step()
            total_loss += float(loss.detach()) * batch.num_graphs
            total_graphs += batch.num_graphs
        sync(device)
        duration = time.perf_counter() - start
        train_seconds += duration

        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_graphs,
            "train_seconds": duration,
            "val_accuracy": "",
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(model, val_loader, device, classes)
            row["val_accuracy"] = val["accuracy"]
            if val["accuracy"] > best_val:
                best_val = val["accuracy"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"{args.dataset}/{args.model}/{args.pooling} epoch={epoch} "
                  f"loss={row['train_loss']:.4f} val={best_val:.4f}", flush=True)

    model.load_state_dict(best_state)
    val = evaluate(model, val_loader, device, classes)
    test = evaluate(model, test_loader, device, classes)
    result = {
        **config,
        "status": "ok",
        "device_used": str(device),
        "train_graphs": len(train_set),
        "val_graphs": len(val_set),
        "test_graphs": len(test_set),
        "features": features,
        "classes": classes,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "val_accuracy": val["accuracy"],
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "train_seconds": train_seconds,
        "mean_epoch_seconds": train_seconds / len(history),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pyg": torch_geometric.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }

    torch.save({
        "state_dict": best_state,
        "config": config,
        "in_channels": features,
        "out_channels": classes,
    }, args.output / "best.pt")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    save_json(args.output / "metrics.json", result)
    print(json.dumps({k: result[k] for k in (
        "test_accuracy", "test_macro_f1", "train_seconds")}), flush=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
