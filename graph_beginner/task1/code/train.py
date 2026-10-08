"""Train one model, select by validation accuracy, then evaluate test once."""
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
from torch_geometric.loader import NeighborLoader

from data import DATASETS, TASK_ROOT, describe, load_data
from models import NodeClassifier


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, default="Cora")
    p.add_argument("--model", choices=["gcn", "gat", "sage", "gin"], default="gcn")
    p.add_argument("--mode", choices=["full", "sampled"], default="full")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=0, help="Validation checks; 0 disables early stopping")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--neighbors", type=int, nargs="+", default=[15, 10])
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
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def metrics(logits, labels, classes):
    pred = logits.argmax(dim=-1)
    cm = torch.bincount(labels * classes + pred, minlength=classes * classes).reshape(classes, classes).float()
    tp = cm.diag()
    f1 = 2 * tp / (cm.sum(0) + cm.sum(1)).clamp_min(1)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean())}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args):
    if min(args.epochs, args.eval_every, args.layers, args.batch_size, args.threads) < 1:
        raise ValueError("epochs/eval_every/layers/batch_size/threads must be positive")
    if args.lr <= 0 or not 0 <= args.dropout < 1 or args.patience < 0:
        raise ValueError("Invalid learning rate, dropout or patience")
    if any(n == 0 or n < -1 for n in args.neighbors):
        raise ValueError("Neighbor counts must be positive or -1")
    fanouts = args.neighbors * args.layers if len(args.neighbors) == 1 else args.neighbors
    if args.mode == "sampled" and len(fanouts) != args.layers:
        raise ValueError("Use one fanout (repeated per layer) or one per GNN layer")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.json").exists():
        raise FileExistsError(f"Completed result exists: {args.output}; choose another --output")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    save_json(args.output / "config.json", config)
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    data, classes = load_data(args.dataset, args.data_root)
    model = NodeClassifier(args.model, data.num_node_features, args.hidden, classes,
                           args.layers, args.dropout, args.heads).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = None
    if args.mode == "sampled":
        loader = NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=fanouts,
                                batch_size=args.batch_size, shuffle=True, num_workers=0,
                                subgraph_type="directional")
    graph = data.clone().to(device)
    adjacency = model.adjacency(graph.edge_index, graph.num_nodes)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()
    history, best_state = [], None
    best_val, best_epoch, stale, updates = -1.0, 0, 0, 0
    train_total = val_total = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        sync(device)
        tick = time.perf_counter()
        total_loss, count = 0.0, 0
        if loader is None:
            optimiser.zero_grad(set_to_none=True)
            logits = model(graph.x, adjacency)
            loss = F.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask])
            loss.backward()
            optimiser.step()
            total_loss, count = float(loss.detach()), 1
            updates += 1
        else:
            for batch in loader:
                batch = batch.to(device)
                optimiser.zero_grad(set_to_none=True)
                logits = model(batch.x, model.adjacency(batch.edge_index, batch.num_nodes))
                # NeighborLoader places seed nodes first.
                n = batch.batch_size
                loss = F.cross_entropy(logits[:n], batch.y[:n])
                loss.backward()
                optimiser.step()
                total_loss += float(loss.detach()) * n
                count += n
                updates += 1
        sync(device)
        elapsed = time.perf_counter() - tick
        train_total += elapsed
        row = {"epoch": epoch, "train_loss": total_loss / count, "train_seconds": elapsed,
               "val_accuracy": "", "val_seconds": ""}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            sync(device)
            tick = time.perf_counter()
            with torch.no_grad():
                logits = model(graph.x, adjacency)
                val = metrics(logits[graph.val_mask], graph.y[graph.val_mask], classes)
            sync(device)
            duration = time.perf_counter() - tick
            val_total += duration
            row.update(val_accuracy=val["accuracy"], val_seconds=duration)
            if val["accuracy"] > best_val:
                best_val, best_epoch, stale = val["accuracy"], epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"{args.dataset}/{args.model}/{args.mode} seed={args.seed} epoch={epoch} "
                  f"loss={row['train_loss']:.4f} best_val={best_val:.4f}", flush=True)
        if args.patience and stale >= args.patience:
            break
    sync(device)
    fit_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    sync(device)
    tick = time.perf_counter()
    with torch.no_grad():
        logits = model(graph.x, adjacency)
        test = metrics(logits[graph.test_mask], graph.y[graph.test_mask], classes)
    sync(device)
    test_seconds = time.perf_counter() - tick
    result = {**config, "status": "ok", "device_used": str(device),
              "dataset_stats": describe(data, classes), "best_epoch": best_epoch,
              "best_val_accuracy": best_val, "test_accuracy": test["accuracy"],
              "test_macro_f1": test["macro_f1"], "epochs_run": len(history),
              "optimizer_steps": updates, "train_seconds": train_total,
              "mean_epoch_seconds": train_total / len(history), "val_seconds": val_total,
              "fit_seconds": fit_seconds, "test_seconds": test_seconds,
              "parameters": sum(p.numel() for p in model.parameters()),
              "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "pyg": torch_geometric.__version__, "platform": platform.platform(),
                              "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    torch.save({"state_dict": best_state, "config": config,
                "in_channels": data.num_node_features, "out_channels": classes}, args.output / "best.pt")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    save_json(args.output / "metrics.json", result)
    print(json.dumps({k: result[k] for k in ("test_accuracy", "test_macro_f1", "train_seconds")}), flush=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
