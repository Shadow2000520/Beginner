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
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import negative_sampling, to_undirected

from data import DATASETS, TASK_ROOT, load_graph, split_graph, statistics
from models import GNNEncoder, edge_logits


def parser():
    p = argparse.ArgumentParser(description="GNN link prediction")
    p.add_argument("--dataset", choices=DATASETS, default="Cora")
    p.add_argument("--model", choices=["gcn", "gat", "sage", "gin"], default="gcn")
    p.add_argument("--mode", choices=["full", "sampled"], default="full")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--neighbors", type=int, nargs="+", default=[15, 10])
    p.add_argument("--max-train-edges", type=int, default=0)
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


def binary_metrics(logits, labels):
    labels = labels.float()
    probs = logits.sigmoid()
    accuracy = ((probs >= 0.5) == labels.bool()).float().mean()

    order = torch.argsort(probs, descending=True)
    sorted_labels = labels[order]
    positives = sorted_labels.sum().clamp_min(1)
    negatives = (1 - sorted_labels).sum().clamp_min(1)
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1 - sorted_labels, dim=0)
    tpr = torch.cat([torch.zeros(1, device=logits.device), tp / positives])
    fpr = torch.cat([torch.zeros(1, device=logits.device), fp / negatives])
    auc = torch.trapz(tpr, fpr)
    precision = tp / torch.arange(1, len(tp) + 1, device=logits.device)
    average_precision = (precision * sorted_labels).sum() / positives
    return {
        "accuracy": float(accuracy),
        "auc": float(auc),
        "average_precision": float(average_precision),
    }


def sample_negative_edges(forbidden, num_nodes, count):
    candidates = negative_sampling(
        forbidden,
        num_nodes=num_nodes,
        num_neg_samples=max(count * 3, 100),
        method="sparse",
    )
    candidates = candidates[:, candidates[0] < candidates[1]]
    candidates = torch.unique(candidates, dim=1)
    if candidates.size(1) < count:
        raise RuntimeError("Could not sample enough negative edges")
    return candidates[:, :count]


def evaluate(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        z = model(data.x, model.adjacency(data.edge_index, data.num_nodes))
        logits = edge_logits(z, data.edge_label_index)
    return binary_metrics(logits, data.edge_label)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run(args):
    if min(args.epochs, args.eval_every, args.layers, args.batch_size, args.threads) < 1:
        raise ValueError("epochs, layers and batch size must be positive")
    fanouts = args.neighbors * args.layers if len(args.neighbors) == 1 else args.neighbors
    if args.mode == "sampled" and len(fanouts) != args.layers:
        raise ValueError("Provide one fanout or one value per layer")

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.json").exists():
        raise FileExistsError(f"Result exists: {args.output}")

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    save_json(args.output / "config.json", config)
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = get_device(args.device)

    graph = load_graph(args.dataset, args.data_root)
    train_data, val_data, test_data = split_graph(graph, args.seed)
    train_pos = train_data.edge_label_index[:, train_data.edge_label == 1]
    if args.max_train_edges and train_pos.size(1) > args.max_train_edges:
        train_pos = train_pos[:, torch.randperm(train_pos.size(1))[:args.max_train_edges]]
    forbidden = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)

    model = GNNEncoder(
        args.model,
        graph.num_node_features,
        args.hidden,
        args.layers,
        args.dropout,
        args.heads,
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    full_graph = train_data.to(device) if args.mode == "full" else None
    train_pos_device = train_pos.to(device) if args.mode == "full" else None
    history = []
    best_auc = -1.0
    best_epoch = 0
    best_state = None
    train_seconds = 0.0
    updates = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        sync(device)
        start = time.perf_counter()
        neg_edges = sample_negative_edges(forbidden, graph.num_nodes, train_pos.size(1))

        if args.mode == "full":
            optimiser.zero_grad(set_to_none=True)
            z = model(full_graph.x, model.adjacency(full_graph.edge_index, full_graph.num_nodes))
            edges = torch.cat([train_pos_device, neg_edges.to(device)], dim=1)
            labels = torch.cat([
                torch.ones(train_pos.size(1), device=device),
                torch.zeros(train_pos.size(1), device=device),
            ])
            loss = F.binary_cross_entropy_with_logits(edge_logits(z, edges), labels)
            loss.backward()
            optimiser.step()
            loss_value = float(loss.detach())
            updates += 1
        else:
            edge_label_index = torch.cat([train_pos, neg_edges], dim=1)
            edge_label = torch.cat([
                torch.ones(train_pos.size(1)),
                torch.zeros(train_pos.size(1)),
            ])
            loader = LinkNeighborLoader(
                train_data,
                num_neighbors=fanouts,
                edge_label_index=edge_label_index,
                edge_label=edge_label,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                subgraph_type="directional",
            )
            total_loss = 0.0
            total_edges = 0
            for batch in loader:
                batch = batch.to(device)
                optimiser.zero_grad(set_to_none=True)
                z = model(batch.x, model.adjacency(batch.edge_index, batch.num_nodes))
                loss = F.binary_cross_entropy_with_logits(
                    edge_logits(z, batch.edge_label_index), batch.edge_label.float())
                loss.backward()
                optimiser.step()
                total_loss += float(loss.detach()) * batch.edge_label.numel()
                total_edges += batch.edge_label.numel()
                updates += 1
            loss_value = total_loss / total_edges

        sync(device)
        duration = time.perf_counter() - start
        train_seconds += duration
        row = {"epoch": epoch, "train_loss": loss_value,
               "train_seconds": duration, "val_auc": ""}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(model, val_data, device)
            row["val_auc"] = val["auc"]
            if val["auc"] > best_auc:
                best_auc = val["auc"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(row)
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"{args.dataset}/{args.model}/{args.mode} epoch={epoch} "
                  f"loss={loss_value:.4f} val_auc={best_auc:.4f}", flush=True)

    model.load_state_dict(best_state)
    val = evaluate(model, val_data, device)
    test = evaluate(model, test_data, device)
    result = {
        **config,
        "status": "ok",
        "device_used": str(device),
        "graph_stats": statistics(graph),
        "train_positive_edges": train_pos.size(1),
        "val_edges": val_data.edge_label.numel(),
        "test_edges": test_data.edge_label.numel(),
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "val_auc": val["auc"],
        "test_auc": test["auc"],
        "test_average_precision": test["average_precision"],
        "test_accuracy": test["accuracy"],
        "train_seconds": train_seconds,
        "mean_epoch_seconds": train_seconds / len(history),
        "optimizer_steps": updates,
        "parameters": sum(p.numel() for p in model.parameters()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pyg": torch_geometric.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }

    torch.save({"state_dict": best_state, "config": config,
                "in_channels": graph.num_node_features}, args.output / "best.pt")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    save_json(args.output / "metrics.json", result)
    print(json.dumps({k: result[k] for k in (
        "test_auc", "test_average_precision", "test_accuracy", "train_seconds")}), flush=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
