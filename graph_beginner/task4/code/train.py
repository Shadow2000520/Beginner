import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from data import TASK_ROOT, load_knowledge_graph, triples_mask, true_mask
from models import MODELS, build_model


def parser():
    p = argparse.ArgumentParser(description="Knowledge graph completion")
    p.add_argument("--model", choices=MODELS, default="transe")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dimension", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=12.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--negative-samples", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--data-root", type=Path, default=TASK_ROOT / "data" / "nations")
    p.add_argument("--output", type=Path, default=TASK_ROOT / "results" / "single")
    return p


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device(name)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sample_negative_tails(positive, count, entities, known):
    relation = positive[:, 1]
    tail_allowed = ~known[positive[:, 0], relation, :]
    tail_possible = tail_allowed.any(dim=1)
    tail_weights = tail_allowed.float()
    tail_weights[~tail_possible] = 1
    return torch.multinomial(tail_weights, count, replacement=True)


def sample_negatives(positive, count, entities, known):
    negative = positive[:, None, :].expand(-1, count, -1).clone()
    negative[:, :, 2] = sample_negative_tails(
        positive, count, entities, known)
    return negative.flatten(0, 1)


def rank_batch(model, triples, entities, relations, known, mode):
    batch_size = len(triples)
    if mode == "head":
        pairs = torch.stack((triples[:, 2], triples[:, 1] + relations), dim=1)
        filtered = known[:, triples[:, 1], triples[:, 2]].T
        target = triples[:, 0]
    else:
        pairs = triples[:, :2]
        filtered = known[triples[:, 0], triples[:, 1], :]
        target = triples[:, 2]
    scores = model.score_all_tails(pairs)
    filtered[torch.arange(batch_size, device=triples.device), target] = False
    scores = scores.masked_fill(filtered, -torch.inf)
    target_score = scores[torch.arange(batch_size, device=triples.device), target]
    return 1 + (scores > target_score[:, None]).sum(dim=1)


def evaluate(model, triples, entities, relations, known, batch_size, device):
    model.eval()
    ranks = []
    with torch.no_grad():
        for start in range(0, len(triples), batch_size):
            batch = triples[start:start + batch_size].to(device)
            ranks.append(rank_batch(model, batch, entities, relations, known, "head"))
            ranks.append(rank_batch(model, batch, entities, relations, known, "tail"))
    ranks = torch.cat(ranks).float()
    return {
        "mrr": float((1.0 / ranks).mean()),
        "mr": float(ranks.mean()),
        "hits_at_1": float((ranks <= 1).float().mean()),
        "hits_at_3": float((ranks <= 3).float().mean()),
        "hits_at_10": float((ranks <= 10).float().mean()),
    }


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run(args):
    positive = (args.epochs, args.eval_every, args.dimension, args.layers,
                args.batch_size, args.negative_samples, args.threads)
    if min(positive) < 1:
        raise ValueError("training parameters must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.json").exists():
        raise FileExistsError(f"Result exists: {args.output}")

    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    save_json(args.output / "config.json", config)
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    graph = load_knowledge_graph(args.data_root)
    known = true_mask(graph).to(device)
    inverse = graph.train[:, [2, 1, 0]].clone()
    inverse[:, 1] += graph.num_relations
    augmented_train = torch.cat((graph.train, inverse), dim=0)
    train_known = triples_mask(
        augmented_train, graph.num_entities, graph.num_relations * 2).to(device)
    train_triples = augmented_train.to(device)

    model = build_model(
        args.model, graph.num_entities, graph.num_relations * 2, args.dimension,
        args.gamma, args.layers, args.channels, args.dropout).to(device)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    history = []
    best_mrr = -1.0
    best_epoch = 0
    best_state = None
    train_seconds = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_triples), generator=generator, device=device)
        total_loss = 0.0
        total_triples = 0
        sync(device)
        start = time.perf_counter()
        for offset in range(0, len(order), args.batch_size):
            positive_batch = train_triples[order[offset:offset + args.batch_size]]
            negative_tails = sample_negative_tails(
                positive_batch, args.negative_samples, graph.num_entities, train_known)
            optimiser.zero_grad(set_to_none=True)
            scores = model.score_all_tails(positive_batch[:, :2])
            positive_score = scores.gather(1, positive_batch[:, 2, None]).squeeze(1)
            negative_score = scores.gather(1, negative_tails)
            loss = F.softplus(-positive_score).mean() + F.softplus(negative_score).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            model.post_step()
            total_loss += float(loss.detach()) * len(positive_batch)
            total_triples += len(positive_batch)
        sync(device)
        duration = time.perf_counter() - start
        train_seconds += duration

        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_triples,
            "train_seconds": duration,
            "val_mrr": "",
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            validation = evaluate(
                model, graph.valid, graph.num_entities, graph.num_relations, known,
                args.eval_batch_size, device)
            row["val_mrr"] = validation["mrr"]
            if validation["mrr"] > best_mrr:
                best_mrr = validation["mrr"]
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        history.append(row)
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"{args.model} epoch={epoch} loss={row['train_loss']:.4f} "
                  f"val_mrr={best_mrr:.4f}", flush=True)

    model.load_state_dict(best_state)
    validation = evaluate(
        model, graph.valid, graph.num_entities, graph.num_relations, known,
        args.eval_batch_size, device)
    test = evaluate(
        model, graph.test, graph.num_entities, graph.num_relations, known,
        args.eval_batch_size, device)
    result = {
        **config,
        "status": "ok",
        "dataset": "Nations",
        "device_used": str(device),
        "entities": graph.num_entities,
        "relations": graph.num_relations * 2,
        "train_triples": len(graph.train),
        "valid_triples": len(graph.valid),
        "test_triples": len(graph.test),
        "best_epoch": best_epoch,
        "best_val_mrr": best_mrr,
        **{f"val_{key}": value for key, value in validation.items()},
        **{f"test_{key}": value for key, value in test.items()},
        "train_seconds": train_seconds,
        "mean_epoch_seconds": train_seconds / len(history),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    torch.save({
        "state_dict": best_state,
        "config": config,
        "entities": graph.num_entities,
        "relations": graph.num_relations * 2,
    }, args.output / "best.pt")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    save_json(args.output / "metrics.json", result)
    print(json.dumps({
        "test_mrr": result["test_mrr"],
        "test_hits_at_10": result["test_hits_at_10"],
        "train_seconds": result["train_seconds"],
    }), flush=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
