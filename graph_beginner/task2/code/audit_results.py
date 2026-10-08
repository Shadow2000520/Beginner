import argparse
import csv
import itertools
import json
import math
from pathlib import Path


def audit(root, datasets, seeds, lrs, layers, epochs):
    errors = []
    expected = itertools.product(
        datasets, ["gcn", "gat", "sage", "gin"],
        ["full", "sampled"], seeds, lrs, layers)
    count = 0

    for dataset, model, mode, seed, lr, depth in expected:
        count += 1
        name = f"{dataset}_{model}_{mode}_s{seed}_lr{lr:g}_l{depth}"
        folder = root / name
        metrics_path = folder / "metrics.json"
        history_path = folder / "history.csv"
        if not metrics_path.exists() or not history_path.exists():
            errors.append(f"Missing: {name}")
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = list(csv.DictReader(history_path.open(encoding="utf-8")))
        if len(history) != epochs:
            errors.append(f"Epoch count: {name}")
        for key in ("test_auc", "test_average_precision", "test_accuracy"):
            if not 0 <= metrics[key] <= 1:
                errors.append(f"Invalid {key}: {name}")
        if not all(math.isfinite(float(row["train_loss"])) for row in history):
            errors.append(f"Invalid loss: {name}")
        if not (folder / "best.pt").exists():
            errors.append(f"Missing checkpoint: {name}")

    result = {"expected_runs": count, "passed": not errors, "errors": errors}
    (root / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer", "Flickr"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--lrs", nargs="+", type=float, default=[0.01])
    p.add_argument("--layers", nargs="+", type=int, default=[2])
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    audit(args.input, args.datasets, args.seeds, args.lrs, args.layers, args.epochs)
