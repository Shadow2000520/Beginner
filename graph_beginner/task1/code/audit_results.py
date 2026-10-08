"""Check that an experiment matrix is complete and its saved evidence agrees."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path


def audit(root, datasets, seeds, lrs, layers, epochs):
    errors = []
    expected = list(itertools.product(datasets, ["gcn", "gat", "sage", "gin"],
                                      ["full", "sampled"], seeds, lrs, layers))
    for dataset, model, mode, seed, lr, depth in expected:
        name = f"{dataset}_{model}_{mode}_s{seed}_lr{lr:g}_l{depth}"
        folder = root / name
        if not (folder / "metrics.json").exists():
            errors.append(f"Missing completed run: {name}")
            continue
        result = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        history = list(csv.DictReader((folder / "history.csv").open(encoding="utf-8")))
        for key in ("test_accuracy", "test_macro_f1", "best_val_accuracy"):
            if not 0 <= result[key] <= 1:
                errors.append(f"Invalid {key}: {name}")
        if len(history) != epochs or result["epochs_run"] != epochs:
            errors.append(f"Incomplete epoch budget: {name}")
        if not all(math.isfinite(float(r["train_loss"])) for r in history):
            errors.append(f"Non-finite training loss: {name}")
        vals = [(float(r["val_accuracy"]), int(r["epoch"])) for r in history if r["val_accuracy"]]
        if vals:
            best = max(v for v, e in vals)
            first_best = next(e for v, e in vals if v == best)
            if best != result["best_val_accuracy"] or first_best != result["best_epoch"]:
                errors.append(f"Validation selection mismatch: {name}")
        if not math.isclose(sum(float(r["train_seconds"]) for r in history), result["train_seconds"], rel_tol=1e-8):
            errors.append(f"Timing mismatch: {name}")
        steps = epochs if mode == "full" else epochs * math.ceil(result["dataset_stats"]["train"] / result["batch_size"])
        if steps != result["optimizer_steps"]:
            errors.append(f"Optimizer step mismatch: {name}")
        if not (folder / "best.pt").exists():
            errors.append(f"Missing local checkpoint: {name}")
    output = {"expected_runs": len(expected), "passed": not errors, "errors": errors}
    (root / "audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer", "Flickr"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--lrs", type=float, nargs="+", default=[0.01])
    p.add_argument("--layers", type=int, nargs="+", default=[2])
    p.add_argument("--epochs", type=int, default=100)
    a = p.parse_args()
    audit(a.input, a.datasets, a.seeds, a.lrs, a.layers, a.epochs)
