import argparse
import json
from pathlib import Path


def values(text, cast):
    return [cast(item) for item in text.split(",") if item]


def main():
    parser = argparse.ArgumentParser(description="Check an experiment matrix")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--datasets", default="MUTAG,ZINC")
    parser.add_argument("--models", default="gcn,gat,sage,gin")
    parser.add_argument("--poolings", default="mean,max,min")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--lrs", default="0.01")
    parser.add_argument("--layers", default="2")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    expected = {
        (dataset, model, pooling, seed, lr, layers)
        for dataset in values(args.datasets, str)
        for model in values(args.models, str)
        for pooling in values(args.poolings, str)
        for seed in values(args.seeds, int)
        for lr in values(args.lrs, float)
        for layers in values(args.layers, int)
    }
    found = set()
    errors = []
    for path in args.input.glob("*/metrics.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (row["dataset"], row["model"], row["pooling"], int(row["seed"]),
               float(row["lr"]), int(row["layers"]))
        if key in found:
            errors.append(f"duplicate: {key}")
        found.add(key)
        if int(row["epochs"]) != args.epochs:
            errors.append(f"wrong epochs: {path.parent.name}")
        for metric in ("test_accuracy", "test_macro_f1", "best_val_accuracy"):
            if not 0 <= float(row[metric]) <= 1:
                errors.append(f"invalid {metric}: {path.parent.name}")
        if not (path.parent / "history.csv").exists():
            errors.append(f"missing history: {path.parent.name}")
        if not (path.parent / "best.pt").exists():
            errors.append(f"missing checkpoint: {path.parent.name}")

    missing = expected - found
    extra = found - expected
    errors.extend(f"missing: {key}" for key in sorted(missing))
    errors.extend(f"unexpected: {key}" for key in sorted(extra))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"audit passed: {len(found)} runs")


if __name__ == "__main__":
    main()
