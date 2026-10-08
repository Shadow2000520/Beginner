import argparse
import json
from pathlib import Path


def values(text, cast):
    return [cast(value) for value in text.split(",") if value]


def main():
    parser = argparse.ArgumentParser(description="Check a KGE experiment matrix")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--models", default="transe,rotate,conve")
    parser.add_argument("--lrs", default="0.005")
    parser.add_argument("--dimensions", default="64")
    parser.add_argument("--layers", default="1")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    expected = {
        (model, lr, dimension, layers, seed)
        for model in values(args.models, str)
        for lr in values(args.lrs, float)
        for dimension in values(args.dimensions, int)
        for layers in values(args.layers, int)
        for seed in values(args.seeds, int)
    }
    found = set()
    errors = []
    for path in args.input.glob("*/metrics.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (row["model"], float(row["lr"]), int(row["dimension"]),
               int(row["layers"]), int(row["seed"]))
        if key in found:
            errors.append(f"duplicate: {key}")
        found.add(key)
        if int(row["epochs"]) != args.epochs:
            errors.append(f"wrong epochs: {path.parent.name}")
        for metric in ("test_mrr", "test_hits_at_1", "test_hits_at_3", "test_hits_at_10"):
            if not 0 <= float(row[metric]) <= 1:
                errors.append(f"invalid {metric}: {path.parent.name}")
        if not (path.parent / "history.csv").exists():
            errors.append(f"missing history: {path.parent.name}")
        if not (path.parent / "best.pt").exists():
            errors.append(f"missing checkpoint: {path.parent.name}")
    errors.extend(f"missing: {key}" for key in sorted(expected - found))
    errors.extend(f"unexpected: {key}" for key in sorted(found - expected))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"audit passed: {len(found)} runs")


if __name__ == "__main__":
    main()
