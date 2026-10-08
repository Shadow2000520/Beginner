import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

from data import DATASETS, TASK_ROOT
from train import parser as train_parser, run


MODELS = ("gcn", "gat", "sage", "gin")
POOLINGS = ("mean", "max", "min")


def parse_list(values, cast):
    return [cast(value) for value in values.split(",") if value]


def main():
    parser = argparse.ArgumentParser(description="Run graph-classification experiments")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--poolings", default=",".join(POOLINGS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--lrs", default="0.01")
    parser.add_argument("--layers", default="2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train-graphs", type=int, default=3000)
    parser.add_argument("--zinc-bins", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=TASK_ROOT / "results" / "baseline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--in-process", action="store_true")
    args = parser.parse_args()

    datasets = parse_list(args.datasets, str)
    models = parse_list(args.models, str)
    poolings = parse_list(args.poolings, str)
    seeds = parse_list(args.seeds, int)
    lrs = parse_list(args.lrs, float)
    layers = parse_list(args.layers, int)
    unknown = set(datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []

    combinations = itertools.product(datasets, models, poolings, seeds, lrs, layers)
    for dataset, model, pooling, seed, lr, depth in combinations:
        name = f"{dataset}_{model}_{pooling}_s{seed}_lr{lr:g}_l{depth}"
        output = args.output / name
        if args.resume and (output / "metrics.json").exists():
            print(f"skip {name}")
            continue
        command = [
            "--dataset", dataset, "--model", model, "--pooling", pooling,
            "--seed", str(seed), "--lr", str(lr), "--layers", str(depth),
            "--epochs", str(args.epochs), "--eval-every", str(args.eval_every),
            "--hidden", str(args.hidden), "--batch-size", str(args.batch_size),
            "--max-train-graphs", str(args.max_train_graphs),
            "--zinc-bins", str(args.zinc_bins), "--device", args.device,
            "--output", str(output),
        ]
        try:
            if args.in_process:
                run(train_parser().parse_args(command))
            else:
                subprocess.run(
                    [sys.executable, str(Path(__file__).with_name("train.py")), *command],
                    check=True)
        except Exception as error:
            failures.append({"run": name, "error": str(error)})
            print(f"failed {name}: {error}", file=sys.stderr)

    if failures:
        (args.output / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8")
        raise SystemExit(1)

    subprocess.run([
        sys.executable, str(Path(__file__).with_name("report.py")),
        "--input", str(args.output), "--output", str(args.output),
    ], check=True)


if __name__ == "__main__":
    main()
