import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

from data import TASK_ROOT
from models import MODELS
from train import parser as train_parser, run


def parse_list(text, cast):
    return [cast(value) for value in text.split(",") if value]


def main():
    parser = argparse.ArgumentParser(description="Run KGE experiments")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--lrs", default="0.005")
    parser.add_argument("--dimensions", default="64")
    parser.add_argument("--layers", default="1")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negative-samples", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=TASK_ROOT / "results" / "baseline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--in-process", action="store_true")
    args = parser.parse_args()

    models = parse_list(args.models, str)
    lrs = parse_list(args.lrs, float)
    dimensions = parse_list(args.dimensions, int)
    layers = parse_list(args.layers, int)
    seeds = parse_list(args.seeds, int)
    unknown = set(models) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []

    for model, lr, dimension, depth, seed in itertools.product(
            models, lrs, dimensions, layers, seeds):
        name = f"{model}_lr{lr:g}_d{dimension}_l{depth}_s{seed}"
        output = args.output / name
        if args.resume and (output / "metrics.json").exists():
            print(f"skip {name}")
            continue
        command = [
            "--model", model, "--lr", str(lr), "--dimension", str(dimension),
            "--layers", str(depth), "--seed", str(seed),
            "--epochs", str(args.epochs), "--eval-every", str(args.eval_every),
            "--batch-size", str(args.batch_size),
            "--negative-samples", str(args.negative_samples),
            "--device", args.device, "--output", str(output),
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
