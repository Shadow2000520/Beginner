import argparse
import contextlib
import gc
import itertools
import json
import subprocess
import sys
import traceback
from pathlib import Path

from data import DATASETS, TASK_ROOT


def main():
    p = argparse.ArgumentParser(description="Run link prediction experiments")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--models", nargs="+", choices=["gcn", "gat", "sage", "gin"],
                   default=["gcn", "gat", "sage", "gin"])
    p.add_argument("--modes", nargs="+", choices=["full", "sampled"],
                   default=["full", "sampled"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--lrs", nargs="+", type=float, default=[0.01])
    p.add_argument("--layers", nargs="+", type=int, default=[2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--fanout", type=int, default=15)
    p.add_argument("--max-train-edges", type=int, default=20000)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--output", type=Path, default=TASK_ROOT / "results" / "baseline")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--in-process", action="store_true")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    combinations = list(itertools.product(
        args.datasets, args.models, args.modes, args.seeds, args.lrs, args.layers))

    for index, (dataset, model, mode, seed, lr, layers) in enumerate(combinations, 1):
        name = f"{dataset}_{model}_{mode}_s{seed}_lr{lr:g}_l{layers}"
        output = args.output / name
        if (output / "metrics.json").exists():
            if args.resume:
                continue
            raise FileExistsError(f"Result exists: {output}")

        output.mkdir(parents=True, exist_ok=True)
        command_args = [
            "--dataset", dataset,
            "--model", model,
            "--mode", mode,
            "--seed", str(seed),
            "--lr", str(lr),
            "--layers", str(layers),
            "--epochs", str(args.epochs),
            "--hidden", str(args.hidden),
            "--neighbors", str(args.fanout),
            "--batch-size", str(args.batch_size),
            "--max-train-edges", str(args.max_train_edges),
            "--device", args.device,
            "--output", str(output),
        ]
        print(f"[{index}/{len(combinations)}] {name}", flush=True)

        with (output / "run.log").open("w", encoding="utf-8") as log:
            if args.in_process:
                import torch
                from train import parser as train_parser, run
                return_code = 0
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    try:
                        run(train_parser().parse_args(command_args))
                    except Exception:
                        traceback.print_exc()
                        return_code = 1
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                command = [sys.executable, str(Path(__file__).with_name("train.py")), *command_args]
                return_code = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT).returncode

        if return_code:
            failures.append(name)

    (args.output / "failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8")
    subprocess.run([
        sys.executable,
        str(Path(__file__).with_name("report.py")),
        "--input", str(args.output),
    ], check=True)
    if failures:
        raise SystemExit(f"{len(failures)} experiments failed")


if __name__ == "__main__":
    main()
