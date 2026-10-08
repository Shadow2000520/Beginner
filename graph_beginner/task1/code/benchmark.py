"""Sequential isolated experiment matrix; supports resume and parameter sweeps."""
import argparse
import contextlib
import gc
import itertools
import json
import subprocess
import sys
import traceback
from pathlib import Path
from data import TASK_ROOT, DATASETS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--models", nargs="+", choices=["gcn", "gat", "sage", "gin"], default=["gcn", "gat", "sage", "gin"])
    p.add_argument("--modes", nargs="+", choices=["full", "sampled"], default=["full", "sampled"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--lrs", nargs="+", type=float, default=[0.01])
    p.add_argument("--layers", nargs="+", type=int, default=[2])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--fanout", type=int, default=15)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--output", type=Path, default=TASK_ROOT / "results" / "baseline")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--in-process", action="store_true",
                   help="Reuse Python/CUDA runtime for parameter sweeps; timing is not comparable to isolated runs")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    combos = list(itertools.product(args.datasets, args.models, args.modes, args.seeds, args.lrs, args.layers))
    for index, (dataset, model, mode, seed, lr, layers) in enumerate(combos, 1):
        name = f"{dataset}_{model}_{mode}_s{seed}_lr{lr:g}_l{layers}"
        output = args.output / name
        if (output / "metrics.json").exists():
            if not args.resume:
                raise FileExistsError(f"Completed run exists: {output}; use --resume or another --output")
            previous = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            expected = {"epochs": args.epochs, "batch_size": args.batch_size, "hidden": args.hidden,
                        "neighbors": [args.fanout], "device": args.device}
            if any(previous.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Resume configuration mismatch: {output}")
            execution_path = output / "execution.json"
            isolated = json.loads(execution_path.read_text(encoding="utf-8"))["isolated_process"] if execution_path.exists() else True
            if isolated == args.in_process:
                raise ValueError(f"Cannot mix isolated and in-process timings: {output}")
            continue
        output.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).with_name("train.py")),
               "--dataset", dataset, "--model", model, "--mode", mode, "--seed", str(seed),
               "--lr", str(lr), "--layers", str(layers), "--epochs", str(args.epochs),
               "--hidden", str(args.hidden), "--neighbors", str(args.fanout),
               "--batch-size", str(args.batch_size), "--device", args.device, "--output", str(output)]
        print(f"[{index}/{len(combos)}] {name}", flush=True)
        with (output / "run.log").open("w", encoding="utf-8") as log:
            if args.in_process:
                import torch
                from train import parser as train_parser, run
                returncode = 0
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    try:
                        run(train_parser().parse_args(cmd[2:]))
                    except Exception:
                        traceback.print_exc()
                        returncode = 1
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                returncode = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        (output / "execution.json").write_text(json.dumps({"isolated_process": not args.in_process}), encoding="utf-8")
        if returncode:
            failures.append({"run": name, "returncode": returncode})
            print(f"FAILED: see {output / 'run.log'}", flush=True)
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(Path(__file__).with_name("report.py")),
                    "--input", str(args.output)], check=True)
    if failures:
        raise SystemExit(f"{len(failures)} experiments failed; failures.json and run.log contain details")


if __name__ == "__main__":
    main()
