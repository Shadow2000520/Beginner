"""Aggregate measured runs into CSV, Markdown and a comparison figure."""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


GROUP = ["dataset", "model", "mode", "lr", "layers"]
VALUES = ["test_accuracy", "test_macro_f1", "best_val_accuracy", "train_seconds",
          "mean_epoch_seconds", "fit_seconds", "test_seconds", "optimizer_steps"]


def aggregate(root):
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/metrics.json"))]
    if not runs:
        raise RuntimeError(f"No completed experiments in {root}")
    fields = GROUP + ["seed", "epochs_run", "best_epoch", "device_used", "parameters", "peak_cuda_allocated_mb"] + VALUES
    with (root / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    grouped = defaultdict(list)
    for row in runs:
        grouped[tuple(row[k] for k in GROUP)].append(row)
    rows = []
    for key, group in sorted(grouped.items()):
        row = dict(zip(GROUP, key), n=len(group))
        for col in VALUES:
            vals = [r[col] for r in group]
            row[col + "_mean"] = statistics.mean(vals)
            row[col + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rows.append(row)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 实测结果", "", "± 表示不同随机种子的标准差。", "",
             "| 数据集 | 模型 | 训练 | LR | 层数 | n | 测试 Accuracy (%) | Macro-F1 (%) | 训练秒数 | 每 epoch 秒数 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['mode']} | {r['lr']:g} | {r['layers']} | {r['n']} | "
                     f"{100*r['test_accuracy_mean']:.2f} ± {100*r['test_accuracy_std']:.2f} | "
                     f"{100*r['test_macro_f1_mean']:.2f} ± {100*r['test_macro_f1_std']:.2f} | "
                     f"{r['train_seconds_mean']:.3f} | {r['mean_epoch_seconds_mean']:.4f} |")
    lines += ["", "训练时间包括采样、数据传输、前向、反向和参数更新。"]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR", str(root.resolve() / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = sorted(set((r["dataset"], r["lr"], r["layers"]) for r in rows))
    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 3.1 * len(panels)), squeeze=False)
    models = ["gcn", "gat", "sage", "gin"]
    for axrow, (dataset, lr, layers) in zip(axes, panels):
        for j, (col, label, factor) in enumerate([
                ("test_accuracy", "Test accuracy (%)", 100), ("train_seconds", "Training time (s)", 1)]):
            ax = axrow[j]
            for mode, offset, color in [("full", -0.19, "#2563eb"), ("sampled", 0.19, "#f59e0b")]:
                selected = {r["model"]: r for r in rows if (r["dataset"], r["lr"], r["layers"], r["mode"]) == (dataset, lr, layers, mode)}
                indices = [i for i, m in enumerate(models) if m in selected]
                ax.bar([i + offset for i in indices], [selected[models[i]][col + "_mean"] * factor for i in indices],
                       yerr=[selected[models[i]][col + "_std"] * factor for i in indices],
                       width=0.36, label=mode, color=color, capsize=3)
            ax.set_xticks(range(4), ["GCN", "GAT", "GraphSAGE", "GIN"])
            ax.set_ylabel(label)
            ax.set_title(f"{dataset} | lr={lr:g} | layers={layers}")
            ax.legend()
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
    if len({r["lr"] for r in rows}) > 1 and len({r["layers"] for r in rows}) > 1:
        import numpy as np
        lrs = sorted({r["lr"] for r in rows})
        depths = sorted({r["layers"] for r in rows})
        for dataset in sorted({r["dataset"] for r in rows}):
            fig, axes = plt.subplots(4, 2, figsize=(9, 13), layout="constrained")
            for i, model in enumerate(models):
                for j, mode in enumerate(["full", "sampled"]):
                    ax = axes[i, j]
                    values = np.full((len(lrs), len(depths)), np.nan)
                    for r in rows:
                        if (r["dataset"], r["model"], r["mode"]) == (dataset, model, mode):
                            values[lrs.index(r["lr"]), depths.index(r["layers"])] = 100 * r["best_val_accuracy_mean"]
                    im = ax.imshow(values, vmin=0, vmax=100, cmap="Blues", aspect="auto")
                    for y in range(len(lrs)):
                        for x in range(len(depths)):
                            if np.isfinite(values[y, x]):
                                ax.text(x, y, f"{values[y, x]:.1f}", ha="center", va="center",
                                        color="white" if values[y, x] > 55 else "black")
                    ax.set_xticks(range(len(depths)), depths)
                    ax.set_yticks(range(len(lrs)), [f"{v:g}" for v in lrs])
                    ax.set_xlabel("Message passing layers")
                    ax.set_ylabel("Learning rate")
                    ax.set_title(f"{model.upper()} / {mode}")
            fig.suptitle(f"{dataset}: validation accuracy (%)")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.65)
            fig.savefig(root / f"{dataset}_sensitivity.png", dpi=150)
            plt.close(fig)
    print(f"Aggregated {len(runs)} runs into {root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    aggregate(p.parse_args().input)
