import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


GROUP = ["dataset", "model", "mode", "lr", "layers"]
VALUES = ["test_auc", "test_average_precision", "test_accuracy",
          "best_val_auc", "train_seconds", "mean_epoch_seconds", "optimizer_steps"]


def aggregate(root):
    runs = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(root.glob("*/metrics.json"))]
    if not runs:
        raise RuntimeError(f"No results found in {root}")

    fields = GROUP + ["seed", "best_epoch", "train_positive_edges", "parameters"] + VALUES
    with (root / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)

    grouped = defaultdict(list)
    for row in runs:
        grouped[tuple(row[key] for key in GROUP)].append(row)

    summary = []
    for key, group in sorted(grouped.items()):
        row = dict(zip(GROUP, key), n=len(group))
        for column in VALUES:
            values = [item[column] for item in group]
            row[column + "_mean"] = statistics.mean(values)
            row[column + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(row)

    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# 实测结果", "",
        "| 数据集 | 模型 | 训练 | LR | 层数 | AUC (%) | AP (%) | Accuracy (%) | 时间 (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['mode']} | {row['lr']:g} | "
            f"{row['layers']} | {100 * row['test_auc_mean']:.2f} | "
            f"{100 * row['test_average_precision_mean']:.2f} | "
            f"{100 * row['test_accuracy_mean']:.2f} | {row['train_seconds_mean']:.2f} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    os.environ.setdefault("MPLCONFIGDIR", str(root.resolve() / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = sorted(set((row["dataset"], row["lr"], row["layers"]) for row in summary))
    models = ["gcn", "gat", "sage", "gin"]
    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 3.1 * len(panels)), squeeze=False)
    for ax_row, (dataset, lr, layers) in zip(axes, panels):
        for j, (column, label, factor) in enumerate([
                ("test_auc", "Test AUC (%)", 100),
                ("train_seconds", "Training time (s)", 1)]):
            ax = ax_row[j]
            for mode, offset, color in [
                    ("full", -0.19, "#2563eb"), ("sampled", 0.19, "#f59e0b")]:
                selected = {
                    row["model"]: row for row in summary
                    if (row["dataset"], row["lr"], row["layers"], row["mode"])
                    == (dataset, lr, layers, mode)
                }
                indexes = [i for i, model in enumerate(models) if model in selected]
                ax.bar(
                    [i + offset for i in indexes],
                    [selected[models[i]][column + "_mean"] * factor for i in indexes],
                    width=0.36, label=mode, color=color)
            ax.set_xticks(range(4), ["GCN", "GAT", "GraphSAGE", "GIN"])
            ax.set_ylabel(label)
            ax.set_title(f"{dataset} | lr={lr:g} | layers={layers}")
            ax.legend()
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)

    if len({row["lr"] for row in summary}) > 1:
        import numpy as np
        lrs = sorted({row["lr"] for row in summary})
        layers_list = sorted({row["layers"] for row in summary})
        for dataset in sorted({row["dataset"] for row in summary}):
            fig, axes = plt.subplots(4, 2, figsize=(9, 13), layout="constrained")
            for i, model in enumerate(models):
                for j, mode in enumerate(["full", "sampled"]):
                    values = np.full((len(lrs), len(layers_list)), np.nan)
                    for row in summary:
                        if (row["dataset"], row["model"], row["mode"]) == (dataset, model, mode):
                            values[lrs.index(row["lr"]), layers_list.index(row["layers"])] = (
                                100 * row["best_val_auc_mean"])
                    ax = axes[i, j]
                    image = ax.imshow(values, vmin=50, vmax=100, cmap="Blues", aspect="auto")
                    for y in range(len(lrs)):
                        for x in range(len(layers_list)):
                            if np.isfinite(values[y, x]):
                                ax.text(x, y, f"{values[y, x]:.1f}", ha="center", va="center")
                    ax.set_xticks(range(len(layers_list)), layers_list)
                    ax.set_yticks(range(len(lrs)), [f"{value:g}" for value in lrs])
                    ax.set_xlabel("Layers")
                    ax.set_ylabel("Learning rate")
                    ax.set_title(f"{model.upper()} / {mode}")
            fig.suptitle(f"{dataset}: validation AUC (%)")
            fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.65)
            fig.savefig(root / f"{dataset}_sensitivity.png", dpi=150)
            plt.close(fig)

    print(f"Aggregated {len(runs)} runs")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    aggregate(p.parse_args().input)
