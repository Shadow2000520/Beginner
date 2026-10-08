import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIELDS = [
    "model", "lr", "dimension", "layers", "seed", "epochs", "test_mrr",
    "test_mr", "test_hits_at_1", "test_hits_at_3", "test_hits_at_10",
    "best_val_mrr", "train_seconds", "parameters", "best_epoch",
]


def load_results(root):
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append({field: value[field] for field in FIELDS})
    if not rows:
        raise FileNotFoundError(f"No metrics.json found in {root}")
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    keys = ("model", "lr", "dimension", "layers")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summary = []
    for group, values in sorted(groups.items()):
        summary.append({
            **dict(zip(keys, group)),
            "runs": len(values),
            "mrr": float(np.mean([row["test_mrr"] for row in values])),
            "hits1": float(np.mean([row["test_hits_at_1"] for row in values])),
            "hits3": float(np.mean([row["test_hits_at_3"] for row in values])),
            "hits10": float(np.mean([row["test_hits_at_10"] for row in values])),
            "seconds": float(np.mean([row["train_seconds"] for row in values])),
        })
    return summary


def write_files(output, rows, summary):
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "summary.md").open("w", encoding="utf-8") as file:
        file.write("| 模型 | 学习率 | 维度 | ConvE层数 | MRR | Hits@1 | Hits@3 | Hits@10 | 时间(s) |\n")
        file.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summary:
            file.write(
                f"| {row['model']} | {float(row['lr']):g} | {row['dimension']} | "
                f"{row['layers']} | {row['mrr']:.4f} | {row['hits1']:.4f} | "
                f"{row['hits3']:.4f} | {row['hits10']:.4f} | {row['seconds']:.2f} |\n")


def comparison_plot(summary, output):
    best = {}
    for row in summary:
        if row["model"] not in best or row["mrr"] > best[row["model"]]["mrr"]:
            best[row["model"]] = row
    order = [model for model in ("transe", "rotate", "conve") if model in best]
    labels = [model.upper() if model != "conve" else "ConvE" for model in order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(order))
    width = 0.25
    for offset, (field, label) in enumerate((
            ("mrr", "MRR"), ("hits3", "Hits@3"), ("hits10", "Hits@10"))):
        axes[0].bar(x + (offset - 1) * width,
                    [best[model][field] for model in order], width, label=label)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Filtered link prediction")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [best[model]["seconds"] for model in order])
    axes[1].set_title("Training time")
    axes[1].set_ylabel("seconds")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


def sensitivity_plot(summary, output):
    chosen = [row for row in summary if row["model"] == "conve"]
    lrs = sorted({float(row["lr"]) for row in chosen})
    layers = sorted({int(row["layers"]) for row in chosen})
    if len(lrs) == 1 and len(layers) == 1:
        return
    matrix = np.full((len(lrs), len(layers)), np.nan)
    for i, lr in enumerate(lrs):
        for j, depth in enumerate(layers):
            values = [row["mrr"] for row in chosen
                      if float(row["lr"]) == lr and int(row["layers"]) == depth]
            if values:
                matrix[i, j] = np.mean(values)
    fig, axis = plt.subplots(figsize=(6, 4.5), layout="constrained")
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(layers)), layers)
    axis.set_yticks(range(len(lrs)), [f"{lr:g}" for lr in lrs])
    axis.set_xlabel("convolution layers")
    axis.set_ylabel("learning rate")
    axis.set_title("ConvE test MRR")
    for i in range(len(lrs)):
        for j in range(len(layers)):
            if not np.isnan(matrix[i, j]):
                axis.text(j, i, f"{matrix[i, j]:.3f}",
                          ha="center", va="center", color="white")
    fig.colorbar(image, ax=axis, label="MRR")
    fig.savefig(output / "sensitivity.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarise KGE experiments")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_results(args.input)
    summary = aggregate(rows)
    write_files(args.output, rows, summary)
    comparison_plot(summary, args.output)
    sensitivity_plot(summary, args.output)
    print(f"summarised {len(rows)} runs")


if __name__ == "__main__":
    main()
