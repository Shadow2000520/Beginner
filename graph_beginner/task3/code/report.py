import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIELDS = [
    "dataset", "model", "pooling", "lr", "layers", "seed", "epochs",
    "test_accuracy", "test_macro_f1", "best_val_accuracy", "train_seconds",
    "mean_epoch_seconds", "parameters", "best_epoch",
]


def load_results(root):
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append({field: value[field] for field in FIELDS})
    if not rows:
        raise FileNotFoundError(f"No metrics.json found in {root}")
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = defaultdict(list)
    keys = ("dataset", "model", "pooling", "lr", "layers")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group, values in sorted(groups.items()):
        result.append({
            **dict(zip(keys, group)),
            "runs": len(values),
            "accuracy_mean": float(np.mean([x["test_accuracy"] for x in values])),
            "accuracy_std": float(np.std([x["test_accuracy"] for x in values])),
            "f1_mean": float(np.mean([x["test_macro_f1"] for x in values])),
            "train_seconds_mean": float(np.mean([x["train_seconds"] for x in values])),
        })
    return result


def comparison_plot(rows, output):
    models = [name for name in ("gcn", "gat", "sage", "gin")
              if any(row["model"] == name for row in rows)]
    poolings = [name for name in ("mean", "max", "min")
                if any(row["pooling"] == name for row in rows)]
    datasets = sorted({row["dataset"] for row in rows})
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 4 * len(datasets)),
                             squeeze=False)
    width = 0.8 / len(poolings)
    for row_index, dataset in enumerate(datasets):
        selected = [row for row in rows if row["dataset"] == dataset]
        x = np.arange(len(models))
        for offset, pooling in enumerate(poolings):
            values = []
            times = []
            for model in models:
                group = [r for r in selected
                         if r["model"] == model and r["pooling"] == pooling]
                values.append(np.mean([r["test_accuracy"] for r in group]))
                times.append(np.mean([r["train_seconds"] for r in group]))
            position = x + (offset - (len(poolings) - 1) / 2) * width
            axes[row_index, 0].bar(position, values, width, label=pooling)
            axes[row_index, 1].bar(position, times, width, label=pooling)
        axes[row_index, 0].set_title(f"{dataset} test accuracy")
        axes[row_index, 1].set_title(f"{dataset} training time")
        axes[row_index, 0].set_ylim(0, 1)
        axes[row_index, 0].set_ylabel("accuracy")
        axes[row_index, 1].set_ylabel("seconds")
        for axis in axes[row_index]:
            axis.set_xticks(x, [name.upper() for name in models])
            axis.legend()
            axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def sensitivity_plots(rows, output):
    for dataset in sorted({row["dataset"] for row in rows}):
        chosen = [row for row in rows if row["dataset"] == dataset]
        lrs = sorted({float(row["lr"]) for row in chosen})
        depths = sorted({int(row["layers"]) for row in chosen})
        if len(lrs) == 1 and len(depths) == 1:
            continue
        models = sorted({row["model"] for row in chosen})
        poolings = sorted({row["pooling"] for row in chosen})
        if len(poolings) == 1:
            columns = min(2, len(models))
            rows_count = math.ceil(len(models) / columns)
            panels = [(i // columns, i % columns, model, poolings[0])
                      for i, model in enumerate(models)]
        else:
            columns = len(poolings)
            rows_count = len(models)
            panels = [(i, j, model, pooling)
                      for i, model in enumerate(models)
                      for j, pooling in enumerate(poolings)]
        fig, axes = plt.subplots(rows_count, columns,
                                 figsize=(4.5 * columns, 3.7 * rows_count),
                                 squeeze=False, layout="constrained")
        for i, j, model, pooling in panels:
            matrix = np.full((len(lrs), len(depths)), np.nan)
            for y, lr in enumerate(lrs):
                for x, depth in enumerate(depths):
                    values = [row["test_accuracy"] for row in chosen
                              if row["model"] == model
                              and row["pooling"] == pooling
                              and float(row["lr"]) == lr
                              and int(row["layers"]) == depth]
                    if values:
                        matrix[y, x] = np.mean(values)
            image = axes[i, j].imshow(matrix, vmin=0, vmax=1, cmap="viridis")
            axes[i, j].set_title(f"{model.upper()} / {pooling}")
            axes[i, j].set_xticks(range(len(depths)), depths)
            axes[i, j].set_yticks(range(len(lrs)), [f"{lr:g}" for lr in lrs])
            axes[i, j].set_xlabel("layers")
            axes[i, j].set_ylabel("learning rate")
            for y in range(len(lrs)):
                for x in range(len(depths)):
                    if not np.isnan(matrix[y, x]):
                        axes[i, j].text(x, y, f"{matrix[y, x]:.3f}",
                                        ha="center", va="center", color="white")
        for index in range(len(panels), rows_count * columns):
            axes[index // columns, index % columns].axis("off")
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="test accuracy")
        fig.savefig(output / f"{dataset.lower()}_sensitivity.png", dpi=180)
        plt.close(fig)


def write_summary(path, rows):
    with path.open("w", encoding="utf-8") as file:
        file.write("| 数据集 | 模型 | 池化 | 学习率 | 层数 | 测试准确率 | Macro-F1 | 训练时间(s) |\n")
        file.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            file.write(
                f"| {row['dataset']} | {row['model'].upper()} | {row['pooling']} | "
                f"{float(row['lr']):g} | {row['layers']} | "
                f"{row['accuracy_mean']:.4f} | {row['f1_mean']:.4f} | "
                f"{row['train_seconds_mean']:.2f} |\n")


def main():
    parser = argparse.ArgumentParser(description="Summarise experiment results")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_results(args.input)
    write_csv(args.output / "runs.csv", rows)
    summary = aggregate(rows)
    write_summary(args.output / "summary.md", summary)
    comparison_plot(rows, args.output / "comparison.png")
    sensitivity_plots(rows, args.output)
    print(f"summarised {len(rows)} runs")


if __name__ == "__main__":
    main()
