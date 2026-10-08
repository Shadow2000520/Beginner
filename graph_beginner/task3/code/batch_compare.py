import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(root, pooling):
    rows = {}
    for path in root.glob("*/metrics.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["pooling"] == pooling:
            rows[(value["dataset"], value["model"])] = value
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare mini-batch and full-batch training")
    parser.add_argument("--mini", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pooling", default="mean")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    mini = load(args.mini, args.pooling)
    full = load(args.full, args.pooling)
    if mini.keys() != full.keys():
        raise ValueError("The two result sets do not contain the same experiments")

    rows = []
    for dataset, model in sorted(mini):
        for mode, source in (("mini", mini), ("full", full)):
            value = source[(dataset, model)]
            rows.append({
                "dataset": dataset,
                "model": model,
                "mode": mode,
                "batch_size": value["batch_size"],
                "test_accuracy": value["test_accuracy"],
                "test_macro_f1": value["test_macro_f1"],
                "train_seconds": value["train_seconds"],
            })
    fields = list(rows[0])
    with (args.output / "runs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with (args.output / "summary.md").open("w", encoding="utf-8") as file:
        file.write("| 数据集 | 模型 | 方式 | 测试准确率 | Macro-F1 | 训练时间(s) |\n")
        file.write("| --- | --- | --- | ---: | ---: | ---: |\n")
        for row in rows:
            file.write(
                f"| {row['dataset']} | {row['model'].upper()} | {row['mode']} | "
                f"{row['test_accuracy']:.4f} | {row['test_macro_f1']:.4f} | "
                f"{row['train_seconds']:.2f} |\n")

    datasets = sorted({row[0] for row in mini})
    models = [model for model in ("gcn", "gat", "sage", "gin")
              if any(key[1] == model for key in mini)]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 4 * len(datasets)),
                             squeeze=False)
    x = np.arange(len(models))
    for i, dataset in enumerate(datasets):
        for offset, (mode, source) in enumerate((("mini", mini), ("full", full))):
            accuracy = [source[(dataset, model)]["test_accuracy"] for model in models]
            seconds = [source[(dataset, model)]["train_seconds"] for model in models]
            positions = x + (-0.2 if offset == 0 else 0.2)
            axes[i, 0].bar(positions, accuracy, 0.4, label=mode)
            axes[i, 1].bar(positions, seconds, 0.4, label=mode)
        axes[i, 0].set_title(f"{dataset} test accuracy")
        axes[i, 1].set_title(f"{dataset} training time")
        axes[i, 0].set_ylim(0, 1)
        axes[i, 0].set_ylabel("accuracy")
        axes[i, 1].set_ylabel("seconds")
        for axis in axes[i]:
            axis.set_xticks(x, [model.upper() for model in models])
            axis.legend()
            axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=180)
    plt.close(fig)
    print(f"compared {len(mini)} experiment pairs")


if __name__ == "__main__":
    main()
