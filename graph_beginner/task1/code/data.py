"""Dataset loading, official fixed splits and split integrity checks."""
from pathlib import Path
import torch
from torch_geometric.datasets import Flickr, Planetoid
from torch_geometric.transforms import NormalizeFeatures

TASK_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Cora", "CiteSeer", "Flickr")


def load_data(name, root=None):
    root = Path(root) if root else TASK_ROOT / "data"
    if name in ("Cora", "CiteSeer"):
        dataset = Planetoid(str(root / name), name, split="public",
                           transform=NormalizeFeatures())
    elif name == "Flickr":
        dataset = Flickr(str(root / name))
    else:
        raise ValueError(f"Unknown dataset: {name}")
    data = dataset[0]
    validate_splits(data)
    return data, dataset.num_classes


def validate_splits(data):
    masks = [data.train_mask, data.val_mask, data.test_mask]
    for mask in masks:
        if mask.dtype != torch.bool or mask.shape != (data.num_nodes,) or not mask.any():
            raise ValueError("Every split must be a non-empty boolean node mask")
    if (sum(m.int() for m in masks) > 1).any():
        raise ValueError("Train/validation/test splits overlap")


def describe(data, classes):
    return {"nodes": data.num_nodes, "directed_edges": data.num_edges,
            "features": data.num_node_features, "classes": classes,
            **{s: int(data[f"{s}_mask"].sum()) for s in ("train", "val", "test")}}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Download and inspect official datasets")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    summary = {}
    for name in args.datasets:
        data, classes = load_data(name)
        summary[name] = describe(data, classes)
        summary[name]["class_counts"] = {
            split: data.y[data[f"{split}_mask"]].bincount(minlength=classes).tolist()
            for split in ("train", "val", "test")}
        print(name, summary[name], flush=True)
    (TASK_ROOT / "data" / "statistics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
