from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from torch.nn import functional as F
from torch_geometric.datasets import TUDataset, ZINC


TASK_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("MUTAG", "ZINC")


class ZINCClassification(Dataset):
    def __init__(self, dataset, bin_edges, num_node_types):
        self.dataset = dataset
        self.bin_edges = bin_edges
        self.num_node_types = num_node_types

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index].clone()
        data.x = F.one_hot(
            data.x.view(-1).long(), num_classes=self.num_node_types).float()
        data.y = torch.bucketize(data.y.view(-1).float(), self.bin_edges).long()
        return data


def stratified_split(labels, seed, train_ratio=0.8, val_ratio=0.1):
    generator = torch.Generator().manual_seed(seed)
    train, val, test = [], [], []
    for class_id in torch.unique(labels).tolist():
        indices = torch.where(labels == class_id)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        train_end = int(len(indices) * train_ratio)
        val_end = train_end + int(len(indices) * val_ratio)
        train.extend(indices[:train_end].tolist())
        val.extend(indices[train_end:val_end].tolist())
        test.extend(indices[val_end:].tolist())
    return train, val, test


def limit_training_set(dataset, count, seed):
    if not count or len(dataset) <= count:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices)


def load_splits(name, root=None, seed=42, zinc_bins=10, max_train_graphs=0):
    root = Path(root) if root else TASK_ROOT / "data"
    if name == "MUTAG":
        dataset = TUDataset(str(root / "MUTAG"), name="MUTAG")
        labels = torch.tensor([int(dataset[i].y) for i in range(len(dataset))])
        train_idx, val_idx, test_idx = stratified_split(labels, seed)
        train = limit_training_set(Subset(dataset, train_idx), max_train_graphs, seed)
        val = Subset(dataset, val_idx)
        test = Subset(dataset, test_idx)
        return train, val, test, dataset.num_node_features, dataset.num_classes

    if name == "ZINC":
        raw_train = ZINC(str(root / "ZINC"), subset=True, split="train")
        raw_val = ZINC(str(root / "ZINC"), subset=True, split="val")
        raw_test = ZINC(str(root / "ZINC"), subset=True, split="test")
        targets = raw_train._data.y.view(-1).float()
        quantiles = torch.linspace(0, 1, zinc_bins + 1)[1:-1]
        bin_edges = torch.unique(torch.quantile(targets, quantiles))
        num_node_types = 1 + max(
            int(dataset._data.x.max())
            for dataset in (raw_train, raw_val, raw_test)
        )
        train = ZINCClassification(raw_train, bin_edges, num_node_types)
        val = ZINCClassification(raw_val, bin_edges, num_node_types)
        test = ZINCClassification(raw_test, bin_edges, num_node_types)
        train = limit_training_set(train, max_train_graphs, seed)
        return train, val, test, num_node_types, len(bin_edges) + 1

    raise ValueError(f"Unknown dataset: {name}")


if __name__ == "__main__":
    import json

    result = {}
    for name in DATASETS:
        train, val, test, features, classes = load_splits(name)
        result[name] = {
            "train_graphs": len(train),
            "val_graphs": len(val),
            "test_graphs": len(test),
            "features": features,
            "classes": classes,
        }
        print(name, result[name])
    (TASK_ROOT / "data" / "statistics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
