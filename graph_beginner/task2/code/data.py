from pathlib import Path
import random

import numpy as np
import torch
from torch_geometric.datasets import Flickr, Planetoid
from torch_geometric.transforms import NormalizeFeatures, RandomLinkSplit
from torch_geometric.utils import remove_self_loops, to_undirected


TASK_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Cora", "CiteSeer", "Flickr")


def load_graph(name, root=None):
    root = Path(root) if root else TASK_ROOT / "data"
    if name in ("Cora", "CiteSeer"):
        dataset = Planetoid(str(root / name), name, transform=NormalizeFeatures())
    elif name == "Flickr":
        dataset = Flickr(str(root / name))
    else:
        raise ValueError(f"Unknown dataset: {name}")

    data = dataset[0]
    edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
    return data


def split_graph(data, seed, val_ratio=0.05, test_ratio=0.10,
                disjoint_train_ratio=0.30):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        disjoint_train_ratio=disjoint_train_ratio,
    )
    return transform(data)


def statistics(data):
    return {
        "nodes": data.num_nodes,
        "undirected_edges": data.num_edges // 2,
        "features": data.num_node_features,
    }


if __name__ == "__main__":
    import json

    result = {}
    for dataset_name in DATASETS:
        graph = load_graph(dataset_name)
        train_data, val_data, test_data = split_graph(graph, seed=42)
        result[dataset_name] = {
            **statistics(graph),
            "train_message_edges": train_data.num_edges // 2,
            "train_positive_edges": int((train_data.edge_label == 1).sum()),
            "val_positive_edges": int((val_data.edge_label == 1).sum()),
            "test_positive_edges": int((test_data.edge_label == 1).sum()),
        }
        print(dataset_name, result[dataset_name])

    (TASK_ROOT / "data" / "statistics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
