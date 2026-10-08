import argparse
import json
from pathlib import Path

import torch

from data import load_graph, split_graph
from models import GNNEncoder
from train import evaluate, get_device


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a saved link prediction model")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--data-root", type=Path)
    args = p.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    graph = load_graph(config["dataset"], args.data_root)
    _, val_data, test_data = split_graph(graph, config["seed"])
    device = get_device(args.device)
    model = GNNEncoder(
        config["model"],
        graph.num_node_features,
        config["hidden"],
        config["layers"],
        config["dropout"],
        config["heads"],
    ).to(device)
    model.load_state_dict(saved["state_dict"])
    print(json.dumps({
        "validation": evaluate(model, val_data, device),
        "test": evaluate(model, test_data, device),
    }, indent=2))
