import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from data import load_splits
from models import GraphClassifier
from train import evaluate, get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    _, val_set, test_set, features, classes = load_splits(
        config["dataset"], config["data_root"], config["seed"],
        config["zinc_bins"], config["max_train_graphs"])
    model = GraphClassifier(
        config["model"], config["pooling"], features, config["hidden"],
        classes, config["layers"], config["dropout"], config["heads"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    batch_size = config["batch_size"]
    result = {
        "validation": evaluate(
            model, DataLoader(val_set, batch_size=batch_size), device, classes),
        "test": evaluate(
            model, DataLoader(test_set, batch_size=batch_size), device, classes),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
