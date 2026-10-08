"""Evaluate a saved validation-selected checkpoint without retraining."""
import argparse
import json
from pathlib import Path
import torch
from data import load_data
from models import NodeClassifier
from train import get_device, metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--data-root", type=Path)
    args = p.parse_args()
    device = get_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    c = saved["config"]
    data, classes = load_data(c["dataset"], args.data_root)
    model = NodeClassifier(c["model"], data.num_node_features, c["hidden"], classes,
                           c["layers"], c["dropout"], c["heads"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data.x, model.adjacency(data.edge_index, data.num_nodes))
    print(json.dumps({s: metrics(logits[data[f"{s}_mask"]], data.y[data[f"{s}_mask"]], classes)
                      for s in ("val", "test")}, indent=2))
