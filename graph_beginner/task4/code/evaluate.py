import argparse
import json
from pathlib import Path

import torch

from data import load_knowledge_graph, true_mask
from models import build_model
from train import evaluate, get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved KGE model")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    graph = load_knowledge_graph(config["data_root"])
    known = true_mask(graph).to(device)
    model = build_model(
        config["model"], graph.num_entities, graph.num_relations * 2,
        config["dimension"], config["gamma"], config["layers"],
        config["channels"], config["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    result = {
        "validation": evaluate(
            model, graph.valid, graph.num_entities, graph.num_relations, known,
            config["eval_batch_size"], device),
        "test": evaluate(
            model, graph.test, graph.num_entities, graph.num_relations, known,
            config["eval_batch_size"], device),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
