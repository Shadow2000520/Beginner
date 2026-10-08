import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch


TASK_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://raw.githubusercontent.com/pykeen/pykeen/master/src/pykeen/datasets/nations"
FILES = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}


@dataclass
class KnowledgeGraph:
    train: torch.Tensor
    valid: torch.Tensor
    test: torch.Tensor
    entity_to_id: dict
    relation_to_id: dict

    @property
    def num_entities(self):
        return len(self.entity_to_id)

    @property
    def num_relations(self):
        return len(self.relation_to_id)

    @property
    def all_triples(self):
        return torch.cat((self.train, self.valid, self.test), dim=0)


def download_nations(root=None):
    root = Path(root) if root else TASK_ROOT / "data" / "nations"
    root.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        path = root / filename
        if not path.exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{filename}", path)
    return root


def read_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            head, relation, tail = line.split("\t")
            rows.append((head, relation, tail))
    return rows


def load_knowledge_graph(root=None):
    root = download_nations(root)
    raw = {split: read_rows(root / filename) for split, filename in FILES.items()}
    entities = sorted({entity for rows in raw.values() for h, _, t in rows
                       for entity in (h, t)})
    relations = sorted({relation for rows in raw.values() for _, relation, _ in rows})
    entity_to_id = {name: index for index, name in enumerate(entities)}
    relation_to_id = {name: index for index, name in enumerate(relations)}

    def encode(rows):
        return torch.tensor([
            (entity_to_id[h], relation_to_id[r], entity_to_id[t])
            for h, r, t in rows
        ], dtype=torch.long)

    return KnowledgeGraph(
        encode(raw["train"]), encode(raw["valid"]), encode(raw["test"]),
        entity_to_id, relation_to_id)


def triples_mask(triples, entities, relations):
    mask = torch.zeros(
        entities, relations, entities, dtype=torch.bool)
    mask[triples[:, 0], triples[:, 1], triples[:, 2]] = True
    return mask


def true_mask(graph):
    return triples_mask(graph.all_triples, graph.num_entities, graph.num_relations)


if __name__ == "__main__":
    graph = load_knowledge_graph()
    statistics = {
        "dataset": "Nations",
        "entities": graph.num_entities,
        "relations": graph.num_relations,
        "train_triples": len(graph.train),
        "valid_triples": len(graph.valid),
        "test_triples": len(graph.test),
    }
    path = TASK_ROOT / "data" / "statistics.json"
    path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    print(statistics)
