import unittest

import torch
from torch import nn

from data import load_knowledge_graph, true_mask
from models import build_model
from train import evaluate, sample_negatives


class Oracle(nn.Module):
    def __init__(self, known):
        super().__init__()
        self.known = known

    def score_all_tails(self, pairs):
        return self.known[pairs[:, 0], pairs[:, 1], :].float()


class Task4Test(unittest.TestCase):
    def test_models(self):
        triples = torch.tensor([
            [0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 2, 4],
        ])
        for name in ("transe", "rotate", "conve"):
            model = build_model(name, 6, 3, dimension=32, conv_layers=2)
            score = model.score(triples)
            self.assertEqual(tuple(score.shape), (4,))
            score.sum().backward()

    def test_negative_sampling(self):
        positive = torch.tensor([[0, 0, 1], [1, 0, 2]])
        known = torch.zeros(4, 1, 4, dtype=torch.bool)
        known[positive[:, 0], positive[:, 1], positive[:, 2]] = True
        negative = sample_negatives(positive, 20, 4, known)
        self.assertEqual(tuple(negative.shape), (40, 3))
        self.assertFalse(known[negative[:, 0], negative[:, 1], negative[:, 2]].any())

    def test_filtered_ranking(self):
        triples = torch.tensor([[0, 0, 1], [1, 0, 2]])
        known = torch.zeros(4, 1, 4, dtype=torch.bool)
        known[triples[:, 0], triples[:, 1], triples[:, 2]] = True
        inverse = triples[:, [2, 1, 0]].clone()
        inverse[:, 1] += 1
        reciprocal = torch.zeros(4, 2, 4, dtype=torch.bool)
        combined = torch.cat((triples, inverse))
        reciprocal[combined[:, 0], combined[:, 1], combined[:, 2]] = True
        metrics = evaluate(
            Oracle(reciprocal), triples, 4, 1, known, 2, torch.device("cpu"))
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(metrics["hits_at_1"], 1.0)

    def test_dataset_splits(self):
        graph = load_knowledge_graph()
        train = {tuple(row) for row in graph.train.tolist()}
        valid = {tuple(row) for row in graph.valid.tolist()}
        test = {tuple(row) for row in graph.test.tolist()}
        self.assertFalse(train & valid or train & test or valid & test)
        self.assertEqual((graph.num_entities, graph.num_relations), (14, 55))


if __name__ == "__main__":
    unittest.main()
