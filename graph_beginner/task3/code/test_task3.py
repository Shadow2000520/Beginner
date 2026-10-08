import unittest

import torch
from torch_geometric.data import Batch, Data

from data import ZINCClassification, stratified_split
from models import GraphClassifier, global_min_pool


class Task3Test(unittest.TestCase):
    def test_pooling(self):
        x = torch.tensor([[1.0, 4.0], [3.0, 2.0], [-1.0, 5.0]])
        batch = torch.tensor([0, 0, 1])
        expected = torch.tensor([[1.0, 2.0], [-1.0, 5.0]])
        self.assertTrue(torch.equal(global_min_pool(x, batch), expected))

    def test_all_models_and_poolings(self):
        graphs = [
            Data(x=torch.randn(3, 5), edge_index=torch.tensor(
                [[0, 1, 1, 2], [1, 0, 2, 1]]), y=torch.tensor([0])),
            Data(x=torch.randn(2, 5), edge_index=torch.tensor(
                [[0, 1], [1, 0]]), y=torch.tensor([1])),
        ]
        batch = Batch.from_data_list(graphs)
        for model_name in ("gcn", "gat", "sage", "gin"):
            for pooling in ("mean", "max", "min"):
                model = GraphClassifier(model_name, pooling, 5, 8, 2, 2, 0.0, 2)
                output = model(batch.x, batch.edge_index, batch.batch)
                self.assertEqual(tuple(output.shape), (2, 2))
                output.sum().backward()

    def test_stratified_split(self):
        labels = torch.tensor([0] * 20 + [1] * 20)
        train, val, test = stratified_split(labels, seed=42)
        self.assertEqual(len(set(train + val + test)), len(labels))
        self.assertEqual((len(train), len(val), len(test)), (32, 4, 4))
        self.assertEqual({int(labels[i]) for i in val}, {0, 1})

    def test_zinc_conversion(self):
        raw = [Data(x=torch.tensor([[0], [2]]), y=torch.tensor([1.5]))]
        dataset = ZINCClassification(raw, torch.tensor([1.0, 2.0]), 3)
        graph = dataset[0]
        self.assertEqual(tuple(graph.x.shape), (2, 3))
        self.assertEqual(int(graph.y), 1)


if __name__ == "__main__":
    unittest.main()
