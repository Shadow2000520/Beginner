import unittest

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from data import split_graph
from models import GNNEncoder, edge_logits
from train import binary_metrics, sample_negative_edges


class LinkPredictionTests(unittest.TestCase):
    def setUp(self):
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 0, 3],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5, 3, 0],
        ])
        self.data = Data(x=torch.randn(6, 8), edge_index=to_undirected(edge_index))

    def test_metrics(self):
        result = binary_metrics(
            torch.tensor([4.0, 2.0, -2.0, -4.0]),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(result["auc"], 1.0)
        self.assertAlmostEqual(result["average_precision"], 1.0)
        self.assertAlmostEqual(result["accuracy"], 1.0)

    def test_split_has_no_training_leakage(self):
        train_data, val_data, test_data = split_graph(
            self.data, seed=42, val_ratio=0.1, test_ratio=0.1,
            disjoint_train_ratio=0.5)
        train_again, val_again, test_again = split_graph(
            self.data, seed=42, val_ratio=0.1, test_ratio=0.1,
            disjoint_train_ratio=0.5)
        message_edges = {
            tuple(sorted(edge)) for edge in train_data.edge_index.t().tolist()
        }
        label_edges = {
            tuple(sorted(edge)) for edge in train_data.edge_label_index.t().tolist()
        }
        self.assertTrue(message_edges.isdisjoint(label_edges))
        torch.testing.assert_close(train_data.edge_label_index, train_again.edge_label_index)
        torch.testing.assert_close(val_data.edge_label_index, val_again.edge_label_index)
        torch.testing.assert_close(test_data.edge_label_index, test_again.edge_label_index)

    def test_models_and_negative_sampling(self):
        forbidden = to_undirected(self.data.edge_index)
        negatives = sample_negative_edges(forbidden, self.data.num_nodes, 2)
        real_edges = {tuple(edge) for edge in forbidden.t().tolist()}
        self.assertTrue(all(tuple(edge) not in real_edges for edge in negatives.t().tolist()))

        for name in ("gcn", "gat", "sage", "gin"):
            with self.subTest(model=name):
                model = GNNEncoder(name, 8, 8, num_layers=2, dropout=0)
                z = model(self.data.x, model.adjacency(self.data.edge_index, self.data.num_nodes))
                logits = edge_logits(z, self.data.edge_index[:, :4])
                self.assertEqual(z.shape, (6, 8))
                self.assertEqual(logits.shape, (4,))
                logits.sum().backward()


if __name__ == "__main__":
    unittest.main()
