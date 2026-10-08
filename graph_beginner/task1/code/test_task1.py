"""Small offline tests for graph aggregation, sampling and split integrity."""
import unittest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from data import validate_splits
from models import NodeClassifier
from train import metrics, seed_everything


class TaskOneTests(unittest.TestCase):
    def setUp(self):
        seed_everything(42)
        self.data = Data(x=torch.randn(6, 8), y=torch.tensor([0, 1, 0, 1, 0, 1]),
                         edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
                                                  [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]]),
                         train_mask=torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool),
                         val_mask=torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool),
                         test_mask=torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool))

    def test_splits(self):
        validate_splits(self.data)
        self.data.test_mask[0] = True
        with self.assertRaises(ValueError):
            validate_splits(self.data)

    def test_metrics(self):
        result = metrics(torch.tensor([[3., 0.], [3., 0.], [0., 3.], [0., 3.]]),
                         torch.tensor([0, 1, 1, 1]), 2)
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 4 / 5) / 2, places=6)

    def test_models_and_sampling(self):
        data = self.data
        for name in ("gcn", "gat", "sage", "gin"):
            for depth in (1, 2, 3):
                with self.subTest(model=name, depth=depth):
                    model = NodeClassifier(name, 8, 8, 2, depth, dropout=0)
                    model.eval()
                    full = model(data.x, model.adjacency(data.edge_index, 6))
                    self.assertEqual(full.shape, (6, 2))
                    batch = next(iter(NeighborLoader(data, num_neighbors=[-1] * depth,
                                                     batch_size=6, shuffle=False)))
                    out = model(batch.x, model.adjacency(batch.edge_index, batch.num_nodes))
                    torch.testing.assert_close(out, full[batch.n_id], atol=1e-5, rtol=1e-5)
                    sampled = next(iter(NeighborLoader(data, input_nodes=data.train_mask,
                                                       num_neighbors=[2] * depth, batch_size=2)))
                    self.assertTrue(data.train_mask[sampled.n_id[:sampled.batch_size]].all())
                    logits = model(sampled.x, model.adjacency(sampled.edge_index, sampled.num_nodes))
                    loss = torch.nn.functional.cross_entropy(logits[:sampled.batch_size], sampled.y[:sampled.batch_size])
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
