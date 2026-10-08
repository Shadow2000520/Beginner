import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import MiniGPT
from src.sampling import sample_next_token, sampling_probs


class SamplingTests(unittest.TestCase):
    def test_temperature_and_greedy(self):
        logits = torch.tensor([[0., 1., 2.], [3., 0., 1.]])
        original = logits.clone()
        torch.testing.assert_close(sampling_probs(logits, temperature=2), (logits / 2).softmax(-1))
        self.assertEqual(sample_next_token(logits, temperature=0).tolist(), [[2], [0]])
        self.assertEqual(sample_next_token(logits, temperature=1e-300).tolist(), [[2], [0]])
        torch.testing.assert_close(logits, original)

    def test_top_k_exact_count_and_large_k(self):
        logits = torch.tensor([[1., 3., 2., 0.], [0., 0., 0., 0.]])
        probs = sampling_probs(logits, top_k=2)
        self.assertEqual((probs > 0).sum(-1).tolist(), [2, 2])
        self.assertEqual(probs[0, [0, 3]].tolist(), [0., 0.])
        torch.testing.assert_close(probs.sum(-1), torch.ones(2))
        torch.testing.assert_close(sampling_probs(logits, top_k=99), logits.softmax(-1))

    def test_top_p_crossing_and_scatter(self):
        logits = torch.tensor([[.1, .6, .3], [.3, .1, .6]]).log()
        probs = sampling_probs(logits, top_p=.7)
        expected = torch.tensor([[0., 2/3, 1/3], [1/3, 0., 2/3]])
        torch.testing.assert_close(probs, expected)
        self.assertEqual((sampling_probs(logits, top_p=.01) > 0).sum(-1).tolist(), [1, 1])
        equal_logits = torch.zeros(1, 4)
        self.assertEqual((sampling_probs(equal_logits, top_p=.5) > 0).sum().item(), 2)

    def test_seed_and_combined_filters(self):
        logits = torch.tensor([[0., 3., 2., 1.]]).repeat(32, 1)
        a = sample_next_token(logits, top_k=2, top_p=.95, generator=torch.Generator().manual_seed(42))
        b = sample_next_token(logits, top_k=2, top_p=.95, generator=torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(((a == 1) | (a == 2)).all())

    def test_invalid_inputs(self):
        logits = torch.zeros(1, 4)
        for kwargs in [dict(top_k=-1), dict(top_k=1.5), dict(top_p=0),
                       dict(top_p=1.1), dict(temperature=-1), dict(temperature=float('nan'))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sampling_probs(logits, **kwargs)
        for values in [[float('-inf')]*4, [float('nan'), 0., 0., 0.], [float('inf'), 0., 0., 0.]]:
            with self.assertRaises(ValueError):
                sampling_probs(torch.tensor([values]))
        self.assertEqual(sampling_probs(torch.tensor([[0., float('-inf')]])).tolist(), [[1., 0.]])

    def test_generation_cache_window_and_specials(self):
        torch.manual_seed(5)
        model = MiniGPT(20, d_model=16, n_heads=2, n_layers=2, block_size=8)
        ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
        for kwargs in [dict(temperature=0), dict(temperature=.8),
                       dict(temperature=.8, top_k=5), dict(temperature=.8, top_p=.8)]:
            outputs = [model.generate(ids, 12, **kwargs, use_cache=cache, forbidden_ids=[0, 1],
                                      generator=torch.Generator().manual_seed(42)) for cache in (True, False)]
            self.assertTrue(torch.equal(*outputs))
            self.assertTrue(model.training)
            self.assertEqual(outputs[0].shape, (2, 15))
            self.assertFalse(((outputs[0][:, 3:] == 0) | (outputs[0][:, 3:] == 1)).any())
        self.assertTrue(torch.equal(model.generate(ids, 0), ids))


if __name__ == "__main__":
    unittest.main()
