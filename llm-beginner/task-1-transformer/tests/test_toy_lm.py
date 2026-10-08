import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CharTokenizer
from src.toy_lm import ToyLanguageModel, causal_leakage
from train_toy_lm import make_examples, collate_lm


class ToyLMTests(unittest.TestCase):
    def test_shifted_targets_and_tail_padding(self):
        tokenizer = CharTokenizer.build(["甲乙丙丁戊己"], max_length=3)
        examples = make_examples("甲乙丙丁戊己", tokenizer, 3)
        x, y = collate_lm(examples)
        self.assertEqual(x[0].tolist(), tokenizer("甲乙丙").tolist())
        self.assertEqual(y[0].tolist(), tokenizer("乙丙丁").tolist())
        self.assertEqual(x[1, -1].item(), 0)
        self.assertEqual(y[1, -1].item(), -100)

    def test_model_causality_and_backward(self):
        torch.manual_seed(0)
        model = ToyLanguageModel(12, d_model=16, n_heads=2, n_layers=2, block_size=8, dropout=0)
        ids = torch.randint(2, 12, (2, 8))
        self.assertLess(causal_leakage(model, ids), 1e-6)
        logits = model(ids)
        self.assertEqual(logits.shape, (2, 8, 12))
        targets = torch.randint(2, 12, (2, 8))
        F.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_generation_sliding_window_and_training_mode(self):
        model = ToyLanguageModel(12, d_model=16, n_heads=2, n_layers=1, block_size=4)
        prompt = torch.tensor([[2, 3, 4]])
        model.eval()
        with torch.no_grad():
            logits = model(prompt)[:, -1].clone()
            logits[:, :2] = -torch.inf
            expected_first = logits.argmax(-1)
        model.train()
        result = model.generate(prompt, max_new_tokens=7, temperature=0)
        self.assertTrue(model.training)
        self.assertEqual(result.shape, (1, 10))
        torch.testing.assert_close(result[:, :3], prompt)
        torch.testing.assert_close(result[:, 3], expected_first)
        self.assertTrue((result[:, 3:] >= 2).all())


if __name__ == "__main__":
    unittest.main()
