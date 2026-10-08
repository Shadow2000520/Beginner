import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import MiniGPT, load_for_eval
from src.rope import RotaryEmbedding
from src.tokenizer import BPETokenizer
from eval.test_kv_cache_equivalence import test_kv_cache_equivalence


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = MiniGPT(259, d_model=32, n_heads=4, n_layers=2, block_size=16)
        self.ids = torch.randint(0, 259, (2, 12))

    def test_shape_and_backward(self):
        logits = self.model(self.ids)
        self.assertEqual(logits.shape, (2, 12, 259))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 259), self.ids[:, 1:].reshape(-1))
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_rope_rotation_and_offset(self):
        rope = RotaryEmbedding(4)
        x = torch.tensor([[[[1., 2., 3., 4.]]]])
        rotated = rope(x, offset=1)
        angles = torch.tensor([1., .01])
        expected = torch.stack((x[..., ::2] * angles.cos() - x[..., 1::2] * angles.sin(),
                                x[..., ::2] * angles.sin() + x[..., 1::2] * angles.cos()), -1).flatten(-2)
        torch.testing.assert_close(rotated, expected)
        torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1))
        torch.testing.assert_close(rope(x), x)

    def test_causality(self):
        self.model.eval()
        changed = self.ids.clone()
        changed[:, 6:] = (changed[:, 6:] + 1) % 259
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.ids)[:, :6], self.model(changed)[:, :6],
                                       rtol=0, atol=1e-7)

    def test_cache_equivalence(self):
        result = test_kv_cache_equivalence()
        self.assertTrue(result["pass"], result)

    def test_cache_shape_and_no_mutation(self):
        self.model.eval()
        with torch.no_grad():
            _, cache = self.model(self.ids[:, :5], return_cache=True)
            snapshot = [(k.clone(), v.clone()) for k, v in cache]
            out, updated = self.model(self.ids[:, 5:8], cache, True)
            torch.testing.assert_close(out, self.model(self.ids[:, 5:8], cache))
        for (k, v), (old_k, old_v), (new_k, new_v) in zip(cache, snapshot, updated):
            torch.testing.assert_close(k, old_k)
            torch.testing.assert_close(v, old_v)
            self.assertEqual(new_k.shape, (2, 4, 8, 8))
            self.assertEqual(new_v.shape, new_k.shape)

    def test_context_and_invalid_cache(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 17), dtype=torch.long))
        with self.assertRaises(ValueError):
            MiniGPT(259, d_model=12, n_heads=4)
        with self.assertRaises(ValueError):
            self.model(self.ids, [])
        with torch.no_grad():
            _, cache = self.model(self.ids, return_cache=True)
        with self.assertRaises(ValueError):
            self.model(self.ids, cache)
        with self.assertRaises(ValueError):
            self.model(self.ids[:, :1], [cache[0], None])

    def test_generation_cache_and_window(self):
        cached = self.model.generate(self.ids, 9)
        uncached = self.model.generate(self.ids, 9, use_cache=False)
        self.assertTrue(torch.equal(cached, uncached))
        self.assertTrue(self.model.training)
        self.assertEqual(cached.shape, (2, 21))
        self.assertTrue(torch.equal(cached[:, :12], self.ids))

    def test_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            BPETokenizer().save_pretrained(path / "tokenizer.json")
            torch.save({"config": self.model.config, "model_state_dict": self.model.state_dict()},
                       path / "best.pt")
            loaded, tokenizer = load_for_eval(path / "best.pt")
            self.model.eval()
            with torch.no_grad():
                torch.testing.assert_close(loaded(self.ids), self.model(self.ids))
            self.assertEqual(tokenizer.vocab_size, 259)

    def test_tied_weights_reload_and_cache(self):
        model = MiniGPT(259, d_model=32, n_heads=4, n_layers=2,
                        block_size=16, tie_weights=True).eval()
        self.assertIs(model.embedding.weight, model.lm_head.weight)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            BPETokenizer().save_pretrained(path / "tokenizer.json")
            torch.save({"config": model.config, "model_state_dict": model.state_dict()}, path / "best.pt")
            loaded, _ = load_for_eval(path / "best.pt")
            self.assertIs(loaded.embedding.weight, loaded.lm_head.weight)
            with torch.no_grad():
                full = loaded(self.ids)
                torch.testing.assert_close(full, model(self.ids))
                cache, outputs = None, []
                for i in range(self.ids.size(1)):
                    logits, cache = loaded(self.ids[:, i:i + 1], cache, True)
                    outputs.append(logits)
                torch.testing.assert_close(full, torch.cat(outputs, dim=1), atol=1e-4, rtol=0)


if __name__ == "__main__":
    unittest.main()
