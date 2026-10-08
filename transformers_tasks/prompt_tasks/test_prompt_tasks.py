"""Checks for leakage, mask placement, and trainable soft prompts."""

import unittest
from tempfile import TemporaryDirectory

import torch
from transformers import AutoTokenizer, BertConfig

from run_prompt_tasks import DEFAULT_MODEL, PromptClassifier, ReviewDataset, encode_labels, load_splits


class PromptTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.valid, cls.test, cls.labels = load_splits()
        cls.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, local_files_only=True)

    def test_splits_are_disjoint_and_reproducible(self):
        self.assertEqual((len(self.train), len(self.valid), len(self.test)), (63, 293, 297))
        sets = [{text for _, text in rows} for rows in (self.train, self.valid, self.test)]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertEqual((self.train, self.valid, self.test, self.labels), load_splits())
        self.assertEqual(len(self.labels), 10)

    def test_exactly_two_masks_and_truncation(self):
        for method in ("pet", "ptuning"):
            dataset = ReviewDataset([("水果", "苹果" * 500)], self.tokenizer, self.labels, method, 128, 15)
            ids, positions, _ = dataset[0]
            self.assertLessEqual(len(ids), 128)
            self.assertEqual(2, len(positions))
            self.assertEqual([self.tokenizer.mask_token_id] * 2, [ids[p] for p in positions])
            self.assertEqual(ids.count(self.tokenizer.mask_token_id), 2)

    def test_soft_prompt_receives_gradient(self):
        model = PromptClassifier(DEFAULT_MODEL, encode_labels(self.tokenizer, self.labels), "ptuning", 15)
        ids, positions, label = ReviewDataset([("水果", "苹果很好吃")], self.tokenizer, self.labels, "ptuning", 128, 15)[0]
        logits = model(torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long), torch.tensor([positions]))
        self.assertEqual(tuple(logits.shape), (1, 10))
        torch.nn.functional.cross_entropy(logits, torch.tensor([label])).backward()
        self.assertIsNotNone(model.soft_prompt.grad)
        self.assertGreater(float(model.soft_prompt.grad.abs().sum()), 0)

    def test_masked_lm_checkpoint_uses_vocab_logits(self):
        with TemporaryDirectory() as directory:
            config = BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)
            config.architectures = ["BertForMaskedLM"]
            config.save_pretrained(directory)
            model = PromptClassifier(directory, [[10, 11], [12, 13]], "pet", from_config=True)
            scores = model(torch.tensor([[2, 4, 4, 3]]), torch.ones(1, 4, dtype=torch.long), torch.tensor([[1, 2]]))
            self.assertEqual(model.scoring, "mlm")
            self.assertEqual(tuple(scores.shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
