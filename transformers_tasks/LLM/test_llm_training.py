"""Data isolation and training-objective checks for the local LLM pipeline."""

import unittest

import torch
from transformers import AutoTokenizer

from run_llm_reward import load_splits as reward_splits, pairwise_loss
from run_llm_training import DEFAULT_MODEL, encode_row, first_exchange, load_splits, make_loader


class LLMTrainingTests(unittest.TestCase):
    def test_supervised_labels_only_train_on_answer(self):
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        row = {"prompt": "用户：你好\n助手：", "target": "您好！"}
        encoded = encode_row(row, tokenizer, "sft", 64, 32)
        self.assertEqual(encoded["labels"][-1], tokenizer.eos_token_id)
        first_label = next(index for index, value in enumerate(encoded["labels"]) if value != -100)
        self.assertGreater(first_label, 0)
        self.assertEqual(encoded["input_ids"][first_label:], encoded["labels"][first_label:])
        batch = next(iter(make_loader([row, {"prompt": "用户：很长的一句话\n助手：", "target": "好"}], tokenizer, "sft", 2, 64, 32)))
        self.assertTrue(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))

    def test_first_exchange_and_group_splits(self):
        self.assertEqual(first_exchange([{"role": "user", "content": "你好"}, {"role": "assistant", "content": "您好"}])["target"], "您好")
        for stage, key in (("pretrain", "text"), ("sft", "prompt"), ("mixed", "prompt")):
            train, valid, test, _ = load_splits(stage, max_train=16, max_eval=8)
            keys = [{row[key] for row in rows} for rows in (train, valid, test)]
            self.assertFalse(keys[0] & keys[1])
            self.assertFalse(keys[0] & keys[2])
            self.assertFalse(keys[1] & keys[2])

    def test_reward_preferences_are_isolated_and_loss_direction(self):
        train, valid, test, _ = reward_splits(max_train=16, max_eval=8)
        texts = [{value for row in rows for value in (row["selected"], row["rejected"])} for rows in (train, valid, test)]
        self.assertFalse(texts[0] & texts[1])
        self.assertFalse(texts[0] & texts[2])
        self.assertFalse(texts[1] & texts[2])
        good = pairwise_loss(torch.tensor([[2.0, -2.0]]))
        bad = pairwise_loss(torch.tensor([[-2.0, 2.0]]))
        self.assertLess(good, bad)


if __name__ == "__main__":
    unittest.main()
