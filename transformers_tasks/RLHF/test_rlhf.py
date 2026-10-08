"""Core checks for ranking loss and PPO's clipped policy update."""

import unittest
from tempfile import TemporaryDirectory

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from run_ppo import load_policy, ppo_update, sequence_logprob_and_value
from run_reward_model import load_splits, ranking_loss


class RewardAndPPOTests(unittest.TestCase):
    def test_rank_splits_have_no_sentence_leakage(self):
        train, valid, test, removed = load_splits(max_train=100, max_eval=100)
        self.assertEqual((len(train), len(valid), len(test)), (100, 50, 50))
        self.assertGreater(removed, 0)
        original_train, _, _, _ = load_splits()
        all_train_sentences = {text for row in original_train for text in row}
        self.assertFalse(all_train_sentences & {text for row in valid + test for text in row})
        self.assertFalse({text for row in valid for text in row} & {text for row in test for text in row})

    def test_pairwise_rank_loss_prefers_ordered_scores(self):
        good = ranking_loss(torch.tensor([[3.0, 2.0, 1.0, 0.0]]))
        bad = ranking_loss(torch.tensor([[0.0, 1.0, 2.0, 3.0]]))
        self.assertLess(float(good), float(bad))

    def test_sequence_ppo_changes_policy_parameters(self):
        torch.manual_seed(7)
        config = GPT2Config(vocab_size=32, n_positions=32, n_ctx=32, n_embd=32, n_layer=1, n_head=2, bos_token_id=1, eos_token_id=2)
        model = GPT2LMHeadModel(config)
        reference = GPT2LMHeadModel(config)
        reference.load_state_dict(model.state_dict())
        reference.eval()
        value_head = nn.Linear(32, 1)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(value_head.parameters()), lr=1e-3)
        sequences = torch.tensor([[1, 4, 5, 6, 7], [1, 3, 4, 8, 9]])
        attention_mask = torch.ones_like(sequences)
        logp, values = sequence_logprob_and_value(model, value_head, sequences, attention_mask, 3, 2)
        self.assertEqual(tuple(logp.shape), (2,))
        self.assertEqual(tuple(values.shape), (2,))
        before = model.transformer.h[0].attn.c_attn.weight.detach().clone()
        args = type("Args", (), {"kl_coef": 0.05, "ppo_epochs": 2, "clip_range": 0.2, "value_coef": 0.5})()
        stats = ppo_update(model, reference, value_head, optimizer, sequences, attention_mask, 3, torch.tensor([0.9, 0.1]), args)
        self.assertIn("policy_loss", stats)
        self.assertFalse(torch.equal(before, model.transformer.h[0].attn.c_attn.weight))

    def test_gpt2_policy_freezes_earlier_layers(self):
        config = GPT2Config(vocab_size=32, n_positions=32, n_ctx=32, n_embd=32, n_layer=2, n_head=2, bos_token_id=1, eos_token_id=2)
        with TemporaryDirectory() as directory:
            GPT2LMHeadModel(config).save_pretrained(directory, safe_serialization=True)
            model = load_policy(directory, torch.device("cpu"), train_last_layers=1)
            self.assertFalse(any(parameter.requires_grad for parameter in model.transformer.h[0].parameters()))
            self.assertTrue(all(parameter.requires_grad for parameter in model.transformer.h[1].parameters()))


if __name__ == "__main__":
    unittest.main()
