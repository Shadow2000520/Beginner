import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chat import get_tokenizer
from src.dpo import dpo_loss, sequence_log_probs, encode_pair, collate_preferences


class DPOTests(unittest.TestCase):
    def test_initial_loss_and_gradient_direction(self):
        chosen = torch.tensor([-3., -4.], requires_grad=True)
        rejected = torch.tensor([-5., -6.], requires_grad=True)
        rc = chosen.detach().clone().requires_grad_()
        rr = rejected.detach().clone().requires_grad_()
        loss, margin, _, _ = dpo_loss(chosen, rejected, rc, rr)
        torch.testing.assert_close(loss, torch.full((2,), math.log(2)))
        self.assertTrue((margin == 0).all())
        loss.mean().backward()
        self.assertTrue((chosen.grad < 0).all())
        self.assertTrue((rejected.grad > 0).all())
        self.assertIsNone(rc.grad)
        self.assertIsNone(rr.grad)

    def test_sequence_shift_mask_and_sum(self):
        probs = torch.tensor([[[.8, .2], [.3, .7], [.4, .6], [.9, .1]]])
        labels = torch.tensor([[-100, 1, -100, 0]])
        result = sequence_log_probs(probs.log(), labels)
        torch.testing.assert_close(result, torch.tensor([math.log(.2) + math.log(.4)]))
        changed = probs.clone()
        changed[:, 1] = torch.tensor([.99, .01])
        torch.testing.assert_close(result, sequence_log_probs(changed.log(), labels))
        with self.assertRaises(ValueError):
            sequence_log_probs(probs.log(), torch.full_like(labels, -100))

    def test_margin_and_stability(self):
        z = torch.zeros(1)
        good, margin, _, _ = dpo_loss(torch.tensor([10.]), z, z, z, 1.)
        bad, _, _, _ = dpo_loss(torch.tensor([-10000.]), z, z, z, 1.)
        self.assertLess(good.item(), math.log(2))
        self.assertGreater(margin.item(), 0)
        self.assertTrue(torch.isfinite(bad).all())

    def test_pair_masks_only_final_answer(self):
        tok = get_tokenizer()
        row = {'conversations': [{'from': 'human', 'value': '你好'}, {'from': 'gpt', 'value': '历史回答'},
                                 {'from': 'human', 'value': '再问一次'}]}
        row['chosen'] = {'from': 'gpt', 'value': '较好的回答'}
        row['rejected'] = {'from': 'gpt', 'value': '差'}
        pair = encode_pair(row, tok, 128)
        for key, expected in [('chosen', '较好的回答'), ('rejected', '差')]:
            ids = pair[key]['labels']
            self.assertEqual(tok.decode(ids[ids != -100]), expected)
        batch = collate_preferences([pair, pair], tok.pad_token_id)
        self.assertEqual(batch['chosen']['input_ids'].shape[0], 2)
        row['rejected'] = row['chosen']
        with self.assertRaises(ValueError):
            encode_pair(row, tok, 128)
        with self.assertRaises(ValueError):
            encode_pair(row, tok, 3)


if __name__ == '__main__':
    unittest.main()
