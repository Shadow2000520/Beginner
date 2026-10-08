import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# IDE 或直接运行本文件时，确保能找到与 tests 同级的 src 包。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attention import MultiHeadAttention, scaled_dot_product_attention
from src.block import TransformerBlock


class ComponentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_masked_attention_matches_reference_and_has_finite_gradients(self):
        Q, K, V = [torch.randn(2, 4, 5, 8, requires_grad=True) for _ in range(3)]
        mask = torch.zeros(2, 1, 5, 5, dtype=torch.bool)
        mask[..., -1] = True
        mask[0, :, 0, :] = True  # 包含整行屏蔽的边界情况
        out, weights = scaled_dot_product_attention(Q, K, V, mask, return_weights=True)
        ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=~mask)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
        self.assertTrue((weights.masked_select(mask.expand_as(weights)) == 0).all())
        out.square().sum().backward()
        for tensor in (Q, K, V):
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_multihead_matches_pytorch_with_same_parameters(self):
        ours = MultiHeadAttention(16, 4)
        reference = nn.MultiheadAttention(16, 4, dropout=0, batch_first=True)
        with torch.no_grad():
            reference.in_proj_weight.copy_(torch.cat([ours.q_proj.weight, ours.k_proj.weight, ours.v_proj.weight]))
            reference.in_proj_bias.copy_(torch.cat([ours.q_proj.bias, ours.k_proj.bias, ours.v_proj.bias]))
            reference.out_proj.load_state_dict(ours.out_proj.state_dict())
        x = torch.randn(2, 5, 16)
        padding = torch.tensor([[False, False, False, True, True], [False] * 5])
        actual, weights = ours(x, padding[:, None, None, :], return_weights=True)
        expected, ref_weights = reference(x, x, x, key_padding_mask=padding, average_attn_weights=False)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(weights, ref_weights)

    def test_block_shapes_gradients_and_causal_isolation(self):
        block = TransformerBlock(16, 4, dropout=0)
        x = torch.randn(2, 5, 16, requires_grad=True)
        causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        out, weights = block(x, causal, return_weights=True)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(weights.shape, (2, 4, 5, 5))
        changed = x.detach().clone()
        changed[:, -1] = torch.randn(2, 16) * 100
        torch.testing.assert_close(out[:, :-1], block(changed, causal)[:, :-1])
        out.square().mean().backward()
        for name, parameter in block.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_block_residual_preserves_input_when_branches_are_zero(self):
        block = TransformerBlock(16, 4, dropout=0)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
        x = torch.randn(2, 5, 16)
        torch.testing.assert_close(block(x), x)

    def test_invalid_head_count(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(15, 4)


if __name__ == "__main__":
    unittest.main()
