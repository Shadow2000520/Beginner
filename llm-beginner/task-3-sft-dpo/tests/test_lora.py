import copy
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lora import LoRALinear, inject_lora, merge_lora


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(12, 12)
        self.v_proj = nn.Linear(12, 4, bias=False)
        self.other = nn.Linear(4, 2)

    def forward(self, x):
        return self.other(self.v_proj(self.q_proj(x)))


class LoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_initial_output_and_parameters(self):
        model = SmallModel().eval()
        x = torch.randn(2, 3, 12)
        before = model(x).detach()
        self.assertIs(inject_lora(model, ['q_proj', 'v_proj'], 2, 4), model)
        torch.testing.assert_close(model(x), before, atol=0, rtol=0)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(count, 2 * (12 + 12) + 2 * (12 + 4))
        self.assertFalse(model.q_proj.training)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.endswith(('lora_A', 'lora_B')))

    def test_formula_and_non_square_shapes(self):
        layer = LoRALinear(nn.Linear(7, 3).double(), r=2, alpha=6)
        with torch.no_grad():
            layer.lora_B.normal_()
        x = torch.randn(2, 4, 7, dtype=torch.float64)
        expected = layer.base_layer(x) + 3 * ((x @ layer.lora_A.T) @ layer.lora_B.T)
        torch.testing.assert_close(layer(x), expected)
        self.assertEqual(layer.lora_A.shape, (2, 7))
        self.assertEqual(layer.lora_B.shape, (3, 2))

    def test_gradients_only_adapters(self):
        model = inject_lora(SmallModel(), ['q_proj', 'v_proj'], 2, 4)
        x = torch.randn(2, 3, 12)
        model(x).square().mean().backward()
        self.assertGreater(model.q_proj.lora_B.grad.abs().sum().item(), 0)
        self.assertEqual(model.q_proj.lora_A.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.q_proj.lora_B.normal_(std=.1)
            model.v_proj.lora_B.normal_(std=.1)
        model(x).square().mean().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0, name)
            else:
                self.assertIsNone(p.grad)

    def test_merge_equivalence_and_idempotence(self):
        model = inject_lora(SmallModel().double(), ['q_proj', 'v_proj'], 2, 4).eval()
        with torch.no_grad():
            model.q_proj.lora_B.normal_(std=.1)
            model.v_proj.lora_B.normal_(std=.1)
        x = torch.randn(2, 3, 12, dtype=torch.float64)
        before = model(x).detach()
        merge_lora(model)
        torch.testing.assert_close(model(x), before, rtol=1e-10, atol=1e-10)
        self.assertFalse(any(isinstance(m, LoRALinear) for m in model.modules()))
        weights = copy.deepcopy(model.state_dict())
        merge_lora(model)
        for key, value in weights.items():
            torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)

    def test_nested_target_and_failures(self):
        model = nn.Sequential(SmallModel(), SmallModel())
        inject_lora(model, ['0.q_proj'], 2, 4)
        self.assertIsInstance(model[0].q_proj, LoRALinear)
        self.assertIsInstance(model[1].q_proj, nn.Linear)
        with self.assertRaises(ValueError):
            inject_lora(model, ['q_proj'], 2, 4)
        for targets, rank, alpha in [([], 2, 4), (['missing'], 2, 4),
                                    (['q_proj', 'missing'], 2, 4), (['q_proj'], 0, 4),
                                    (['q_proj'], 2, float('inf'))]:
            fresh = SmallModel()
            with self.assertRaises(ValueError):
                inject_lora(fresh, targets, rank, alpha)
            self.assertTrue(all(p.requires_grad for p in fresh.parameters()))


if __name__ == '__main__':
    unittest.main()
