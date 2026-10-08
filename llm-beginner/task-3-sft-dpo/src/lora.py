import math

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16):
        super().__init__()
        if not isinstance(base_layer, nn.Linear) or not base_layer.weight.is_floating_point():
            raise TypeError("LoRA 仅支持浮点 nn.Linear")
        if type(r) is not int or r <= 0:
            raise ValueError("r 必须是正整数")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha 必须是有限正数")
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        weight = base_layer.weight
        self.lora_A = nn.Parameter(weight.new_empty(r, base_layer.in_features))
        self.lora_B = nn.Parameter(weight.new_zeros(base_layer.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        base_layer.requires_grad_(False)
        self.train(base_layer.training)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    @property
    def in_features(self):
        return self.base_layer.in_features

    @property
    def out_features(self):
        return self.base_layer.out_features

    def forward(self, x):
        update = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return self.base_layer(x) + self.scaling * update

    @torch.no_grad()
    def merged_layer(self):
        # 半精度权重先在 float32 中合并，再写回原 dtype。
        dtype = torch.float64 if self.weight.dtype == torch.float64 else torch.float32
        delta = self.lora_B.to(dtype) @ self.lora_A.to(dtype)
        merged = self.weight.to(dtype) + self.scaling * delta
        self.weight.copy_(merged.to(self.weight.dtype))
        self.base_layer.train(self.training)
        return self.base_layer


def inject_lora(model, target_modules, r=8, alpha=16):
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("模型已注入 LoRA，不能重复注入")
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    targets = list(target_modules)
    if not targets or any(not isinstance(t, str) or not t for t in targets):
        raise ValueError("target_modules 必须包含非空模块名")
    if type(r) is not int or r <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("r 必须是正整数，alpha 必须是有限正数")
    matches, found = [], set()
    for name, module in model.named_modules():
        selected = [t for t in targets if name == t or name.endswith('.' + t)]
        if selected:
            if not isinstance(module, nn.Linear) or not module.weight.is_floating_point():
                raise TypeError(f"目标模块 {name} 不是浮点 nn.Linear")
            matches.append((name, module))
            found.update(selected)
    missing = set(targets) - found
    if missing:
        raise ValueError(f"未找到目标模块：{sorted(missing)}")
    model.requires_grad_(False)
    for name, module in matches:
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, r, alpha))
    return model


def merge_lora(model):
    if isinstance(model, LoRALinear):
        return model.merged_layer()
    for name, module in list(model.named_children()):
        if isinstance(module, LoRALinear):
            setattr(model, name, module.merged_layer())
        else:
            merge_lora(module)
    return model
