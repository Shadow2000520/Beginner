import json
from pathlib import Path

import torch

from .lora import inject_lora


def save_adapter(model, directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {name: p.detach().cpu() for name, p in model.named_parameters()
             if name.endswith(('lora_A', 'lora_B'))}
    if not state:
        raise ValueError('模型未包含 LoRA 参数')
    torch.save(state, directory / 'adapter.pt')
    (directory / 'adapter_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')


def load_adapter(model, directory):
    directory = Path(directory)
    config = json.loads((directory / 'adapter_config.json').read_text(encoding='utf-8'))
    inject_lora(model, config['target_modules'], config['r'], config['alpha'])
    state = torch.load(directory / 'adapter.pt', map_location='cpu', weights_only=True)
    expected = {name for name, p in model.named_parameters() if name.endswith(('lora_A', 'lora_B'))}
    if set(state) != expected:
        raise ValueError('LoRA 权重键与模型不匹配')
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if parameters[name].shape != value.shape:
                raise ValueError(f'LoRA 权重形状不匹配：{name}')
            parameters[name].copy_(value)
    return model
