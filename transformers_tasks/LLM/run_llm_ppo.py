"""Sequence-level PPO using the Task 8 SFT policy and reward checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoTokenizer

from run_llm_reward import load_checkpoint as load_reward_checkpoint
from run_llm_training import DEFAULT_MODEL, ROOT, load_base


sys.path.insert(0, str(ROOT.parent / "RLHF"))
import run_ppo as core  # noqa: E402 - reuse the sequence-level PPO objective from Task 5


PROMPTS = core.PROMPTS
HELDOUT_PROMPTS = core.HELDOUT_PROMPTS


def load_policy(source, init_adapters, device, train_last_layers):
    model = load_base(source, device, init_adapters).float()
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = model.model.layers
    if not 1 <= train_last_layers <= len(layers):
        raise ValueError("train_last_layers must be in range")
    for layer in layers[-train_last_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for parameter in model.model.norm.parameters():
        parameter.requires_grad = True
    return model


def tokenizer_for_policy(source):
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=Path(source).exists())
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


@torch.inference_mode()
def reward_for_texts(model, tokenizer, texts, device, max_length):
    inputs = tokenizer(["下面是一条正面的评论：" + text for text in texts], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        raw = model(**{key: value.to(device) for key, value in inputs.items()}).float()
    return raw, torch.sigmoid(raw / 8.0)


@torch.inference_mode()
def assess(policy, tokenizer, reward_model, reward_tokenizer, prompts, args, device, seed):
    torch.manual_seed(seed)
    _, _, _, responses = core.generate_rollout(policy, tokenizer, prompts, device, args.response_length, args.temperature, args.top_p)
    texts = [prompt + response for prompt, response in zip(prompts, responses)]
    raw, bounded = reward_for_texts(reward_model, reward_tokenizer, texts, device, args.reward_max_length)
    return {"mean_raw_reward": float(raw.mean()), "mean_bounded_reward": float(bounded.mean()), "samples": [{"prompt": prompt, "response": response, "raw_reward": float(value)} for prompt, response, value in zip(prompts, responses, raw)]}


def save_checkpoint(model, value_head, tokenizer, directory, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items() if name in trainable}, str(directory / "policy_delta.safetensors"))
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in value_head.state_dict().items()}, str(directory / "value_head.safetensors"))
    tokenizer.save_pretrained(directory)
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(directory, device):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    model = load_policy(metadata["base_model"], metadata["init_adapters"], device, metadata["train_last_layers"])
    missing, unexpected = model.load_state_dict(load_file(str(directory / "policy_delta.safetensors")), strict=False)
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if unexpected or required.intersection(missing):
        raise ValueError("Saved policy delta is incomplete or incompatible")
    value_head = nn.Linear(model.config.hidden_size, 1).to(device)
    value_head.load_state_dict(load_file(str(directory / "value_head.safetensors")))
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    tokenizer.padding_side = "left"
    return metadata, model.eval(), value_head.eval(), tokenizer


def train(args, device):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    reward_meta, reward_model, reward_tokenizer = load_reward_checkpoint(args.reward_dir, device)
    policy = load_policy(args.model, args.init_adapter, device, args.train_last_layers)
    reference = load_base(args.model, device, args.init_adapter).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    tokenizer = tokenizer_for_policy(args.model)
    value_head = nn.Linear(policy.config.hidden_size, 1).to(device)
    nn.init.zeros_(value_head.weight)
    nn.init.zeros_(value_head.bias)
    optimizer = torch.optim.AdamW([parameter for parameter in policy.parameters() if parameter.requires_grad] + list(value_head.parameters()), lr=args.learning_rate)
    before = assess(policy, tokenizer, reward_model, reward_tokenizer, PROMPTS[: args.eval_prompts], args, device, args.seed + 100)
    print(json.dumps({"before_mean_reward": before["mean_bounded_reward"], "reward_pair_accuracy": reward_meta["test"]["pair_accuracy"]}, ensure_ascii=False), flush=True)
    history = []
    for step in range(1, args.steps + 1):
        prompts = [random.choice(PROMPTS) for _ in range(args.batch_size)]
        sequences, mask, width, responses = core.generate_rollout(policy, tokenizer, prompts, device, args.response_length, args.temperature, args.top_p)
        raw, bounded = reward_for_texts(reward_model, reward_tokenizer, [prompt + response for prompt, response in zip(prompts, responses)], device, args.reward_max_length)
        stats = core.ppo_update(policy, reference, value_head, optimizer, sequences, mask, width, bounded, args)
        history.append({"step": step, "raw_reward_mean": float(raw.mean()), **stats})
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
    after = assess(policy, tokenizer, reward_model, reward_tokenizer, PROMPTS[: args.eval_prompts], args, device, args.seed + 100)
    metadata = {
        "base_model": args.model,
        "init_adapters": args.init_adapter,
        "reward_dir": str(args.reward_dir),
        "reward_test_pair_accuracy": reward_meta["test"]["pair_accuracy"],
        "train_last_layers": args.train_last_layers,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "ppo_epochs": args.ppo_epochs,
        "response_length": args.response_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "reward_max_length": args.reward_max_length,
        "learning_rate": args.learning_rate,
        "kl_coef": args.kl_coef,
        "value_coef": args.value_coef,
        "clip_range": args.clip_range,
        "seed": args.seed,
        "before": before,
        "after": after,
        "history": history,
        "method": "sequence-level clipped PPO with value baseline and frozen reference KL",
    }
    save_checkpoint(policy, value_head, tokenizer, args.output_dir, metadata)
    print(json.dumps({"before_reward": before["mean_bounded_reward"], "after_reward": after["mean_bounded_reward"], "output_dir": str(args.output_dir)}, ensure_ascii=False), flush=True)


def evaluate(args, device):
    metadata, policy, _, tokenizer = load_checkpoint(args.output_dir, device)
    baseline = load_policy(metadata["base_model"], metadata["init_adapters"], device, metadata["train_last_layers"])
    _, reward_model, reward_tokenizer = load_reward_checkpoint(Path(metadata["reward_dir"]), device)
    for key in ("response_length", "temperature", "top_p", "reward_max_length"):
        setattr(args, key, metadata[key])
    prompts = HELDOUT_PROMPTS * args.samples_per_prompt
    before = assess(baseline, tokenizer, reward_model, reward_tokenizer, prompts, args, device, args.seed)
    after = assess(policy, tokenizer, reward_model, reward_tokenizer, prompts, args, device, args.seed)
    result = {"samples_per_prompt": args.samples_per_prompt, "seed": args.seed, "before_mean_bounded_reward": before["mean_bounded_reward"], "after_mean_bounded_reward": after["mean_bounded_reward"], "before_mean_raw_reward": before["mean_raw_reward"], "after_mean_raw_reward": after["mean_raw_reward"], "before_samples": before["samples"], "after_samples": after["samples"]}
    (args.output_dir / "heldout_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("samples")}, ensure_ascii=False))


@torch.inference_mode()
def generate(args, device):
    metadata, policy, _, tokenizer = load_checkpoint(args.output_dir, device)
    prompts = args.prompt or PROMPTS[:2]
    torch.manual_seed(args.seed)
    _, _, _, responses = core.generate_rollout(policy, tokenizer, prompts, device, metadata["response_length"], metadata["temperature"], metadata["top_p"])
    for prompt, response in zip(prompts, responses):
        print(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate", "generate"))
    parser.add_argument("--model", default=str(DEFAULT_MODEL) if DEFAULT_MODEL.exists() else "Qwen/Qwen2.5-0.5B")
    parser.add_argument("--init-adapter", action="append", default=[])
    parser.add_argument("--reward-dir", type=Path, default=ROOT / "runs" / "training" / "reward")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "training" / "ppo")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--response-length", type=int, default=12)
    parser.add_argument("--reward-max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--train-last-layers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--eval-prompts", type=int, default=6)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.ppo_epochs < 1:
        parser.error("steps and ppo_epochs must be positive, batch_size at least 2")
    device = torch.device(args.device)
    if args.mode == "train":
        train(args, device)
    elif args.mode == "evaluate":
        evaluate(args, device)
    else:
        generate(args, device)


if __name__ == "__main__":
    main()
