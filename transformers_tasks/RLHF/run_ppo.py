"""Small sequence-level PPO experiment using a rank-trained reward model.

The whole continuation is treated as one action. The clipped policy objective,
learned value baseline and frozen reference KL are explicit so the experiment
does not depend on the repository's obsolete GPT-2/TRL implementation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_reward_model import load_checkpoint as load_reward_checkpoint


ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY = Path(r"E:\Workspace\Beginner\llm-beginner\task-3-sft-dpo\models\Qwen2.5-0.5B")
PROMPTS = [
    "刚收到货，感觉",
    "这次购物总的来说",
    "说实话，这家酒店",
    "这件衣服穿起来",
    "这个苹果吃起来",
    "新买的手机用起来",
]
HELDOUT_PROMPTS = [
    "快递送到后打开包装，",
    "我在这家宾馆住了一晚，",
    "这台笔记本电脑用了几天，",
    "刚买的水果尝了一口，",
    "洗发水用完以后感觉，",
    "收到的新衣服试穿后，",
]


def prepare_tokenizer(source):
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=Path(source).exists())
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_policy(source, device, train_last_layers=2):
    model = AutoModelForCausalLM.from_pretrained(source, local_files_only=Path(source).exists(), dtype=torch.float32).to(device)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        final_norm = model.model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        final_norm = model.transformer.ln_f
    else:
        raise ValueError("This PPO runner supports Qwen-style and GPT-2-style causal LMs")
    for parameter in model.parameters():
        parameter.requires_grad = False
    n_layers = len(layers)
    if not 1 <= train_last_layers <= n_layers:
        raise ValueError("train_last_layers must be between 1 and the number of decoder layers")
    for layer in layers[-train_last_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for parameter in final_norm.parameters():
        parameter.requires_grad = True
    return model


def load_reference(source, device):
    model = AutoModelForCausalLM.from_pretrained(source, local_files_only=Path(source).exists(), dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def sequence_logprob_and_value(model, value_head, sequences, attention_mask, prompt_width, response_length):
    outputs = model(input_ids=sequences, attention_mask=attention_mask, output_hidden_states=value_head is not None, use_cache=False)
    logits = outputs.logits[:, prompt_width - 1 : prompt_width + response_length - 1]
    targets = sequences[:, prompt_width : prompt_width + response_length]
    log_probs = logits.float().log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    values = None
    if value_head is not None:
        prompt_state = outputs.hidden_states[-1][:, prompt_width - 1].float()
        values = value_head(prompt_state).squeeze(-1)
    return log_probs, values


@torch.no_grad()
def generate_rollout(model, tokenizer, prompts, device, response_length, temperature, top_p):
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = inputs["input_ids"].size(1)
    model.eval()
    sequences = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=response_length,
        min_new_tokens=response_length,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if sequences.size(1) != prompt_width + response_length:
        raise RuntimeError("Generation stopped before the requested response length")
    attention_mask = torch.cat((inputs["attention_mask"], torch.ones(len(prompts), response_length, dtype=torch.long, device=device)), dim=1)
    responses = tokenizer.batch_decode(sequences[:, prompt_width:], skip_special_tokens=True)
    return sequences, attention_mask, prompt_width, responses


@torch.no_grad()
def reward_scores(model, tokenizer, texts, device, max_length):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    raw = model(**{key: value.to(device) for key, value in encoded.items()})
    return raw, torch.sigmoid(raw / 4.0)


def ppo_update(model, reference, value_head, optimizer, sequences, attention_mask, prompt_width, reward, args):
    response_length = sequences.size(1) - prompt_width
    with torch.no_grad():
        old_logp, old_value = sequence_logprob_and_value(model, value_head, sequences, attention_mask, prompt_width, response_length)
        ref_logp, _ = sequence_logprob_and_value(reference, None, sequences, attention_mask, prompt_width, response_length)
        sampled_kl = (old_logp - ref_logp) / response_length
        returns = reward - args.kl_coef * sampled_kl
        advantages = returns - old_value
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    losses = []
    model.train()
    for _ in range(args.ppo_epochs):
        new_logp, new_value = sequence_logprob_and_value(model, value_head, sequences, attention_mask, prompt_width, response_length)
        ratio = (new_logp - old_logp).clamp(-20, 20).exp()
        unclipped = ratio * advantages
        clipped = ratio.clamp(1 - args.clip_range, 1 + args.clip_range) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = 0.5 * (new_value - returns).square().mean()
        loss = policy_loss + args.value_coef * value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad] + list(value_head.parameters()), 1.0)
        optimizer.step()
        losses.append({"policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()), "ratio_mean": float(ratio.mean().detach())})
    return {"reward_mean": float(reward.mean()), "sampled_kl_per_token": float(sampled_kl.mean()), **losses[-1]}


def save_policy(model, value_head, tokenizer, directory, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items() if name in trainable}
    save_file(state, str(directory / "policy_delta.safetensors"))
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in value_head.state_dict().items()}, str(directory / "value_head.safetensors"))
    tokenizer.save_pretrained(directory)
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def restore_policy(directory, device):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    source = metadata["base_model"]
    model = load_policy(source, device, metadata["train_last_layers"])
    missing, unexpected = model.load_state_dict(load_file(str(directory / "policy_delta.safetensors"), device=str(device)), strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected policy weights: {unexpected}")
    value_head = nn.Linear(model.config.hidden_size, 1).to(device)
    value_head.load_state_dict(load_file(str(directory / "value_head.safetensors"), device=str(device)))
    tokenizer = prepare_tokenizer(directory)
    return metadata, model, value_head, tokenizer


@torch.no_grad()
def assess(model, tokenizer, reward_model, reward_tokenizer, prompts, device, args, seed):
    torch.manual_seed(seed)
    sequences, attention_mask, width, responses = generate_rollout(model, tokenizer, prompts, device, args.response_length, args.temperature, args.top_p)
    texts = [prompt + response for prompt, response in zip(prompts, responses)]
    raw, bounded = reward_scores(reward_model, reward_tokenizer, texts, device, args.reward_max_length)
    return {"mean_raw_reward": float(raw.mean()), "mean_bounded_reward": float(bounded.mean()), "samples": [{"prompt": prompt, "response": response, "raw_reward": round(float(score), 4)} for prompt, response, score in zip(prompts, responses, raw)]}


def train(args, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    if not (args.reward_dir / "reward_model.safetensors").exists():
        raise FileNotFoundError("Train the reward model first, then pass --reward-dir")
    reward_model, reward_tokenizer = load_reward_checkpoint(args.reward_dir, device)
    reward_meta = json.loads((args.reward_dir / "metrics.json").read_text(encoding="utf-8"))
    tokenizer = prepare_tokenizer(args.model)
    model = load_policy(args.model, device, args.train_last_layers)
    reference = load_reference(args.model, device)
    value_head = nn.Linear(model.config.hidden_size, 1).to(device)
    nn.init.zeros_(value_head.weight)
    nn.init.zeros_(value_head.bias)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad] + list(value_head.parameters()), lr=args.learning_rate)
    fixed_prompts = PROMPTS[: args.eval_prompts]
    before = assess(model, tokenizer, reward_model, reward_tokenizer, fixed_prompts, device, args, args.seed + 100)
    print(json.dumps({"before": before}, ensure_ascii=False), flush=True)
    history = []
    for step in range(1, args.steps + 1):
        prompts = [random.choice(PROMPTS) for _ in range(args.batch_size)]
        sequences, attention_mask, width, responses = generate_rollout(model, tokenizer, prompts, device, args.response_length, args.temperature, args.top_p)
        texts = [prompt + response for prompt, response in zip(prompts, responses)]
        raw, bounded = reward_scores(reward_model, reward_tokenizer, texts, device, args.reward_max_length)
        stats = ppo_update(model, reference, value_head, optimizer, sequences, attention_mask, width, bounded, args)
        history.append({"step": step, "raw_reward_mean": float(raw.mean()), **stats})
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
    after = assess(model, tokenizer, reward_model, reward_tokenizer, fixed_prompts, device, args, args.seed + 100)
    metadata = {
        "base_model": str(args.model),
        "reward_dir": str(args.reward_dir.resolve()),
        "reward_test_rank_accuracy": reward_meta["test"]["rank_accuracy"],
        "train_last_layers": args.train_last_layers,
        "response_length": args.response_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "reward_max_length": args.reward_max_length,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "ppo_epochs": args.ppo_epochs,
        "kl_coef": args.kl_coef,
        "clip_range": args.clip_range,
        "before": before,
        "after": after,
        "history": history,
    }
    save_policy(model, value_head, tokenizer, args.output_dir, metadata)
    print(json.dumps({"before_reward": before["mean_bounded_reward"], "after_reward": after["mean_bounded_reward"], "output_dir": str(args.output_dir)}, ensure_ascii=False), flush=True)


def generate(args, device):
    metadata, model, _, tokenizer = restore_policy(args.output_dir, device)
    prompts = args.prompt or PROMPTS[:2]
    torch.manual_seed(args.seed)
    _, _, _, responses = generate_rollout(model, tokenizer, prompts, device, metadata["response_length"], metadata["temperature"], metadata["top_p"])
    for prompt, response in zip(prompts, responses):
        print(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False))


def evaluate_policy(args, device):
    metadata, policy, _, tokenizer = restore_policy(args.output_dir, device)
    baseline = load_policy(metadata["base_model"], device, metadata["train_last_layers"])
    reward_model, reward_tokenizer = load_reward_checkpoint(args.reward_dir, device)
    args.response_length = metadata["response_length"]
    args.temperature = metadata["temperature"]
    args.top_p = metadata["top_p"]
    args.reward_max_length = metadata["reward_max_length"]
    prompts = HELDOUT_PROMPTS * args.samples_per_prompt
    before = assess(baseline, tokenizer, reward_model, reward_tokenizer, prompts, device, args, args.seed)
    after = assess(policy, tokenizer, reward_model, reward_tokenizer, prompts, device, args, args.seed)
    result = {
        "prompts": HELDOUT_PROMPTS,
        "samples_per_prompt": args.samples_per_prompt,
        "seed": args.seed,
        "before_mean_bounded_reward": before["mean_bounded_reward"],
        "after_mean_bounded_reward": after["mean_bounded_reward"],
        "before_mean_raw_reward": before["mean_raw_reward"],
        "after_mean_raw_reward": after["mean_raw_reward"],
        "before_samples": before["samples"],
        "after_samples": after["samples"],
    }
    (args.output_dir / "heldout_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("samples")}, ensure_ascii=False))


def human_feedback(args, device):
    metadata, model, value_head, tokenizer = restore_policy(args.output_dir, device)
    reference = load_reference(metadata["base_model"], device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad] + list(value_head.parameters()), lr=args.learning_rate)
    args.kl_coef = metadata["kl_coef"]
    args.clip_range = metadata["clip_range"]
    args.ppo_epochs = metadata["ppo_epochs"]
    args.response_length = metadata["response_length"]
    args.temperature = metadata["temperature"]
    args.top_p = metadata["top_p"]
    feedback = metadata.get("human_feedback", [])
    for _ in range(args.max_human_steps):
        prompt = random.choice(PROMPTS)
        sequences, attention_mask, width, responses = generate_rollout(model, tokenizer, [prompt], device, args.response_length, args.temperature, args.top_p)
        print(f"提示：{prompt}\n续写：{responses[0]}")
        response = input("评分 [-2, 2]，或输入 q 结束：").strip()
        if response.lower() == "q":
            break
        try:
            human_score = float(response)
        except ValueError:
            print("请输入数字或 q。")
            continue
        if not -2 <= human_score <= 2:
            print("评分必须在 -2 到 2 之间。")
            continue
        bounded_reward = torch.tensor([(human_score + 2) / 4], device=device)
        stats = ppo_update(model, reference, value_head, optimizer, sequences, attention_mask, width, bounded_reward, args)
        feedback.append({"prompt": prompt, "response": responses[0], "human_score": human_score, **stats})
        metadata["human_feedback"] = feedback
        save_policy(model, value_head, tokenizer, args.output_dir, metadata)
        print(json.dumps({"human_steps": len(feedback), **stats}, ensure_ascii=False))


def label_ranklists(args, device):
    metadata, model, _, tokenizer = restore_policy(args.output_dir, device)
    args.response_length = metadata["response_length"]
    args.temperature = metadata["temperature"]
    args.top_p = metadata["top_p"]
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(args.max_human_steps):
        prompt = random.choice(PROMPTS)
        prompts = [prompt] * 4
        _, _, _, responses = generate_rollout(model, tokenizer, prompts, device, args.response_length, args.temperature, args.top_p)
        print(f"提示：{prompt}")
        for number, response in enumerate(responses, 1):
            print(f"{number}. {response}")
        raw = input("按优到劣输入四个序号，如 2 4 1 3；输入 q 结束：").strip()
        if raw.lower() == "q":
            break
        try:
            order = [int(part) for part in raw.split()]
        except ValueError:
            print("请输入四个序号。")
            continue
        if sorted(order) != [1, 2, 3, 4]:
            print("序号必须是 1、2、3、4，且每个只出现一次。")
            continue
        with args.output_file.open("a", encoding="utf-8") as handle:
            ranked_texts = [(prompt + responses[index - 1]).replace("\n", " ").replace("\t", " ") for index in order]
            handle.write("\t".join(ranked_texts) + "\n")
        print(f"已保存到 {args.output_file}")


def main():
    if hasattr(sys.stdout, "reconfigure") and (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "generate", "evaluate", "human", "label"])
    parser.add_argument("--model", default=str(DEFAULT_POLICY))
    parser.add_argument("--reward-dir", type=Path, default=ROOT / "runs" / "reward_model")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "ppo")
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
    parser.add_argument("--max-human-steps", type=int, default=10)
    parser.add_argument("--output-file", type=Path, default=ROOT / "data" / "human_labeled" / "total_dataset.tsv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", action="append", help="Prompt to continue in generate mode")
    args = parser.parse_args()
    if args.mode != "generate" and args.prompt:
        parser.error("--prompt is only supported in generate mode")
    if args.steps < 1 or args.batch_size < 2 or args.ppo_epochs < 1:
        parser.error("steps and ppo_epochs must be positive, batch_size at least 2")
    device = torch.device(args.device)
    if args.mode == "train":
        train(args, device)
    elif args.mode == "generate":
        generate(args, device)
    elif args.mode == "evaluate":
        evaluate_policy(args, device)
    elif args.mode == "human":
        human_feedback(args, device)
    else:
        label_ranklists(args, device)


if __name__ == "__main__":
    main()
