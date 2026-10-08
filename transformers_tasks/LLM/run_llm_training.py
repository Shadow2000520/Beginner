"""Local LoRA experiments for continued pretraining and instruction tuning."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-0.5B"
PRETRAIN = ROOT / "LLMsTrainer" / "data" / "shuffled_data" / "pretrain"
SFT = ROOT / "LLMsTrainer" / "data" / "shuffled_data" / "sft" / "sharegpt.jsonl"
MIXED = ROOT / "chatglm_finetune" / "data"


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def first_exchange(dialog):
    user = None
    for message in dialog:
        if message.get("role") == "user":
            user = message.get("content")
        elif message.get("role") == "assistant" and user and message.get("content"):
            return {"prompt": "用户：" + user.replace("\\n", "\n").strip() + "\n助手：", "target": message["content"].replace("\\n", "\n").strip()}
    return None


def split_grouped(rows, seed, valid_count, test_count, key):
    groups = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    valid, test, train = [], [], []
    for group in shuffled:
        if len(valid) < valid_count:
            valid.extend(group)
        elif len(test) < test_count:
            test.extend(group)
        else:
            train.extend(group)
    random.Random(seed + 1).shuffle(train)
    return train, valid, test


def load_splits(stage, seed=42, max_train=None, max_eval=None):
    if stage == "pretrain":
        rows = []
        for name in ("MNBVC_news", "MNBVC_qa", "MNBVC_wiki"):
            rows.extend({"text": item["text"].strip(), "source": name} for item in read_jsonl(PRETRAIN / f"{name}.jsonl") if item.get("text", "").strip())
        train, valid, test = split_grouped(rows, seed, 25, 25, lambda x: x["text"])
    elif stage == "sft":
        rows = []
        for item in read_jsonl(SFT):
            row = first_exchange(item.get("dialog", []))
            if row and row["target"]:
                rows.append(row)
        unique = list({(row["prompt"], row["target"]): row for row in rows}.values())
        train, valid, test = split_grouped(unique, seed, 128, 128, lambda x: x["prompt"])
    elif stage == "mixed":
        train = [row for row in read_jsonl(MIXED / "mixed_train_dataset.jsonl") if row.get("context") and row.get("target")]
        dev = [row for row in read_jsonl(MIXED / "mixed_dev_dataset.jsonl") if row.get("context") and row.get("target")]
        train_keys = {row["context"] for row in train}
        dev = [row for row in dev if row["context"] not in train_keys]
        random.Random(seed).shuffle(train)
        random.Random(seed + 1).shuffle(dev)
        midpoint = len(dev) // 2
        valid, test = dev[:midpoint], dev[midpoint:]
        train = [{"prompt": row["context"], "target": row["target"]} for row in train]
        valid = [{"prompt": row["context"], "target": row["target"]} for row in valid]
        test = [{"prompt": row["context"], "target": row["target"]} for row in test]
    else:
        raise ValueError(f"Unknown stage: {stage}")
    full_counts = {"train": len(train), "valid": len(valid), "test": len(test)}
    if max_train:
        train = train[:max_train]
    if max_eval:
        valid, test = valid[:max_eval], test[:max_eval]
    if not train or not valid or not test:
        raise ValueError("Train, validation and test splits must be nonempty")
    return train, valid, test, full_counts


def encode_row(row, tokenizer, stage, max_length, max_prompt_length):
    eos = tokenizer.eos_token_id
    if stage == "pretrain":
        ids = tokenizer.encode(row["text"], add_special_tokens=False, truncation=True, max_length=max_length - 1) + [eos]
        return {"input_ids": ids, "labels": ids.copy()}
    prompt = row["prompt"]
    target = row["target"]
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) > max_prompt_length:
        keep_tail = min(16, max_prompt_length // 4)
        prompt_ids = prompt_ids[: max_prompt_length - keep_tail] + prompt_ids[-keep_tail:]
    target_ids = tokenizer.encode(target, add_special_tokens=False, truncation=True, max_length=max_length - len(prompt_ids) - 1)
    if not target_ids:
        raise ValueError("No target tokens remain after truncation")
    ids = prompt_ids + target_ids + [eos]
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + target_ids + [eos]}


def make_loader(rows, tokenizer, stage, batch_size, max_length, max_prompt_length, shuffle=False):
    encoded = [encode_row(row, tokenizer, stage, max_length, max_prompt_length) for row in rows]

    def collate(batch):
        width = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for index, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[index, :length] = torch.tensor(item["input_ids"])
            labels[index, :length] = torch.tensor(item["labels"])
            mask[index, :length] = 1
        return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}

    return DataLoader(encoded, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def load_base(source, device, init_adapters=()):
    local = Path(source).exists()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(source, local_files_only=local, dtype=dtype)
    for adapter in init_adapters:
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload()
    return model.to(device)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        count = int((batch["labels"][:, 1:] != -100).sum())
        inputs = {key: value.to(device) for key, value in batch.items()}
        loss = model(**inputs, use_cache=False).loss
        total_loss += float(loss) * count
        total_tokens += count
    nll = total_loss / total_tokens
    return {"token_nll": nll, "perplexity": math.exp(min(nll, 50)), "evaluated_tokens": total_tokens}


def train(args, device):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_rows, valid_rows, test_rows, full_counts = load_splits(args.stage, args.seed, args.max_train, args.max_eval)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=Path(args.model).exists())
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_prompt_length = min(args.max_prompt_length, args.max_length - 2)
    loaders = [make_loader(rows, tokenizer, args.stage, args.batch_size, args.max_length, max_prompt_length, shuffle=(i == 0)) for i, rows in enumerate((train_rows, valid_rows, test_rows))]
    model = load_base(args.model, device, args.init_adapter)
    baseline = evaluate(model, loaders[1], device)
    config = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05, bias="none", target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, config)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=0.01)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_nll, best_epoch, history = float("inf"), 0, []
    print(json.dumps({"stage": args.stage, "full_counts": full_counts, "used_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)}, "baseline_valid": baseline}, ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        weighted_loss, token_count = 0.0, 0
        for batch in loaders[0]:
            count = int((batch["labels"][:, 1:] != -100).sum())
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, use_cache=False).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            weighted_loss += float(loss.detach()) * count
            token_count += count
        valid_metrics = evaluate(model, loaders[1], device)
        history.append({"epoch": epoch, "train_token_nll": weighted_loss / token_count, "valid_token_nll": valid_metrics["token_nll"], "valid_perplexity": valid_metrics["perplexity"]})
        print(json.dumps({"stage": args.stage, **history[-1]}, ensure_ascii=False), flush=True)
        if valid_metrics["token_nll"] < best_nll:
            best_nll, best_epoch = valid_metrics["token_nll"], epoch
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
    del model
    model = PeftModel.from_pretrained(load_base(args.model, device, args.init_adapter), args.output_dir, local_files_only=True)
    metadata = {
        "stage": args.stage,
        "base_model": args.model,
        "init_adapters": args.init_adapter,
        "seed": args.seed,
        "max_train": args.max_train,
        "max_eval": args.max_eval,
        "max_length": args.max_length,
        "max_prompt_length": max_prompt_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "full_counts": full_counts,
        "used_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "baseline_valid": baseline,
        "history": history,
        "best_epoch": best_epoch,
        "valid": evaluate(model, loaders[1], device),
        "test": evaluate(model, loaders[2], device),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": args.stage, "best_epoch": best_epoch, "test": metadata["test"]}, ensure_ascii=False), flush=True)


def load_checkpoint(output_dir, device):
    metadata = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
    model = PeftModel.from_pretrained(load_base(metadata["base_model"], device, metadata["init_adapters"]), output_dir, local_files_only=True).eval()
    return metadata, model, tokenizer


def reevaluate(args, device):
    metadata, model, tokenizer = load_checkpoint(args.output_dir, device)
    if metadata["stage"] != args.stage:
        raise ValueError("Checkpoint stage and --stage differ")
    _, _, test_rows, _ = load_splits(metadata["stage"], metadata["seed"], metadata["max_train"], metadata["max_eval"])
    loader = make_loader(test_rows, tokenizer, args.stage, args.batch_size, metadata["max_length"], metadata["max_prompt_length"])
    print(json.dumps(evaluate(model, loader, device), ensure_ascii=False))


@torch.inference_mode()
def predict(args, device):
    metadata, model, tokenizer = load_checkpoint(args.output_dir, device)
    if metadata["stage"] != args.stage:
        raise ValueError("Checkpoint stage and --stage differ")
    if not args.prompt:
        raise ValueError("predict needs --prompt")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    print(json.dumps({"stage": args.stage, "prompt": args.prompt, "response": response}, ensure_ascii=False))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate", "predict"))
    parser.add_argument("--stage", choices=("pretrain", "sft", "mixed"), required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL) if DEFAULT_MODEL.exists() else "Qwen/Qwen2.5-0.5B")
    parser.add_argument("--init-adapter", action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-prompt-length", type=int, default=112)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-eval", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir = args.output_dir or ROOT / "runs" / "training" / args.stage
    if args.max_length < 8 or args.max_prompt_length < 1 or args.epochs < 1:
        parser.error("max lengths and epochs must be positive")
    device = torch.device(args.device)
    if args.mode == "train":
        train(args, device)
    elif args.mode == "evaluate":
        reevaluate(args, device)
    else:
        predict(args, device)


if __name__ == "__main__":
    main()
