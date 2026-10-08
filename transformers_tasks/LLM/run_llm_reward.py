"""Train a causal-LLM reward head on the LLMsTrainer preference pairs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_llm_training import DEFAULT_MODEL, ROOT, read_jsonl


TRAIN_PATH = ROOT / "LLMsTrainer" / "data" / "reward_model_data" / "sentiment_comments.jsonl"
TEST_PATH = ROOT / "LLMsTrainer" / "eval_data" / "reward_model" / "sentiment_comments_test.jsonl"


def load_splits(seed=42, max_train=None, max_eval=None):
    train = [row for row in read_jsonl(TRAIN_PATH) if all(row.get(key) for key in ("prompt", "selected", "rejected"))]
    test = [row for row in read_jsonl(TEST_PATH) if all(row.get(key) for key in ("prompt", "selected", "rejected"))]
    train_texts = {text for row in train for text in (row["selected"], row["rejected"])}
    test = [row for row in test if row["selected"] not in train_texts and row["rejected"] not in train_texts]
    random.Random(seed).shuffle(train)
    valid, train = train[:100], train[100:]
    full_counts = {"train": len(train), "valid": len(valid), "test": len(test)}
    if max_train:
        train = train[:max_train]
    if max_eval:
        valid, test = valid[:max_eval], test[:max_eval]
    if not train or not valid or not test:
        raise ValueError("Train, validation and test must all be nonempty")
    return train, valid, test, full_counts


def make_loader(rows, tokenizer, batch_size, max_length, shuffle=False):
    def collate(batch):
        texts = [row["prompt"] + response for row in batch for response in (row["selected"], row["rejected"])]
        return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


class RewardModel(nn.Module):
    def __init__(self, source, init_adapters=(), train_last_layers=1):
        super().__init__()
        local = Path(source).exists()
        causal = AutoModelForCausalLM.from_pretrained(source, local_files_only=local, dtype=torch.float32)
        for adapter in init_adapters:
            causal = PeftModel.from_pretrained(causal, adapter, local_files_only=True).merge_and_unload()
        self.backbone = causal.model
        self.score_head = nn.Linear(causal.config.hidden_size, 1)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if train_last_layers < 0 or train_last_layers > len(self.backbone.layers):
            raise ValueError("train_last_layers is out of range")
        for layer in self.backbone.layers[-train_last_layers:] if train_last_layers else ():
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        end = attention_mask.sum(dim=1).clamp_min(1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), end]
        return self.score_head(pooled.float()).squeeze(-1)


def pairwise_loss(scores):
    return nn.functional.softplus(-(scores[:, 0] - scores[:, 1])).mean()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    margins = []
    for batch in loader:
        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            scores = model(**inputs).reshape(-1, 2)
        margins.extend((scores[:, 0] - scores[:, 1]).float().cpu().tolist())
    return {"pair_accuracy": sum(margin > 0 for margin in margins) / len(margins), "mean_margin": sum(margins) / len(margins), "pairs": len(margins)}


def save_delta(model, tokenizer, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items() if key in trainable}, str(output_dir / "reward_delta.safetensors"))
    tokenizer.save_pretrained(output_dir)


def load_checkpoint(output_dir, device):
    metadata = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
    model = RewardModel(metadata["base_model"], metadata["init_adapters"], metadata["train_last_layers"])
    missing, unexpected = model.load_state_dict(load_file(str(output_dir / "reward_delta.safetensors")), strict=False)
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if unexpected or required.intersection(missing):
        raise ValueError("Saved reward delta is incomplete or incompatible")
    return metadata, model.to(device).eval(), tokenizer


def train(args, device):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_rows, valid_rows, test_rows, full_counts = load_splits(args.seed, args.max_train, args.max_eval)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=Path(args.model).exists())
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    loaders = [make_loader(rows, tokenizer, args.batch_size, args.max_length, shuffle=(i == 0)) for i, rows in enumerate((train_rows, valid_rows, test_rows))]
    model = RewardModel(args.model, args.init_adapter, args.train_last_layers).to(device)
    baseline = evaluate(model, loaders[1], device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=0.01)
    history, best_accuracy, best_epoch = [], -1.0, 0
    print(json.dumps({"full_counts": full_counts, "used_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)}, "baseline_valid": baseline}, ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders[0]:
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                scores = model(**inputs).reshape(-1, 2)
                loss = pairwise_loss(scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        valid = evaluate(model, loaders[1], device)
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), **valid})
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if valid["pair_accuracy"] > best_accuracy:
            best_accuracy, best_epoch = valid["pair_accuracy"], epoch
            save_delta(model, tokenizer, args.output_dir)
    metadata = {
        "base_model": args.model,
        "init_adapters": args.init_adapter,
        "train_last_layers": args.train_last_layers,
        "max_length": args.max_length,
        "seed": args.seed,
        "max_train": args.max_train,
        "max_eval": args.max_eval,
        "full_counts": full_counts,
        "used_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "baseline_valid": baseline,
        "history": history,
        "best_epoch": best_epoch,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _, best, saved_tokenizer = load_checkpoint(args.output_dir, device)
    metadata["valid"] = evaluate(best, make_loader(valid_rows, saved_tokenizer, args.batch_size, args.max_length), device)
    metadata["test"] = evaluate(best, make_loader(test_rows, saved_tokenizer, args.batch_size, args.max_length), device)
    (args.output_dir / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": metadata["test"]}, ensure_ascii=False), flush=True)


def reevaluate(args, device):
    metadata, model, tokenizer = load_checkpoint(args.output_dir, device)
    _, _, test_rows, _ = load_splits(metadata["seed"], metadata["max_train"], metadata["max_eval"])
    loader = make_loader(test_rows, tokenizer, args.batch_size, metadata["max_length"])
    print(json.dumps(evaluate(model, loader, device), ensure_ascii=False))


@torch.inference_mode()
def score(args, device):
    metadata, model, tokenizer = load_checkpoint(args.output_dir, device)
    if not args.text:
        raise ValueError("score needs at least one --text")
    texts = [args.prompt + value for value in args.text]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=metadata["max_length"], return_tensors="pt")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        values = model(**{key: value.to(device) for key, value in encoded.items()}).float().cpu().tolist()
    for text, value in zip(args.text, values):
        print(json.dumps({"text": text, "reward": value}, ensure_ascii=False))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate", "score"))
    parser.add_argument("--model", default=str(DEFAULT_MODEL) if DEFAULT_MODEL.exists() else "Qwen/Qwen2.5-0.5B")
    parser.add_argument("--init-adapter", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "training" / "reward")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-last-layers", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-eval", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", default="下面是一条正面的评论：")
    parser.add_argument("--text", action="append")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.mode == "train":
        train(args, device)
    elif args.mode == "evaluate":
        reevaluate(args, device)
    else:
        score(args, device)


if __name__ == "__main__":
    main()
