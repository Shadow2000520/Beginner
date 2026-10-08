"""Train and evaluate a rank-list reward model for Chinese product reviews."""

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
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "reward_datasets" / "sentiment_analysis"
DEFAULT_MODEL = Path(r"E:\Workspace\Beginner\llm-beginner\task-4-rag\models\bge-small-zh-v1.5")


def read_ranklists(path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        texts = [part.strip() for part in line.split("\t")]
        if len(texts) != 4 or any(not text for text in texts):
            raise ValueError(f"Expected four ranked nonempty texts at {path}:{number}")
        rows.append(tuple(texts))
    return rows


def load_splits(seed=42, max_train=None, max_eval=None, train_path=None, dev_path=None):
    train = read_ranklists(train_path or DATA / "train.tsv")
    if train_path is not None and dev_path is None:
        rng_for_holdout = random.Random(seed)
        rng_for_holdout.shuffle(train)
        n_dev = max(4, round(0.2 * len(train)))
        dev, train = train[:n_dev], train[n_dev:]
    else:
        dev = read_ranklists(dev_path or DATA / "dev.tsv")
    train_sentences = {text for row in train for text in row}
    filtered = []
    seen = set(train_sentences)
    removed = 0
    for row in dev:
        if any(text in seen for text in row):
            removed += 1
            continue
        filtered.append(row)
        seen.update(row)
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(filtered)
    if max_train is not None:
        train = train[:max_train]
    if max_eval is not None:
        filtered = filtered[:max_eval]
    midpoint = len(filtered) // 2
    return train, filtered[:midpoint], filtered[midpoint:], removed


def make_loader(rows, tokenizer, batch_size, max_length, shuffle=False):
    def collate(batch):
        texts = [text for row in batch for text in row]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded["rank_count"] = len(batch)
        return encoded

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


class RewardModel(nn.Module):
    def __init__(self, source, from_config=False):
        super().__init__()
        local = Path(source).exists()
        if from_config:
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(source, local_files_only=local))
        else:
            self.encoder = AutoModel.from_pretrained(source, local_files_only=local)
        self.score_head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, **inputs):
        attention_mask = inputs["attention_mask"]
        hidden = self.encoder(**inputs).last_hidden_state
        pooled = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.score_head(pooled).squeeze(-1)


def ranking_loss(scores):
    differences = scores[:, :, None] - scores[:, None, :]
    upper = torch.triu(torch.ones(scores.size(1), scores.size(1), dtype=torch.bool, device=scores.device), diagonal=1)
    return nn.functional.softplus(-differences[:, upper]).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    exact, pairs, margins, count = 0, 0, [], 0
    for batch in loader:
        rank_count = batch.pop("rank_count")
        scores = model(**{key: value.to(device) for key, value in batch.items()}).reshape(rank_count, 4)
        diffs = scores[:, :, None] - scores[:, None, :]
        upper = torch.triu(torch.ones(4, 4, dtype=torch.bool, device=device), diagonal=1)
        pair_diffs = diffs[:, upper]
        exact += (pair_diffs > 0).all(dim=1).sum().item()
        pairs += (pair_diffs > 0).sum().item()
        margins.extend(pair_diffs.mean(dim=1).cpu().tolist())
        count += rank_count
    return {"rank_accuracy": exact / count, "pair_accuracy": pairs / (count * 6), "mean_pair_margin": float(np.mean(margins)), "ranklists": count}


def save_checkpoint(model, tokenizer, directory):
    directory.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(directory / "reward_model.safetensors"))


def load_checkpoint(directory, device):
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model = RewardModel(directory, from_config=True).to(device)
    model.load_state_dict(load_file(str(directory / "reward_model.safetensors"), device=str(device)))
    model.eval()
    return model, tokenizer


def train(args, device):
    train_rows, valid_rows, test_rows, removed = load_splits(args.seed, args.max_train, args.max_eval, args.train_path, args.dev_path)
    if not train_rows or not valid_rows or not test_rows:
        raise ValueError("Training, validation and test splits must all be nonempty")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=Path(args.model).exists())
    model = RewardModel(args.model).to(device)
    train_loader = make_loader(train_rows, tokenizer, args.batch_size, args.max_length, shuffle=True)
    valid_loader = make_loader(valid_rows, tokenizer, args.batch_size, args.max_length)
    test_loader = make_loader(test_rows, tokenizer, args.batch_size, args.max_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    baseline = evaluate(model, valid_loader, device)
    best_accuracy, best_epoch, stale = -1.0, 0, 0
    history = []
    print(json.dumps({"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows), "removed_overlap_rows": removed, "baseline_valid": baseline}, ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            rank_count = batch.pop("rank_count")
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            scores = model(**inputs).reshape(rank_count, 4)
            loss = ranking_loss(scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(model, valid_loader, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics})
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if metrics["rank_accuracy"] > best_accuracy + 1e-8:
            best_accuracy, best_epoch, stale = metrics["rank_accuracy"], epoch, 0
            save_checkpoint(model, tokenizer, args.output_dir)
        else:
            stale += 1
            if stale >= args.patience:
                break
    best, saved_tokenizer = load_checkpoint(args.output_dir, device)
    result = {
        "base_model": str(args.model),
        "train_path": str(args.train_path.resolve()) if args.train_path else None,
        "dev_path": str(args.dev_path.resolve()) if args.dev_path else None,
        "seed": args.seed,
        "max_length": args.max_length,
        "train_size": len(train_rows),
        "valid_size": len(valid_rows),
        "test_size": len(test_rows),
        "removed_overlap_rows": removed,
        "best_epoch": best_epoch,
        "baseline_valid": baseline,
        "history": history,
        "valid": evaluate(best, make_loader(valid_rows, saved_tokenizer, args.batch_size, args.max_length), device),
        "test": evaluate(best, test_loader, device),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": result["test"]}, ensure_ascii=False), flush=True)


def reevaluate(args, device):
    metadata = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    train_path = Path(metadata["train_path"]) if metadata.get("train_path") else None
    dev_path = Path(metadata["dev_path"]) if metadata.get("dev_path") else None
    _, _, test_rows, _ = load_splits(metadata["seed"], args.max_train, args.max_eval, train_path, dev_path)
    model, tokenizer = load_checkpoint(args.output_dir, device)
    print(json.dumps(evaluate(model, make_loader(test_rows, tokenizer, args.batch_size, metadata["max_length"]), device), ensure_ascii=False))


def score_texts(args, device):
    metadata = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    model, tokenizer = load_checkpoint(args.output_dir, device)
    for start in range(0, len(args.text), args.batch_size):
        texts = args.text[start : start + args.batch_size]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=metadata["max_length"], return_tensors="pt")
        with torch.no_grad():
            scores = model(**{key: value.to(device) for key, value in encoded.items()}).cpu().tolist()
        for text, score in zip(texts, scores):
            print(json.dumps({"text": text, "reward": round(score, 4)}, ensure_ascii=False))


def main():
    if hasattr(sys.stdout, "reconfigure") and (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate", "score"])
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--train-path", type=Path, help="Optional custom rank-list TSV; split internally if --dev-path is absent")
    parser.add_argument("--dev-path", type=Path, help="Optional custom held-out rank-list TSV")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "reward_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train", type=int, default=None, help="Optional smoke-test cap on training ranklists")
    parser.add_argument("--max-eval", type=int, default=None, help="Optional smoke-test cap on dev ranklists")
    parser.add_argument("--text", action="append")
    args = parser.parse_args()
    if args.mode == "score" and not args.text:
        parser.error("score requires --text")
    if args.mode != "score" and args.text:
        parser.error("--text is only valid with score")
    device = torch.device(args.device)
    if args.mode == "train":
        train(args, device)
    elif args.mode == "evaluate":
        reevaluate(args, device)
    else:
        score_texts(args, device)


if __name__ == "__main__":
    main()
