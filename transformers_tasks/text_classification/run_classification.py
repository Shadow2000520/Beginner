"""Train, evaluate and predict with the repository's eight-class BERT classifier."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "comment_classify"
DEFAULT_MODEL = Path(r"E:\Workspace\Beginner\llm-beginner\task-4-rag\models\bge-small-zh-v1.5")


def read_labels(path=DATA / "label_mapping.txt"):
    mapping = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            index, label = line.split(",", 1)
            index = int(index)
        except ValueError as exc:
            raise ValueError(f"Invalid label mapping at {path}:{number}") from exc
        if index in mapping or label in mapping.values():
            raise ValueError(f"Duplicate label mapping at {path}:{number}")
        mapping[index] = label
    if set(mapping) != set(range(len(mapping))):
        raise ValueError("Label IDs must be consecutive from zero")
    return mapping


def read_rows(path, mapping):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            label, content = line.split("\t", 1)
            label = int(label)
        except ValueError as exc:
            raise ValueError(f"Expected INTEGER<TAB>TEXT at {path}:{number}") from exc
        if label not in mapping or not content:
            raise ValueError(f"Unknown class or empty text at {path}:{number}")
        rows.append((label, content))
    return rows


def load_splits(seed=42, valid_fraction=0.2):
    mapping = read_labels()
    source_train = read_rows(DATA / "train.txt", mapping)
    test = read_rows(DATA / "dev.txt", mapping)
    texts = [content for _, content in source_train + test]
    if len(texts) != len(set(texts)):
        raise ValueError("Duplicate text detected across train/dev; split would leak examples")
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1")
    by_label = defaultdict(list)
    for row in source_train:
        by_label[row[0]].append(row)
    rng = random.Random(seed)
    train, valid = [], []
    for label in sorted(mapping):
        rows = by_label[label]
        rng.shuffle(rows)
        n_valid = max(1, round(len(rows) * valid_fraction))
        valid.extend(rows[:n_valid])
        train.extend(rows[n_valid:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid, test, mapping


class ReviewDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_loader(rows, tokenizer, batch_size, max_length, shuffle=False):
    def collate(batch):
        labels, texts = zip(*batch)
        encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return DataLoader(ReviewDataset(rows), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def evaluate(model, loader, device, mapping):
    model.eval()
    gold, pred = [], []
    for batch in loader:
        targets = batch.pop("labels")
        inputs = {key: value.to(device) for key, value in batch.items()}
        predictions = model(**inputs).logits.argmax(dim=-1).cpu().tolist()
        gold.extend(targets.tolist())
        pred.extend(predictions)
    ids = list(mapping)
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=ids, average="macro", zero_division=0),
        "weighted_f1": f1_score(gold, pred, labels=ids, average="weighted", zero_division=0),
        "report": classification_report(gold, pred, labels=ids, target_names=[mapping[i] for i in ids], output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(gold, pred, labels=ids).tolist(),
        "support": dict(Counter(mapping[i] for i in gold)),
    }


def train(args, device):
    train_rows, valid_rows, test_rows, mapping = load_splits(args.seed, args.valid_fraction)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=Path(args.model).exists())
    config = AutoConfig.from_pretrained(args.model, local_files_only=Path(args.model).exists())
    config.id2label = mapping
    config.label2id = {name: index for index, name in mapping.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        local_files_only=Path(args.model).exists(),
        config=config,
    ).to(device)
    train_loader = make_loader(train_rows, tokenizer, args.batch_size, args.max_length, shuffle=True)
    valid_loader = make_loader(valid_rows, tokenizer, args.batch_size, args.max_length)
    test_loader = make_loader(test_rows, tokenizer, args.batch_size, args.max_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    best_score, best_epoch, stale = -1.0, 0, 0
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows), "train_distribution": dict(Counter(mapping[i] for i, _ in train_rows))}, ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(model, valid_loader, device, mapping)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "valid_accuracy": metrics["accuracy"], "valid_macro_f1": metrics["macro_f1"]})
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if metrics["macro_f1"] > best_score + 1e-8:
            best_score, best_epoch, stale = metrics["macro_f1"], epoch, 0
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
        else:
            stale += 1
            if stale >= args.patience:
                break
    best = AutoModelForSequenceClassification.from_pretrained(args.output_dir, local_files_only=True).to(device)
    result = {
        "base_model": str(args.model),
        "seed": args.seed,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_size": len(train_rows),
        "valid_size": len(valid_rows),
        "test_size": len(test_rows),
        "best_epoch": best_epoch,
        "label_mapping": mapping,
        "history": history,
        "valid": evaluate(best, valid_loader, device, mapping),
        "test": evaluate(best, test_loader, device, mapping),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test_accuracy": result["test"]["accuracy"], "test_macro_f1": result["test"]["macro_f1"]}, ensure_ascii=False), flush=True)


def load_checkpoint(args, device):
    meta = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir, local_files_only=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir, local_files_only=True)
    mapping = {int(index): label for index, label in meta["label_mapping"].items()}
    return meta, model, tokenizer, mapping


def predict(args, device):
    meta, model, tokenizer, mapping = load_checkpoint(args, device)
    for start in range(0, len(args.text), args.batch_size):
        texts = args.text[start : start + args.batch_size]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=meta["max_length"], return_tensors="pt")
        with torch.no_grad():
            probabilities = model(**{key: value.to(device) for key, value in encoded.items()}).logits.softmax(dim=-1).cpu()
        for text, scores in zip(texts, probabilities):
            index = int(scores.argmax())
            print(json.dumps({"text": text, "label_id": index, "label": mapping[index], "confidence": round(float(scores[index]), 4)}, ensure_ascii=False))


def reevaluate(args, device):
    meta, model, tokenizer, mapping = load_checkpoint(args, device)
    _, _, test_rows, _ = load_splits(meta["seed"])
    loader = make_loader(test_rows, tokenizer, args.batch_size, meta["max_length"])
    metrics = evaluate(model, loader, device, mapping)
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"]}, ensure_ascii=False))


def main():
    if hasattr(sys.stdout, "reconfigure") and (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate", "predict"])
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text", action="append")
    args = parser.parse_args()
    if args.mode == "predict" and not args.text:
        parser.error("predict requires at least one --text")
    if args.mode != "predict" and args.text:
        parser.error("--text is only valid in predict mode")
    device = torch.device(args.device)
    if args.mode == "train":
        train(args, device)
    elif args.mode == "evaluate":
        reevaluate(args, device)
    else:
        predict(args, device)


if __name__ == "__main__":
    main()
