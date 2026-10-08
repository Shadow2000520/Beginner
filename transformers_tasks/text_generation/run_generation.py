"""Reproducible Chinese T5 training for DuReader QA and span filling."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "answer_generation" / "models" / "t5-small-chinese-cluecorpussmall"


def parse_rows(task):
    if task == "qa":
        root = ROOT / "answer_generation" / "data" / "DuReaderQG"
        def parse(path):
            rows, invalid = [], 0
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                item = json.loads(line)
                if not all(item.get(key) for key in ("context", "question", "answer")):
                    invalid += 1
                    continue
                rows.append({"key": item["context"], "source": f"问题：{item['question']}[SEP]原文：{item['context']}", "target": f"答案：{item['answer']}", "reference": item["answer"]})
            return rows, invalid
        train, bad_train = parse(root / "train.json")
        dev, bad_dev = parse(root / "dev.json")
    else:
        root = ROOT / "data_augment" / "filling_model" / "data"
        def parse(path):
            rows, invalid = [], 0
            for line in path.read_text(encoding="utf-8").splitlines():
                if "\t" not in line:
                    invalid += 1
                    continue
                source, target = line.split("\t", 1)
                if not source or not target or source.count("[MASK]") != 2:
                    invalid += 1
                    continue
                rows.append({"key": source, "source": source, "target": target, "reference": target})
            return rows, invalid
        train, bad_train = parse(root / "train.tsv")
        dev, bad_dev = parse(root / "dev.tsv")
    return train, dev, bad_train + bad_dev


def load_splits(task, seed=42, max_train=None, max_valid=256, max_test=512):
    train, dev, invalid = parse_rows(task)
    original_counts = {"raw_train": len(train), "raw_dev": len(dev), "invalid_rows": invalid}
    train_keys = {row["key"] for row in train}
    filtered = [row for row in dev if row["key"] not in train_keys]
    removed_overlap = len(dev) - len(filtered)
    groups = {}
    for row in filtered:
        groups.setdefault(row["key"], []).append(row)
    rng = random.Random(seed)
    group_items = list(groups.values())
    rng.shuffle(group_items)
    valid, test = [], []
    halfway = len(filtered) / 2
    for group in group_items:
        (valid if len(valid) < halfway else test).extend(group)
    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    if max_train is not None and max_train > 0:
        train = train[:max_train]
    if max_valid is not None and max_valid > 0:
        valid = valid[:max_valid]
    if max_test is not None and max_test > 0:
        test = test[:max_test]
    if not train or not valid or not test:
        raise ValueError("All train/validation/test splits must be nonempty")
    original_counts["removed_dev_overlap"] = removed_overlap
    original_counts["filtered_dev"] = len(filtered)
    return train, valid, test, original_counts


def configure_tokens(model, tokenizer):
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("This UER T5 checkpoint requires a BERT tokenizer with CLS, SEP and PAD tokens")
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
    model.generation_config.eos_token_id = tokenizer.sep_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id


def make_loader(rows, tokenizer, batch_size, source_length, target_length, shuffle=False):
    def collate(batch):
        sources = [row["source"] for row in batch]
        encoded = tokenizer(sources, padding=True, truncation=True, max_length=source_length, return_tensors="pt")
        encoded.pop("token_type_ids", None)
        target_ids = [tokenizer.encode(row["target"], add_special_tokens=False, truncation=True, max_length=target_length - 1) + [tokenizer.sep_token_id] for row in batch]
        labels = torch.full((len(batch), max(map(len, target_ids))), -100, dtype=torch.long)
        for index, ids in enumerate(target_ids):
            labels[index, : len(ids)] = torch.tensor(ids)
        encoded["labels"] = labels
        return encoded, [row["reference"] for row in batch], sources

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def normalize(text, task):
    text = re.sub(r"\s+", "", text)
    if task == "qa" and text.startswith("答案："):
        text = text[3:]
    return text


def char_f1(prediction, reference):
    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0
    overlap = sum((Counter(prediction) & Counter(reference)).values())
    precision = overlap / len(prediction)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def char_bleu4(predictions, references):
    matches = [0] * 4
    possible = [0] * 4
    prediction_length = reference_length = 0
    for prediction, reference in zip(predictions, references):
        prediction_length += len(prediction)
        reference_length += len(reference)
        for n in range(1, 5):
            pred_ngrams = Counter(prediction[i : i + n] for i in range(max(0, len(prediction) - n + 1)))
            ref_ngrams = Counter(reference[i : i + n] for i in range(max(0, len(reference) - n + 1)))
            matches[n - 1] += sum((pred_ngrams & ref_ngrams).values())
            possible[n - 1] += sum(pred_ngrams.values())
    if prediction_length == 0:
        return 0.0
    brevity = min(0.0, 1 - reference_length / prediction_length)
    return math.exp(brevity + sum(math.log((matches[i] + 1) / (possible[i] + 1)) for i in range(4)) / 4)


@torch.no_grad()
def evaluate(model, tokenizer, loader, task, device, generation_length, sample_count=10):
    model.eval()
    predictions, references, samples = [], [], []
    for encoded, batch_references, sources in loader:
        inputs = {key: value.to(device) for key, value in encoded.items() if key != "labels"}
        outputs = model.generate(**inputs, max_new_tokens=generation_length, do_sample=False, eos_token_id=tokenizer.sep_token_id, decoder_start_token_id=tokenizer.cls_token_id)
        batch_predictions = [normalize(text, task) for text in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
        batch_references = [normalize(text, task) for text in batch_references]
        predictions.extend(batch_predictions)
        references.extend(batch_references)
        for source, prediction, reference in zip(sources, batch_predictions, batch_references):
            if len(samples) < sample_count:
                samples.append({"source": source, "prediction": prediction, "reference": reference})
    return {
        "exact_match": sum(pred == ref for pred, ref in zip(predictions, references)) / len(references),
        "char_f1": float(np.mean([char_f1(pred, ref) for pred, ref in zip(predictions, references)])),
        "char_bleu4_add1": char_bleu4(predictions, references),
        "count": len(references),
        "samples": samples,
    }


def train(args, device):
    train_rows, valid_rows, test_rows, counts = load_splits(args.task, args.seed, args.max_train, args.max_valid, args.max_test)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=Path(args.model).exists())
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, local_files_only=Path(args.model).exists()).to(device)
    configure_tokens(model, tokenizer)
    train_loader = make_loader(train_rows, tokenizer, args.batch_size, args.max_source_length, args.max_target_length, shuffle=True)
    valid_loader = make_loader(valid_rows, tokenizer, args.eval_batch_size, args.max_source_length, args.max_target_length)
    test_loader = make_loader(test_rows, tokenizer, args.eval_batch_size, args.max_source_length, args.max_target_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    history = []
    best_f1, best_epoch, stale = -1.0, 0, 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"task": args.task, "train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows), **counts}, ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for encoded, _, _ in train_loader:
            inputs = {key: value.to(device) for key, value in encoded.items()}
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**inputs).loss
            else:
                loss = model(**inputs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        valid_metrics = evaluate(model, tokenizer, valid_loader, args.task, device, args.generation_length)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_exact_match": valid_metrics["exact_match"], "valid_char_f1": valid_metrics["char_f1"], "valid_char_bleu4": valid_metrics["char_bleu4_add1"]})
        print(json.dumps({"task": args.task, **history[-1]}, ensure_ascii=False), flush=True)
        if valid_metrics["char_f1"] > best_f1 + 1e-8:
            best_f1, best_epoch, stale = valid_metrics["char_f1"], epoch, 0
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
        else:
            stale += 1
            if stale >= args.patience:
                break
    model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir, local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir, local_files_only=True)
    configure_tokens(model, tokenizer)
    metadata = {
        "task": args.task,
        "base_model": str(args.model),
        "seed": args.seed,
        "max_train": args.max_train,
        "max_valid": args.max_valid,
        "max_test": args.max_test,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "generation_length": args.generation_length,
        "train_size": len(train_rows),
        "valid_size": len(valid_rows),
        "test_size": len(test_rows),
        "data_counts": counts,
        "best_epoch": best_epoch,
        "history": history,
        "valid": evaluate(model, tokenizer, valid_loader, args.task, device, args.generation_length),
        "test": evaluate(model, tokenizer, test_loader, args.task, device, args.generation_length),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"task": args.task, "best_epoch": best_epoch, "test_exact_match": metadata["test"]["exact_match"], "test_char_f1": metadata["test"]["char_f1"], "test_char_bleu4": metadata["test"]["char_bleu4_add1"]}, ensure_ascii=False), flush=True)


def load_checkpoint(directory, device):
    metadata = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(directory, local_files_only=True).to(device)
    configure_tokens(model, tokenizer)
    return metadata, model, tokenizer


def reevaluate(args, device):
    metadata, model, tokenizer = load_checkpoint(args.output_dir, device)
    if metadata["task"] != args.task:
        raise ValueError(f"Checkpoint task is {metadata['task']!r}, but --task is {args.task!r}")
    _, _, test_rows, _ = load_splits(metadata["task"], metadata["seed"], metadata["max_train"], metadata["max_valid"], metadata["max_test"])
    loader = make_loader(test_rows, tokenizer, args.eval_batch_size, metadata["max_source_length"], metadata["max_target_length"])
    metrics = evaluate(model, tokenizer, loader, metadata["task"], device, metadata["generation_length"])
    print(json.dumps({key: value for key, value in metrics.items() if key != "samples"}, ensure_ascii=False))


def predict(args, device):
    metadata, model, tokenizer = load_checkpoint(args.output_dir, device)
    if metadata["task"] != args.task:
        raise ValueError(f"Checkpoint task is {metadata['task']!r}, but --task is {args.task!r}")
    task = metadata["task"]
    if task == "qa":
        if not args.question or not args.context:
            raise ValueError("QA prediction needs --question and --context")
        sources = [f"问题：{args.question}[SEP]原文：{args.context}"]
    else:
        if not args.masked_text:
            raise ValueError("Filling prediction needs at least one --masked-text")
        sources = args.masked_text
    model.eval()
    encoded = tokenizer(sources, padding=True, truncation=True, max_length=metadata["max_source_length"], return_tensors="pt")
    encoded.pop("token_type_ids", None)
    with torch.no_grad():
        outputs = model.generate(**{key: value.to(device) for key, value in encoded.items()}, max_new_tokens=metadata["generation_length"], do_sample=False, eos_token_id=tokenizer.sep_token_id, decoder_start_token_id=tokenizer.cls_token_id)
    for source, output in zip(sources, outputs):
        prediction = normalize(tokenizer.decode(output, skip_special_tokens=True), task)
        print(json.dumps({"task": task, "source": source, "prediction": prediction}, ensure_ascii=False))


def main():
    if hasattr(sys.stdout, "reconfigure") and (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate", "predict"])
    parser.add_argument("--task", choices=["qa", "filling"], required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-source-length", type=int)
    parser.add_argument("--max-target-length", type=int)
    parser.add_argument("--generation-length", type=int)
    parser.add_argument("--max-train", type=int, default=None, help="Optional deterministic training subset; omit or use 0 for all data")
    parser.add_argument("--max-valid", type=int, default=256)
    parser.add_argument("--max-test", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--question")
    parser.add_argument("--context")
    parser.add_argument("--masked-text", action="append")
    args = parser.parse_args()
    args.output_dir = args.output_dir or ROOT / "text_generation" / "runs" / args.task
    args.max_source_length = args.max_source_length or (256 if args.task == "qa" else 128)
    args.max_target_length = args.max_target_length or (32 if args.task == "qa" else 48)
    args.generation_length = args.generation_length or args.max_target_length
    if args.mode != "predict" and (args.question or args.context or args.masked_text):
        parser.error("Text inputs are only supported in predict mode")
    device = torch.device(args.device)
    if args.mode == "train":
        train(args, device)
    elif args.mode == "evaluate":
        reevaluate(args, device)
    else:
        predict(args, device)


if __name__ == "__main__":
    main()
