"""Reproducible PET and continuous prompt-tuning experiments on product reviews.

Both methods classify by scoring the two [MASK] positions against the tokenized
verbalizer. The encoder is fine-tuned in both methods; p-tuning additionally
learns 15 continuous prompt embeddings. No validation/test data is trained on.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "PET" / "data" / "comment_classify"
DEFAULT_MODEL = Path(r"E:\Workspace\Beginner\llm-beginner\task-4-rag\models\bge-small-zh-v1.5")


def read_rows(path: Path) -> list[tuple[str, str]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            label, content = line.split("\t", 1)
        except ValueError as exc:
            raise ValueError(f"{path}:{number} must be LABEL<TAB>TEXT") from exc
        if not label or not content:
            raise ValueError(f"{path}:{number} has an empty field")
        rows.append((label, content))
    return rows


def load_splits(seed: int = 42):
    train = read_rows(DATA / "train.txt")
    heldout = read_rows(DATA / "dev.txt")
    if len({text for _, text in train}) != len(train) or len({text for _, text in heldout}) != len(heldout):
        raise ValueError("Duplicate review text found within a split")
    if {text for _, text in train} & {text for _, text in heldout}:
        raise ValueError("Training text overlaps held-out text")
    labels = [line.split("\t", 1)[0] for line in (DATA / "verbalizer.txt").read_text(encoding="utf-8").splitlines()]
    labels += sorted(set(label for label, _ in train + heldout) - set(labels))
    by_label = defaultdict(list)
    for row in heldout:
        by_label[row[0]].append(row)
    rng = random.Random(seed)
    valid, test = [], []
    for label in sorted(by_label):
        rows = by_label[label]
        rng.shuffle(rows)
        midpoint = len(rows) // 2
        valid.extend(rows[:midpoint])
        test.extend(rows[midpoint:])
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test, labels


class ReviewDataset(Dataset):
    def __init__(self, rows, tokenizer, labels, method, max_length, prompt_length):
        self.items = []
        label_to_id = {label: i for i, label in enumerate(labels)}
        if method == "pet":
            pattern = (DATA / "prompt.txt").read_text(encoding="utf-8").strip()
            if pattern != "这是一条{MASK}评论：{textA}。":
                raise ValueError("This runner expects the repository's supplied PET pattern")
            prefix = tokenizer.encode("这是一条", add_special_tokens=False)
            middle = tokenizer.encode("评论：", add_special_tokens=False)
            suffix = tokenizer.encode("。", add_special_tokens=False)
        else:
            prefix, middle, suffix = [], [], []
        for label, content in rows:
            if method == "pet":
                start = [tokenizer.cls_token_id] + prefix
                mask_positions = [len(start), len(start) + 1]
                fixed = start + [tokenizer.mask_token_id] * 2 + middle
            else:
                start = [tokenizer.cls_token_id] + [tokenizer.pad_token_id] * prompt_length
                mask_positions = [len(start), len(start) + 1]
                fixed = start + [tokenizer.mask_token_id] * 2
            text_ids = tokenizer.encode(content, add_special_tokens=False, truncation=True, max_length=max_length)
            budget = max_length - len(fixed) - len(suffix) - 1
            if budget < 1:
                raise ValueError("max_length is too short for the prompt")
            ids = fixed + text_ids[:budget] + suffix + [tokenizer.sep_token_id]
            self.items.append((ids, mask_positions, label_to_id[label]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def make_collate(pad_id):
    def collate(batch):
        width = max(len(ids) for ids, _, _ in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, (ids, _, _) in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attention_mask[i, : len(ids)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mask_positions": torch.tensor([positions for _, positions, _ in batch]),
            "labels": torch.tensor([label for _, _, label in batch]),
        }

    return collate


class PromptClassifier(nn.Module):
    def __init__(self, base_model, label_token_ids, method, prompt_length=15, from_config=False, scoring=None):
        super().__init__()
        local = Path(base_model).exists()
        config = AutoConfig.from_pretrained(base_model, local_files_only=local)
        self.scoring = scoring or ("mlm" if any("ForMaskedLM" in name for name in (config.architectures or [])) else "embedding")
        model_class = AutoModelForMaskedLM if self.scoring == "mlm" else AutoModel
        if from_config:
            self.encoder = model_class.from_config(config)
        else:
            self.encoder = model_class.from_pretrained(base_model, local_files_only=local)
        self.method = method
        self.prompt_length = prompt_length if method == "ptuning" else 0
        self.register_buffer("label_token_ids", torch.tensor(label_token_ids, dtype=torch.long))
        if self.prompt_length:
            hidden = self.encoder.config.hidden_size
            self.soft_prompt = nn.Parameter(torch.empty(self.prompt_length, hidden))
            nn.init.normal_(self.soft_prompt, std=0.02)

    def forward(self, input_ids, attention_mask, mask_positions):
        if self.prompt_length:
            embeds = self.encoder.get_input_embeddings()(input_ids)
            prompt = self.soft_prompt.unsqueeze(0).expand(input_ids.size(0), -1, -1)
            embeds = torch.cat((embeds[:, :1], prompt, embeds[:, 1 + self.prompt_length :]), dim=1)
            outputs = self.encoder(inputs_embeds=embeds, attention_mask=attention_mask)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        index = torch.arange(input_ids.size(0), device=input_ids.device)[:, None]
        if self.scoring == "mlm":
            mask_logits = outputs.logits[index, mask_positions]
            log_probs = mask_logits.log_softmax(dim=-1)
            label_ids = self.label_token_ids.unsqueeze(0).expand(input_ids.size(0), -1, -1)
            return log_probs.unsqueeze(1).expand(-1, label_ids.size(1), -1, -1).gather(-1, label_ids.unsqueeze(-1)).squeeze(-1).mean(dim=-1)
        hidden = outputs.last_hidden_state[index, mask_positions]
        word_embeddings = self.encoder.get_input_embeddings().weight[self.label_token_ids]
        hidden = nn.functional.normalize(hidden, dim=-1)
        word_embeddings = nn.functional.normalize(word_embeddings, dim=-1)
        return torch.einsum("bmh,cmh->bc", hidden, word_embeddings) * 10.0 / self.label_token_ids.size(1)


def encode_labels(tokenizer, labels):
    encoded = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(ids) != 2 or tokenizer.unk_token_id in ids for ids in encoded):
        raise ValueError("Every label must tokenize to two known tokens; adjust verbalizer or tokenizer")
    return encoded


@torch.no_grad()
def evaluate(model, loader, device, labels):
    model.eval()
    gold, pred = [], []
    for batch in loader:
        inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        logits = model(**inputs)
        gold.extend(batch["labels"].tolist())
        pred.extend(logits.argmax(dim=-1).cpu().tolist())
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=list(range(len(labels))), average="macro", zero_division=0),
        "weighted_f1": f1_score(gold, pred, labels=list(range(len(labels))), average="weighted", zero_division=0),
        "report": classification_report(gold, pred, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0),
        "support": dict(Counter(labels[i] for i in gold)),
        "predicted": dict(Counter(labels[i] for i in pred)),
    }


def create_loader(rows, tokenizer, labels, method, args, shuffle=False):
    dataset = ReviewDataset(rows, tokenizer, labels, method, args.max_length, args.prompt_length)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=make_collate(tokenizer.pad_token_id))


def train_one(method, args, splits, tokenizer, device):
    train, valid, test, labels = splits
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = PromptClassifier(args.model, encode_labels(tokenizer, labels), method, args.prompt_length).to(device)
    train_loader = create_loader(train, tokenizer, labels, method, args, shuffle=True)
    valid_loader = create_loader(valid, tokenizer, labels, method, args)
    test_loader = create_loader(test, tokenizer, labels, method, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    best_f1, best_epoch, stale = -1.0, 0, 0
    save_dir = args.output_dir / method
    save_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            targets = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(model, valid_loader, device, labels)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "valid_accuracy": metrics["accuracy"], "valid_macro_f1": metrics["macro_f1"]})
        print(json.dumps({"method": method, **history[-1]}, ensure_ascii=False), flush=True)
        if metrics["macro_f1"] > best_f1 + 1e-8:
            best_f1, best_epoch, stale = metrics["macro_f1"], epoch, 0
            save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(save_dir / "model.safetensors"))
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(load_file(str(save_dir / "model.safetensors"), device=str(device)))
    metadata = {
        "method": method,
        "base_model": str(args.model),
        "scoring": model.scoring,
        "labels": labels,
        "prompt_length": model.prompt_length,
        "max_length": args.max_length,
        "seed": args.seed,
        "train_size": len(train),
        "valid_size": len(valid),
        "test_size": len(test),
        "best_epoch": best_epoch,
        "history": history,
        "valid": evaluate(model, valid_loader, device, labels),
        "test": evaluate(model, test_loader, device, labels),
    }
    (save_dir / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"method": method, "best_epoch": best_epoch, "test_accuracy": metadata["test"]["accuracy"], "test_macro_f1": metadata["test"]["macro_f1"]}, ensure_ascii=False), flush=True)
    return metadata


def predict_one(method, args, tokenizer, device):
    save_dir = args.output_dir / method
    metadata = json.loads((save_dir / "metrics.json").read_text(encoding="utf-8"))
    labels = metadata["labels"]
    model = PromptClassifier(save_dir, encode_labels(tokenizer, labels), method, metadata["prompt_length"], from_config=True, scoring=metadata["scoring"]).to(device)
    model.load_state_dict(load_file(str(save_dir / "model.safetensors"), device=str(device)))
    dataset = ReviewDataset([("__inference__", text) for text in args.text], tokenizer, ["__inference__"], method, metadata["max_length"], metadata["prompt_length"])
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=make_collate(tokenizer.pad_token_id))
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            scores = model(**inputs).softmax(dim=-1).cpu()
            for row in scores:
                index = int(row.argmax())
                predictions.append({"label": labels[index], "confidence": round(float(row[index]), 4)})
    for text, result in zip(args.text, predictions):
        print(json.dumps({"method": method, "text": text, **result}, ensure_ascii=False))


def main():
    if hasattr(sys.stdout, "reconfigure") and (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--method", choices=["pet", "ptuning", "both"], default="both")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "full")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--prompt-length", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text", action="append", help="Review to classify; may be repeated")
    args = parser.parse_args()
    if args.mode == "predict" and not args.text:
        parser.error("predict requires at least one --text")
    if args.mode == "train" and args.text:
        parser.error("--text is only supported in predict mode")
    device = torch.device(args.device)
    methods = ["pet", "ptuning"] if args.method == "both" else [args.method]
    if args.mode == "train":
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=Path(args.model).exists())
        splits = load_splits(args.seed)
        print(json.dumps({"train": len(splits[0]), "valid": len(splits[1]), "test": len(splits[2]), "labels": splits[3]}, ensure_ascii=False), flush=True)
        for method in methods:
            train_one(method, args, splits, tokenizer, device)
    else:
        for method in methods:
            metadata = json.loads((args.output_dir / method / "metrics.json").read_text(encoding="utf-8"))
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir / method, local_files_only=True)
            predict_one(method, args, tokenizer, device)


if __name__ == "__main__":
    main()
