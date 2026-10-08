"""Reproducible Windows/CPU/CUDA entry point for the four repository models."""
import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import spearmanr
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from supervised.model import PointwiseMatching, DSSM, SentenceTransformer
from unsupervised.simcse.model import SimCSE
from unsupervised.simcse.utils import word_repetition

ROOT = Path(__file__).resolve().parent
METHODS = ('pointwise', 'dssm', 'sentence_bert', 'simcse')
CLASSES = dict(zip(METHODS, (PointwiseMatching, DSSM, SentenceTransformer, SimCSE)))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_pairs(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        parts = line.split('\t')
        if len(parts) != 3 or parts[2] not in ('0', '1') or not all(parts[:2]):
            raise ValueError(f'{path}:{number}: expected text1<TAB>text2<TAB>0/1')
        rows.append((parts[0], parts[1], int(parts[2])))
    return rows


def prepare(output, seed):
    """Keep entire reviews together; exclude evaluation texts from unsupervised training."""
    source = ROOT / 'supervised/data/comment_classify'
    old_train, old_dev = read_pairs(source / 'train.txt'), read_pairs(source / 'dev.txt')
    by_pair = {}
    for a, b, label in old_train + old_dev:
        by_pair.setdefault((a, b), set()).add(label)
    conflicts = sum(len(labels) > 1 for labels in by_pair.values())
    rows = [(a, b, next(iter(labels))) for (a, b), labels in by_pair.items() if len(labels) == 1]
    groups = [r[1] for r in rows]
    train_idx, rest_idx = next(GroupShuffleSplit(n_splits=1, test_size=.3, random_state=seed).split(rows, groups=groups))
    rest = [rows[i] for i in rest_idx]
    dev_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed).split(rest, groups=[r[1] for r in rest]))
    supervised = {'train': [rows[i] for i in train_idx], 'dev': [rest[i] for i in dev_idx], 'test': [rest[i] for i in test_idx]}
    sets = {k: {r[1] for r in v} for k, v in supervised.items()}
    assert not (sets['train'] & sets['dev'] or sets['train'] & sets['test'] or sets['dev'] & sets['test'])
    for split, data in supervised.items():
        if len({r[2] for r in data}) != 2:
            raise ValueError(f'{split} must contain both classes')
    source_lcqmc = ROOT / 'unsupervised/simcse/data/LCQMC'
    original_texts = (source_lcqmc / 'train.txt').read_text(encoding='utf-8').splitlines()
    labeled = read_pairs(source_lcqmc / 'dev.tsv')
    official_test = []
    for number, line in enumerate((source_lcqmc/'test.tsv').read_text(encoding='utf-8').splitlines(), 1):
        pair = line.split('\t')
        if len(pair) != 2 or not all(pair):
            raise ValueError(f'test.tsv:{number}: expected two tab-separated texts')
        official_test.append(pair)
    # Connected components keep a sentence from appearing in both evaluation splits.
    parent = {}
    def find(text):
        parent.setdefault(text, text)
        if parent[text] != text:
            parent[text] = find(parent[text])
        return parent[text]
    for a, b, _ in labeled:
        parent[find(a)] = find(b)
    groups = [find(a) for a, _, _ in labeled]
    dev_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed).split(labeled, groups=groups))
    clean_dev = [labeled[i] for i in dev_idx]
    test = [labeled[i] for i in test_idx]
    assert not ({t for r in clean_dev for t in r[:2]} & {t for r in test for t in r[:2]})
    excluded = {t for r in labeled for t in r[:2]} | {t for r in official_test for t in r}
    texts = list(dict.fromkeys(t.strip() for t in original_texts if t.strip() and t.strip() not in excluded))
    unsupervised = {'train': texts, 'dev': clean_dev, 'test': test}
    audit = {
        'seed': seed,
        'supervised_original': {'train_pairs': len(old_train), 'dev_pairs': len(old_dev),
            'shared_reviews': len({r[1] for r in old_train} & {r[1] for r in old_dev}),
            'conflicting_pairs_removed': conflicts, 'unique_unambiguous_pairs': len(rows)},
        'supervised': {k: {'pairs': len(v), 'reviews': len(sets[k]), 'labels': dict(Counter(r[2] for r in v))} for k, v in supervised.items()},
        'lcqmc': {'original_train_lines': len(original_texts), 'original_unique_train_texts': len(set(original_texts)),
            'train_unique_after_exclusion': len(texts), 'original_dev_pairs': len(labeled), 'dev_pairs': len(clean_dev),
            'heldout_test_pairs': len(test), 'official_unlabeled_test_pairs': len(official_test),
            'train_eval_text_overlap_after_cleaning': len(set(texts) & excluded)},
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [source/'train.txt', source/'dev.txt', source_lcqmc/'train.txt', source_lcqmc/'dev.tsv', source_lcqmc/'test.tsv']},
    }
    for family, splits in [('supervised', supervised), ('simcse', unsupervised)]:
        for split, values in splits.items():
            p = output / 'data' / family / f'{split}.tsv'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('\n'.join(v if isinstance(v, str) else '\t'.join(map(str, v)) for v in values) + '\n', encoding='utf-8')
    write_json(output / 'data_audit.json', audit)
    return supervised, unsupervised, audit


class Collator:
    def __init__(self, tokenizer, method, max_length, training=False):
        self.tokenizer, self.method, self.max_length, self.training = tokenizer, method, max_length, training

    def encode(self, first, second=None):
        out = self.tokenizer(first, text_pair=second, truncation=True, max_length=self.max_length,
                             padding=True, return_tensors='pt')
        out.setdefault('token_type_ids', torch.zeros_like(out['input_ids']))
        return dict(out)

    def __call__(self, rows):
        if self.method == 'simcse' and self.training:
            encoded = self.encode(rows)
            return {'query': encoded, 'doc': {k: v.clone() for k, v in encoded.items()}}
        a, b, labels = zip(*rows)
        if self.method == 'pointwise':
            return {'pair': self.encode(list(a), list(b)), 'labels': torch.tensor(labels)}
        return {'query': self.encode(list(a)), 'doc': self.encode(list(b)), 'labels': torch.tensor(labels)}


def to_device(batch, device):
    return {k: to_device(v, device) if isinstance(v, dict) else v.to(device) for k, v in batch.items()}


def forward(model, method, batch):
    if method == 'pointwise':
        return model(**batch['pair'])
    if method == 'simcse':
        q = model.get_pooled_embedding(**batch['query'])
        d = model.get_pooled_embedding(**batch['doc'])
        return torch.nn.functional.cosine_similarity(q, d)
    arguments = {f'{side}_{key}': value for side in ('query', 'doc') for key, value in batch[side].items()}
    return model.get_similarity(**arguments) if method == 'dssm' else model.get_similarity_label(**arguments)


def autocast(device, enabled):
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16) if enabled and device.type == 'cuda' else nullcontext()


def metrics(labels, scores, threshold):
    predictions = (np.asarray(scores) >= threshold).astype(int)
    corr = float(spearmanr(labels, scores).statistic)
    return {'accuracy': float(accuracy_score(labels, predictions)), 'precision': float(precision_score(labels, predictions, zero_division=0)),
            'recall': float(recall_score(labels, predictions, zero_division=0)), 'f1': float(f1_score(labels, predictions, zero_division=0)),
            'spearman': corr if math.isfinite(corr) else None,
            'confusion_matrix': confusion_matrix(labels, predictions, labels=[0, 1]).tolist(), 'threshold': float(threshold), 'n': len(labels)}


def select_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    return float(thresholds[best[np.argmin(np.abs(thresholds[best] - .5))]])


@torch.inference_mode()
def evaluate(model, method, loader, device, amp, threshold=None):
    model.eval()
    labels, scores = [], []
    for batch in loader:
        labels.extend(batch['labels'].tolist())
        with autocast(device, amp):
            values = forward(model, method, to_device(batch, device)).float()
        if method in ('pointwise', 'sentence_bert'):
            values = values.softmax(-1)[:, 1]
        scores.extend(values.cpu().tolist())
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite evaluation scores')
    if threshold is None:
        threshold = select_threshold(labels, scores) if method in ('dssm', 'simcse') else .5
    return metrics(labels, scores, threshold), labels, scores


def save_checkpoint(path, model, tokenizer, metadata):
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(path)
    tokenizer.save_pretrained(path)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path/'matching.safetensors'))
    write_json(path / 'matching_config.json', metadata)


def load_checkpoint(path, device):
    metadata = json.loads((path/'matching_config.json').read_text(encoding='utf-8'))
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(path, local_files_only=True))
    model = CLASSES[metadata['method']](encoder)
    model.load_state_dict(load_file(str(path/'matching.safetensors')))
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer, metadata


def train_one(args, method, data):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    run = args.output / method
    if (run/'result.json').exists():
        raise FileExistsError(f'{run} already contains a completed run; choose a new --output')
    run.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_only)
    encoder = AutoModel.from_pretrained(args.model, local_files_only=args.local_only)
    model = CLASSES[method](encoder, **({'dropout': .3} if method == 'simcse' else {})).to(device)
    length = args.simcse_max_length if method == 'simcse' else args.max_length
    batch_size = args.simcse_batch_size if method == 'simcse' else args.batch_size
    epochs = args.simcse_epochs if method == 'simcse' else args.epochs
    lr = args.simcse_lr if method == 'simcse' else args.lr
    train_rows = data['train']
    if args.max_train_samples:
        indices = np.random.default_rng(args.seed).permutation(len(train_rows))[:args.max_train_samples]
        train_rows = [train_rows[i] for i in indices]
    train_loader = DataLoader(train_rows, batch_size=batch_size, shuffle=True, num_workers=0,
                              drop_last=method == 'simcse', collate_fn=Collator(tokenizer, method, length, True))
    dev_loader = DataLoader(data['dev'], batch_size=args.eval_batch_size, collate_fn=Collator(tokenizer, method, length))
    test_loader = DataLoader(data['test'], batch_size=args.eval_batch_size, collate_fn=Collator(tokenizer, method, length))
    total_steps = len(train_loader) * epochs
    if total_steps < 1:
        raise ValueError('No training batches; increase data size or decrease batch size')
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if 'bias' in name or 'LayerNorm.weight' in name else decay).append(p)
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': .01}, {'params': no_decay, 'weight_decay': 0}], lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(.06 * total_steps), total_steps)
    config = {'method': method, 'model': str(Path(args.model).resolve()) if Path(args.model).exists() else args.model,
              'seed': args.seed, 'epochs': epochs, 'batch_size': batch_size, 'max_length': length, 'learning_rate': lr,
              'train_samples': len(train_rows), 'total_steps': total_steps, 'device': str(device), 'bf16': args.amp,
              'torch_version': torch.__version__, 'duplicate_ratio': .32 if method == 'simcse' else None}
    write_json(run/'config.json', config)
    start = time.perf_counter()
    baseline, _, _ = evaluate(model, method, dev_loader, device, args.amp)
    print(f'{method} baseline dev: {json.dumps(baseline)}', flush=True)
    best_f1, step, losses, history = -1., 0, [], []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            if method == 'simcse':
                for side in ('query', 'doc'):
                    ids, types = word_repetition(batch[side]['input_ids'], batch[side]['token_type_ids'], dup_rate=.32)
                    # Limit augmentation to positional capacity; infer padding mask after repetition.
                    limit = encoder.config.max_position_embeddings
                    batch[side] = {'input_ids': ids[:, :limit], 'token_type_ids': types[:, :limit],
                                   'attention_mask': ids[:, :limit].ne(tokenizer.pad_token_id)}
            batch = to_device(batch, device)
            with autocast(device, args.amp):
                if method == 'simcse':
                    kwargs = {f'{side}_{key}': value for side in ('query', 'doc') for key, value in batch[side].items()}
                    loss = model(**kwargs, device=str(device))
                else:
                    values = forward(model, method, batch).float()
                    loss = torch.nn.functional.mse_loss(values, batch['labels'].float()) if method == 'dssm' else torch.nn.functional.cross_entropy(values, batch['labels'])
            if not torch.isfinite(loss):
                raise ValueError(f'{method}: non-finite training loss at step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            losses.append(loss.item())
            if step % args.log_steps == 0:
                print(f'{method} epoch={epoch} step={step}/{total_steps} loss={np.mean(losses[-args.log_steps:]):.4f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
            interval = args.simcse_eval_steps if method == 'simcse' else len(train_loader)
            if step % interval == 0 or step == total_steps:
                dev_metrics, _, _ = evaluate(model, method, dev_loader, device, args.amp)
                entry = {'step': step, 'epoch': epoch, 'train_loss': float(np.mean(losses[-interval:])), 'dev': dev_metrics}
                history.append(entry)
                print(f'{method} dev step={step}: {json.dumps(dev_metrics)}', flush=True)
                if dev_metrics['f1'] > best_f1:
                    best_f1 = dev_metrics['f1']
                    save_checkpoint(run/'best', model, tokenizer, {**config, 'threshold': dev_metrics['threshold'], 'best_step': step})
                write_json(run/'history.json', history)
                model.train()
    del optimizer, scheduler, model, encoder
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model, tokenizer, metadata = load_checkpoint(run/'best', device)
    dev_metrics, _, _ = evaluate(model, method, dev_loader, device, args.amp, metadata['threshold'])
    test_metrics, labels, scores = evaluate(model, method, test_loader, device, args.amp, metadata['threshold'])
    predictions = [{'text1': row[0], 'text2': row[1], 'label': label, 'score': score, 'prediction': int(score >= metadata['threshold'])}
                   for row, label, score in zip(data['test'], labels, scores)]
    write_json(run/'test_predictions.json', predictions)
    result = {'config': config, 'baseline_dev': baseline, 'best_step': metadata['best_step'],
              'dev': dev_metrics, 'test': test_metrics, 'elapsed_seconds': time.perf_counter()-start,
              'test_majority_accuracy': max(Counter(labels).values()) / len(labels)}
    write_json(run/'result.json', result)
    export_demo(run, model, tokenizer, metadata, device)
    print(f'COMPLETE {method}: {json.dumps(result)}', flush=True)
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def score_pairs(model, tokenizer, metadata, pairs, device):
    batch = Collator(tokenizer, metadata['method'], metadata['max_length'])([(a, b, 0) for a, b in pairs])
    values = forward(model, metadata['method'], to_device(batch, device)).float()
    if metadata['method'] in ('pointwise', 'sentence_bert'):
        values = values.softmax(-1)[:, 1]
    return values.cpu().tolist()


@torch.inference_mode()
def export_demo(run, model, tokenizer, metadata, device):
    method = metadata['method']
    if method == 'simcse':
        pairs = [('这种图片是用什么软件制作的？', '这种图片制作是用什么软件呢？'),
                 ('今天北京天气怎么样？', '怎么安装手机软件？')]
        scores = score_pairs(model, tokenizer, metadata, pairs, device)
        write_json(run/'demo.json', [{'text1': a, 'text2': b, 'score': s, 'match': s >= metadata['threshold']} for (a, b), s in zip(pairs, scores)])
        return
    candidates = [line.split('\t') for line in (ROOT/'supervised/data/comment_classify/types_desc.txt').read_text(encoding='utf-8').splitlines()]
    texts = [f'{label}：{desc}' for _, label, desc in candidates]
    comments = ['苹果很甜，新鲜又好吃。', '手机通话清晰，电池续航也很好。', '这件衣服尺码合适，穿着很舒服。']
    embeddings = None
    if method in ('dssm', 'sentence_bert'):
        encoded = {k: v.to(device) for k, v in Collator(tokenizer, method, metadata['max_length']).encode(texts).items()}
        embeddings = model(**encoded) if method == 'dssm' else model.get_embedding(**encoded)
        write_json(run/'candidate_embeddings.json', [{'id': i, 'label': label, 'text': text, 'embedding': emb} for (i, label, _), text, emb in zip(candidates, texts, embeddings.cpu().tolist())])
    demos = []
    for comment in comments:
        direct = score_pairs(model, tokenizer, metadata, [(t, comment) for t in texts], device)
        if embeddings is not None:
            encoded = {k: v.to(device) for k, v in Collator(tokenizer, method, metadata['max_length']).encode([comment]).items()}
            if method == 'dssm':
                scores = torch.nn.functional.cosine_similarity(embeddings, model(**encoded)).cpu().tolist()
            else:
                doc = model.get_embedding(**encoded).expand_as(embeddings)
                scores = model.classifier(torch.cat([embeddings, doc, (embeddings-doc).abs()], -1)).softmax(-1)[:, 1].cpu().tolist()
            if not np.allclose(direct, scores, atol=2e-5):
                raise AssertionError('Precomputed embeddings disagree with direct pair inference')
        else:
            scores = direct
        ranked = sorted([{'label': row[1], 'score': s} for row, s in zip(candidates, scores)], key=lambda x: -x['score'])
        demos.append({'comment': comment, 'ranking': ranked})
    write_json(run/'demo.json', demos)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'prepare', 'predict'])
    parser.add_argument('--method', choices=['all', *METHODS], default='all')
    parser.add_argument('--model', default='BAAI/bge-small-zh-v1.5')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/full')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--text1')
    parser.add_argument('--text2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--local-only', action='store_true')
    parser.add_argument('--amp', action='store_true', help='Use CUDA BF16 training and evaluation')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--simcse-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--simcse-batch-size', type=int, default=64)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--simcse-max-length', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--simcse-lr', type=float, default=1e-5)
    parser.add_argument('--simcse-eval-steps', type=int, default=1000)
    parser.add_argument('--log-steps', type=int, default=50)
    parser.add_argument('--max-train-samples', type=int, default=0, help='0 uses the entire cleaned training split')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'predict':
        if not args.checkpoint or not args.text1 or not args.text2:
            parser.error('predict requires --checkpoint, --text1 and --text2')
        model, tokenizer, metadata = load_checkpoint(args.checkpoint, torch.device(args.device))
        score = score_pairs(model, tokenizer, metadata, [(args.text1, args.text2)], torch.device(args.device))[0]
        print(json.dumps({'text1': args.text1, 'text2': args.text2, 'score': score, 'threshold': metadata['threshold'], 'match': score >= metadata['threshold']}, ensure_ascii=False))
        return
    supervised, simcse, audit = prepare(args.output, args.seed)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if args.command == 'prepare':
        return
    results = {}
    for method in METHODS if args.method == 'all' else [args.method]:
        results[method] = train_one(args, method, simcse if method == 'simcse' else supervised)
        write_json(args.output/'summary.json', results)


if __name__ == '__main__':
    main()
