import argparse
from contextlib import nullcontext
from functools import partial
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import collate_messages
from src.checkpoint import save_adapter
from src.data import load_moss
from src.lora import inject_lora

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description='手写 LoRA 的 MOSS SFT')
    p.add_argument('--model', type=Path, default=ROOT / 'models/Qwen2.5-0.5B')
    p.add_argument('--data', type=Path, default=ROOT / 'data/moss-sft/moss-003-sft-no-tools.jsonl')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'ckpt/sft')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--train-samples', type=int, default=2000)
    p.add_argument('--dev-samples', type=int, default=128)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=16)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup-ratio', type=float, default=.05)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--r', type=int, default=8)
    p.add_argument('--alpha', type=float, default=16)
    p.add_argument('--target-modules', nargs='+', default=['q_proj', 'v_proj'])
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.train_samples, args.dev_samples, args.max_length, args.batch_size,
           args.grad_accum, args.epochs, args.patience, args.r) <= 0:
        p.error('样本数、长度、batch、累计步数、轮数、rank 均须为正数')
    if not 0 < args.lr < math.inf or not 0 <= args.weight_decay < math.inf or not 0 <= args.warmup_ratio < 1:
        p.error('学习率、weight-decay 或 warmup-ratio 不合法')
    return args


def amp_context(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, count = 0., 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        n = (batch['labels'][:, 1:] != -100).sum().item()
        with amp_context(device):
            loss = model(**batch, use_cache=False).loss
        total += loss.item() * n
        count += n
    if not count or not math.isfinite(total):
        raise RuntimeError('验证 loss 无效或没有监督 token')
    return total / count


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('需要支持 BF16 的 CUDA GPU，或改用 --device cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    print('逐行读取并编码 MOSS 子集……', flush=True)
    train, dev, stats = load_moss(args.data, tokenizer, args.max_length,
                                  args.train_samples, args.dev_samples, args.seed)
    collate = partial(collate_messages, pad_id=tokenizer.pad_token_id)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))
    dev_loader = DataLoader(dev, batch_size=args.batch_size, collate_fn=collate)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float32).to(device)
    inject_lora(model, args.target_modules, args.r, args.alpha)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ['adapter.pt', 'adapter_config.json', 'history.json', 'metadata.json']:
        (output / name).unlink(missing_ok=True)
    config = dict(base_model=str(args.model.resolve()), target_modules=args.target_modules, r=args.r, alpha=args.alpha)
    metadata = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, data=stats,
                    trainable=sum(p.numel() for p in parameters), split='首批有效对话去重后按 seed 打乱，按整段对话划分')
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    baseline = evaluate(model, dev_loader, device)
    print(f'train={len(train)} dev={len(dev)} trainable={metadata["trainable"]} base_dev_loss={baseline:.4f}', flush=True)
    best, stale, step, history = math.inf, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_tokens, group_tokens = 0., 0, 0
        for index, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            count = (batch['labels'][:, 1:] != -100).sum().item()
            with amp_context(device):
                loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError('训练 loss 非有限值')
            (loss * count).backward()
            total_loss += loss.item() * count
            total_tokens += count
            group_tokens += count
            if index % args.grad_accum == 0 or index == len(loader):
                for p in parameters:
                    if p.grad is not None:
                        p.grad.div_(group_tokens)
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                scale = (step+1)/warmup if step < warmup else .1 + .9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup-1)))
                for group in optimizer.param_groups:
                    group['lr'] = args.lr * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                group_tokens = 0
                step += 1
                if step % 20 == 0:
                    print(f'epoch={epoch} step={step}/{total_steps} train_loss={total_loss/total_tokens:.4f}', flush=True)
        dev_loss = evaluate(model, dev_loader, device)
        history.append(dict(epoch=epoch, step=step, train_loss=total_loss/total_tokens, dev_loss=dev_loss,
                            base_dev_loss=baseline, lr=optimizer.param_groups[0]['lr']))
        if dev_loss < best:
            best, stale = dev_loss, 0
            save_adapter(model, output, dict(config, epoch=epoch, dev_loss=dev_loss))
        else:
            stale += 1
        (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f'epoch={epoch} train_loss={total_loss/total_tokens:.4f} dev_loss={dev_loss:.4f} best={best:.4f}', flush=True)
        if stale >= args.patience:
            break
    print(f'SFT 结束，best_dev_loss={best:.4f}；权重：{output}', flush=True)


if __name__ == '__main__':
    main()
