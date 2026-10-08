import argparse
import copy
from functools import partial
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.checkpoint import load_adapter, save_adapter
from src.dpo import load_preferences, collate_preferences, response_log_probs, dpo_loss
from train_sft import amp_context

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description='从 SFT LoRA 接续训练 DPO')
    p.add_argument('--sft', type=Path, default=ROOT / 'ckpt/sft')
    p.add_argument('--data', type=Path, nargs='+', default=[ROOT / 'data/dpo/dpo_zh.json'])
    p.add_argument('--output-dir', type=Path, default=ROOT / 'ckpt/dpo')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--train-samples', type=int, default=1000)
    p.add_argument('--dev-samples', type=int, default=128)
    p.add_argument('--max-length', type=int, default=768)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--beta', type=float, default=.1)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup-ratio', type=float, default=.05)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if min(a.train_samples, a.dev_samples, a.max_length, a.batch_size, a.grad_accum, a.epochs, a.patience) <= 0:
        p.error('样本数、长度、batch、梯度累积和轮数必须为正数')
    if not 0 < a.lr < math.inf or not 0 < a.beta < math.inf or not 0 <= a.weight_decay < math.inf or not 0 <= a.warmup_ratio < 1:
        p.error('学习率、beta、weight-decay 或 warmup-ratio 不合法')
    output, source = a.output_dir.resolve(), a.sft.resolve()
    if output == source or output in source.parents or source in output.parents:
        p.error('DPO 输出目录必须与 SFT 目录分开，且不能相互包含')
    return a


def batch_objective(policy, reference, batch, device, beta):
    batch = {k: {name: v.to(device) for name, v in values.items()} for k, values in batch.items()}
    with torch.no_grad(), amp_context(device):
        rc = response_log_probs(reference, batch['chosen'])
        rr = response_log_probs(reference, batch['rejected'])
    with amp_context(device):
        pc = response_log_probs(policy, batch['chosen'])
        pr = response_log_probs(policy, batch['rejected'])
    return dpo_loss(pc, pr, rc, rr, beta)


@torch.no_grad()
def evaluate(policy, reference, loader, device, beta):
    policy.eval()
    reference.eval()
    sums = torch.zeros(3, device=device)
    count = 0
    for batch in loader:
        loss, margin, _, _ = batch_objective(policy, reference, batch, device, beta)
        sums += torch.stack([loss.sum(), margin.sum(), (margin > 0).float().sum()])
        count += loss.numel()
    values = (sums / count).tolist()
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError('验证指标出现非有限值')
    return dict(loss=values[0], reward_margin=values[1], preference_accuracy=values[2])


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('需要支持 BF16 的 CUDA GPU，或使用 --device cpu')
    config = json.loads((args.sft / 'adapter_config.json').read_text(encoding='utf-8'))
    model_path = Path(config['base_model'])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print('读取并编码 DPO 偏好对……', flush=True)
    train, dev, stats = load_preferences(args.data, tokenizer, args.max_length,
                                         args.train_samples, args.dev_samples, args.seed)
    collate = partial(collate_preferences, pad_id=tokenizer.pad_token_id)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))
    dev_loader = DataLoader(dev, batch_size=args.batch_size, collate_fn=collate)
    policy = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.float32)
    load_adapter(policy, args.sft)
    reference = copy.deepcopy(policy).to(device).eval().requires_grad_(False)
    policy = policy.to(device)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    baseline = evaluate(policy, reference, dev_loader, device, args.beta)
    if abs(baseline['loss'] - math.log(2)) > 1e-3 or abs(baseline['reward_margin']) > 1e-3:
        raise RuntimeError(f'初始 policy/reference 不一致：{baseline}')
    print(f'train={len(train)} dev={len(dev)} initial_loss={baseline["loss"]:.4f} initial_margin={baseline["reward_margin"]:.4f}', flush=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ['adapter.pt', 'adapter_config.json', 'history.json', 'metadata.json', 'comparison.json']:
        (output / name).unlink(missing_ok=True)
    metadata = dict(args={k: [str(p) for p in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    data=stats, reference='冻结的初始 SFT 模型', initial=baseline)
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    best, stale, step, history = math.inf, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        policy.train()
        reference.eval()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_margin, count, group_count = 0., 0., 0, 0
        for index, batch in enumerate(loader, 1):
            loss, margin, _, _ = batch_objective(policy, reference, batch, device, args.beta)
            if not torch.isfinite(loss).all():
                raise RuntimeError('DPO loss 非有限值')
            loss.sum().backward()
            total_loss += loss.sum().item()
            total_margin += margin.sum().item()
            count += loss.numel()
            group_count += loss.numel()
            if index % args.grad_accum == 0 or index == len(loader):
                for p in parameters:
                    if p.grad is not None:
                        p.grad.div_(group_count)
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                scale = (step+1)/warmup if step < warmup else .1 + .45*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup-1)))
                for group in optimizer.param_groups:
                    group['lr'] = args.lr * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                group_count = 0
                step += 1
                if step % 20 == 0:
                    print(f'epoch={epoch} step={step}/{total_steps} loss={total_loss/count:.4f} margin={total_margin/count:.4f}', flush=True)
        metrics = evaluate(policy, reference, dev_loader, device, args.beta)
        history.append(dict(epoch=epoch, step=step, train_loss=total_loss/count, train_reward_margin=total_margin/count,
                            dev_loss=metrics['loss'], dev_reward_margin=metrics['reward_margin'],
                            dev_preference_accuracy=metrics['preference_accuracy'], lr=optimizer.param_groups[0]['lr']))
        if metrics['loss'] < best:
            best, stale = metrics['loss'], 0
            saved_config = {k:config[k] for k in ['base_model', 'target_modules', 'r', 'alpha']}
            save_adapter(policy, output, dict(saved_config, stage='dpo', source_sft=str(args.sft.resolve()), epoch=epoch, beta=args.beta, dev_loss=best))
        else:
            stale += 1
        (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f'epoch={epoch} train_loss={total_loss/count:.4f} dev_loss={metrics["loss"]:.4f} dev_margin={metrics["reward_margin"]:.4f} dev_pref_acc={metrics["preference_accuracy"]:.3f}', flush=True)
        if stale >= args.patience:
            break
    print(f'DPO 结束，best_dev_loss={best:.4f}；权重：{output}', flush=True)


if __name__ == '__main__':
    main()
