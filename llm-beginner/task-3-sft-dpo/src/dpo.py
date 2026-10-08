import hashlib
import json
import math
import random

import torch
import torch.nn.functional as F

from .chat import encode_messages, collate_messages, format_messages


def sequence_log_probs(logits, labels):
    if logits.shape[:2] != labels.shape:
        raise ValueError('logits 和 labels 的 batch/length 不一致')
    targets = labels[:, 1:]
    if not (targets != -100).any(dim=1).all():
        raise ValueError('每条回答至少需要一个可预测的监督 token')
    losses = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                             targets.reshape(-1), ignore_index=-100, reduction='none')
    return -losses.reshape(targets.shape).sum(dim=-1)


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=.1):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('beta 必须是有限正数')
    if not (chosen.shape == rejected.shape == ref_chosen.shape == ref_rejected.shape):
        raise ValueError('四组 log probability 的形状必须相同')
    chosen_reward = beta * (chosen - ref_chosen.detach())
    rejected_reward = beta * (rejected - ref_rejected.detach())
    margin = chosen_reward - rejected_reward
    return -F.logsigmoid(margin), margin.detach(), chosen_reward.detach(), rejected_reward.detach()


def preference_messages(record):
    roles = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
    messages = [dict(role=roles[m['from']], content=m['value']) for m in record['conversations']]
    if not messages or messages[-1]['role'] != 'user':
        raise ValueError('偏好对的上下文必须以 user 结尾')
    answers = []
    for key in ('chosen', 'rejected'):
        answer = record[key]
        if answer['from'] != 'gpt' or not isinstance(answer['value'], str) or not answer['value'].strip():
            raise ValueError('chosen/rejected 必须是非空 assistant 回答')
        answers.append(dict(role='assistant', content=answer['value']))
    return messages, answers


def encode_pair(record, tokenizer, max_length):
    messages, answers = preference_messages(record)
    pair = [encode_messages(messages + [a], tokenizer, max_length, last_assistant_only=True) for a in answers]
    supervised = [p['labels'][p['labels'] != -100] for p in pair]
    if any(not (p['labels'][1:] != -100).any() for p in pair):
        raise ValueError('截断后回答没有监督 token')
    if torch.equal(*supervised):
        raise ValueError('chosen/rejected 编码或截断后相同')
    return dict(chosen=pair[0], rejected=pair[1])


def load_preferences(paths, tokenizer, max_length, train_samples, dev_samples, seed):
    records = []
    for path in paths:
        with open(path, encoding='utf-8') as stream:
            rows = json.load(stream)
        if not isinstance(rows, list):
            raise ValueError('偏好文件必须是 JSON 数组')
        records.extend(rows)
    random.Random(seed).shuffle(records)
    samples, seen = [], set()
    stats = dict(available=len(records), scanned=0, skipped=0, duplicate_prompts=0)
    for record in records:
        stats['scanned'] += 1
        try:
            messages, _ = preference_messages(record)
            digest = hashlib.sha256(format_messages(messages).encode('utf-8')).hexdigest()
            if digest in seen:
                stats['duplicate_prompts'] += 1
                continue
            pair = encode_pair(record, tokenizer, max_length)
        except (ValueError, KeyError, TypeError):
            stats['skipped'] += 1
            continue
        seen.add(digest)
        samples.append(pair)
        if len(samples) == train_samples + dev_samples:
            break
    if len(samples) < train_samples + dev_samples:
        raise ValueError(f'有效偏好对不足：{len(samples)}，请增大 max-length 或减少样本数')
    return samples[dev_samples:], samples[:dev_samples], stats


def collate_preferences(samples, pad_id):
    return {key: collate_messages([s[key] for s in samples], pad_id) for key in ('chosen', 'rejected')}


def response_log_probs(model, batch):
    logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
    return sequence_log_probs(logits, batch['labels'])
