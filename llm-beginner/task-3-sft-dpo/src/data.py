import hashlib
import json
import random

from .chat import encode_messages


def clean_moss(text, prefix, suffix):
    text = text.strip()
    if text.startswith(prefix):
        text = text[len(prefix):].lstrip()
    if text.endswith(suffix):
        text = text[:-len(suffix)].rstrip()
    return text


def moss_messages(record):
    messages = []
    if record.get('meta_instruction'):
        messages.append(dict(role='system', content=record['meta_instruction']))
    for key in sorted(record['chat'], key=lambda k: int(k.split('_')[-1])):
        turn = record['chat'][key]
        human = clean_moss(turn['Human'], '<|Human|>:', '<eoh>')
        answer = clean_moss(turn['MOSS'], '<|MOSS|>:', '<eom>')
        if not human or not answer:
            raise ValueError('空的 user/assistant 消息')
        messages.extend([dict(role='user', content=human), dict(role='assistant', content=answer)])
    return messages


def load_moss(path, tokenizer, max_length, train_samples, dev_samples, seed):
    samples, seen = [], set()
    stats = dict(read=0, skipped=0, duplicates=0)
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            stats['read'] += 1
            if stats['read'] > max(1000, 10 * (train_samples + dev_samples)):
                raise ValueError('扫描记录过多但有效样本不足，请增大 max-length 或检查数据格式')
            try:
                messages = moss_messages(json.loads(line))
                digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                if digest in seen:
                    stats['duplicates'] += 1
                    continue
                seen.add(digest)
                sample = encode_messages(messages, tokenizer, max_length)
                if not (sample['labels'][1:] != -100).any():
                    stats['skipped'] += 1
                    continue
            except (KeyError, ValueError, TypeError):
                stats['skipped'] += 1
                continue
            samples.append(sample)
            if len(samples) >= train_samples + dev_samples:
                break
    if len(samples) < train_samples + dev_samples:
        raise ValueError(f'有效对话不足：{len(samples)}，需要 {train_samples + dev_samples}')
    random.Random(seed).shuffle(samples)
    return samples[dev_samples:], samples[:dev_samples], stats
