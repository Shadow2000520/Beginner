from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models/Qwen2.5-0.5B'


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)


def render_messages(messages):
    if not messages:
        raise ValueError('对话不能为空')
    messages = list(messages)
    if messages[0]['role'] != 'system':
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}] + messages
    text, spans = '', []
    for message in messages:
        role, content = message['role'], message['content']
        if role not in ('system', 'user', 'assistant') or not isinstance(content, str):
            raise ValueError('只支持 system/user/assistant 文本消息')
        if any(t in content for t in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')):
            raise ValueError('消息正文不能包含模板控制符')
        text += f'<|im_start|>{role}\n'
        start = len(text)
        text += content
        if role == 'assistant':
            spans.append((start, len(text)))
        text += '<|im_end|>\n'
    return text, spans


def format_messages(messages):
    return render_messages(messages)[0]


def encode_messages(messages, tokenizer=None, max_length=None, *, last_assistant_only=False):
    tokenizer = tokenizer or get_tokenizer()
    text, spans = render_messages(messages)
    if last_assistant_only:
        spans = spans[-1:] if messages[-1]['role'] == 'assistant' else []
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=max_length is not None, max_length=max_length)
    ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for i, (start, end) in enumerate(encoded['offset_mapping']):
        if end > start and any(left <= start and end <= right for left, right in spans):
            labels[i] = ids[i]
    return {'input_ids': ids, 'labels': labels}


def build_labels(input_ids, messages, tokenizer=None):
    if input_ids.ndim != 1:
        raise ValueError('input_ids 必须是一维未 padding 的 token 序列')
    encoded = encode_messages(messages, tokenizer, max_length=len(input_ids))
    if not torch.equal(encoded['input_ids'], input_ids.cpu()):
        raise ValueError('input_ids 与对话模板编码不一致')
    return encoded['labels'].to(input_ids.device)


def collate_messages(samples, pad_id):
    length = max(len(s['input_ids']) for s in samples)
    ids = torch.full((len(samples), length), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention_mask = torch.zeros_like(ids)
    for i, sample in enumerate(samples):
        n = len(sample['input_ids'])
        ids[i, :n] = sample['input_ids']
        labels[i, :n] = sample['labels']
        attention_mask[i, :n] = 1
    return dict(input_ids=ids, labels=labels, attention_mask=attention_mask)
