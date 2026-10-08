from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    def __init__(self, model_path, device='cpu', batch_size=8):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('batch_size 必须是正整数')
        self.device = torch.device(device)
        self.batch_size = batch_size
        path = str(Path(model_path).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def rerank(self, query, candidates, k=10):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query 不能为空')
        if type(k) is not int or k <= 0:
            raise ValueError('k 必须是正整数')
        if not candidates:
            return []
        if len(self.tokenizer.encode(query, add_special_tokens=False)) > 256:
            raise ValueError('问题超过 256 个词元，请缩短问题')
        ranked = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start:start + self.batch_size]
            inputs = self.tokenizer([query] * len(batch), [item['text'] for item in batch],
                                    padding=True, truncation='only_second', max_length=512,
                                    return_tensors='pt').to(self.device)
            scores = self.model(**inputs).logits.reshape(-1).float()
            if scores.numel() != len(batch) or not torch.isfinite(scores).all():
                raise ValueError('reranker 应为每个文本对返回一个有限分数')
            for item, score in zip(batch, scores.cpu().tolist()):
                ranked.append(dict(item, retrieval_score=item['score'], score=score))
        return sorted(ranked, key=lambda item: item['score'], reverse=True)[:k]
