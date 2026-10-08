import json
from pathlib import Path

import faiss

from .embedding import Embedder
from .indexer import file_hash
from .reranker import Reranker

ROOT = Path(__file__).resolve().parents[1]


class Retriever:
    def __init__(self, index_dir=None, device='cpu', use_reranker=True, candidate_k=40,
                 reranker_path=None, reranker_batch_size=8):
        if type(candidate_k) is not int or candidate_k <= 0:
            raise ValueError('candidate_k 必须是正整数')
        self.candidate_k = candidate_k
        directory = Path(index_dir) if index_dir else ROOT / 'data/index'
        metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        for name in ['index.faiss', 'chunks.json']:
            if file_hash(directory / name) != metadata['files'][name]:
                raise ValueError('索引文件校验失败，请重新建立索引')
        self.index = faiss.read_index(str(directory / 'index.faiss'))
        self.chunks = json.loads((directory / 'chunks.json').read_text(encoding='utf-8'))
        if self.index.ntotal != len(self.chunks) or len(self.chunks) != metadata['count'] or self.index.d != metadata['dimension']:
            raise ValueError('索引与片段元数据不一致')
        self.embedder = Embedder(metadata['embedding_model'], device)
        self.reranker = Reranker(reranker_path or ROOT / 'models/bge-reranker-base',
                                device, reranker_batch_size) if use_reranker else None

    def retrieve(self, query, k=10):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query 不能为空')
        if type(k) is not int or k <= 0:
            raise ValueError('k 必须是正整数')
        vectors = self.embedder.encode([query], is_query=True)
        count = max(k, self.candidate_k) if self.reranker else k
        scores, indices = self.index.search(vectors, min(count, self.index.ntotal))
        candidates = [dict(self.chunks[int(i)], score=float(score))
                      for score, i in zip(scores[0], indices[0]) if i >= 0]
        return self.reranker.rerank(query, candidates, k) if self.reranker else candidates
