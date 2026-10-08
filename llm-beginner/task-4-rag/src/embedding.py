import numpy as np
from sentence_transformers import SentenceTransformer

QUERY_PREFIX = '为这个句子生成表示以用于检索相关文章：'


class Embedder:
    def __init__(self, model_path, device='cpu'):
        self.model = SentenceTransformer(str(model_path), device=device, local_files_only=True)

    def encode(self, texts, batch_size=32, is_query=False):
        texts = [QUERY_PREFIX + text if is_query else text for text in texts]
        lengths = self.model.tokenizer(texts, truncation=False, padding=False)['input_ids']
        if any(len(ids) > self.model.max_seq_length for ids in lengths):
            raise ValueError('文本超出 embedding 模型长度，请减小 chunk-size 或缩短问题')
        vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=not is_query)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise ValueError('embedding 必须是有限且 L2 归一化的向量')
        return vectors
