import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import faiss
import numpy as np
import torch
from transformers import BatchEncoding

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reranker import Reranker
from src.retriever import Retriever


class RetrievalTests(unittest.TestCase):
    def test_pair_scoring_batches_and_sources(self):
        reranker = Reranker.__new__(Reranker)
        reranker.device = torch.device('cpu')
        reranker.batch_size = 2
        reranker.tokenizer = Mock(return_value=BatchEncoding({'input_ids': torch.ones(1, 1)}))
        reranker.tokenizer.encode.return_value = [1]
        reranker.model = Mock(side_effect=[SimpleNamespace(logits=torch.tensor([[1.], [3.]])),
                                          SimpleNamespace(logits=torch.tensor([[2.]]))])
        candidates = [dict(text=str(i), source=f'page{i}', score=i / 10) for i in range(3)]
        result = reranker.rerank('问题', candidates, 2)
        self.assertEqual([x['text'] for x in result], ['1', '2'])
        self.assertEqual(result[0]['source'], 'page1')
        self.assertEqual(result[0]['retrieval_score'], .1)
        self.assertEqual(candidates[1]['score'], .1)
        self.assertEqual(reranker.tokenizer.call_args_list[0].args, (['问题', '问题'], ['0', '1']))
        self.assertEqual(reranker.rerank('问题', [], 3), [])

    def test_candidate_pool_and_k(self):
        retriever = Retriever.__new__(Retriever)
        retriever.chunks = [dict(text=str(i), source=str(i)) for i in range(5)]
        retriever.index = faiss.IndexFlatIP(2)
        retriever.index.add(np.array([[1, 0], [.8, .2], [.6, .4], [.4, .6], [0, 1]], dtype='float32'))
        retriever.embedder = Mock()
        retriever.embedder.encode.return_value = np.array([[1, 0]], dtype='float32')
        retriever.candidate_k = 4
        retriever.reranker = Mock()
        retriever.reranker.rerank.side_effect = lambda query, candidates, k: candidates[::-1][:k]
        self.assertEqual([x['text'] for x in retriever.retrieve('q', 2)], ['3', '2'])
        self.assertEqual(len(retriever.retrieve('q', 10)), 5)
        retriever.embedder.encode.assert_called_with(['q'], is_query=True)
        retriever.reranker = None
        self.assertEqual([x['text'] for x in retriever.retrieve('q', 2)], ['0', '1'])
        with self.assertRaises(ValueError):
            retriever.retrieve('q', 0)


if __name__ == '__main__':
    unittest.main()
