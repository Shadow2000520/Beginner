import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.decoding import TextConstraints, utf8_transition
from src.model import MiniGPT
from src.tokenizer import BPETokenizer


class DecodingTests(unittest.TestCase):
    def test_invalid_utf8_sequences(self):
        for piece in [b'\x80', b'\xc0\x80', b'\xed\xa0\x80', b'\xf4\x90\x80\x80', b'\xff']:
            self.assertIsNone(utf8_transition(b'', piece))
        self.assertEqual(utf8_transition(b'\xe4\xb8', b'\xad'), ('中', b''))

    def test_budget_and_continuation(self):
        tok = BPETokenizer()
        processor = TextConstraints(tok, 1., 0)
        scores = torch.zeros(1, tok.vocab_size)
        full = processor(scores, torch.tensor([[ord('a')]]), 0)
        self.assertTrue(torch.isfinite(full[0, ord('b')]))
        self.assertFalse(torch.isfinite(full[0, 0xe4]))
        self.assertFalse(torch.isfinite(full[0, tok.bos_id]))
        pending = processor(scores, torch.tensor([[0xe4, 0xb8]]), 0)
        self.assertTrue(torch.isfinite(pending[0, 0xad]))
        self.assertFalse(torch.isfinite(pending[0, ord('a')]))

    def test_character_repetition_across_byte_tokens(self):
        tok = BPETokenizer()
        history = tok.encode('春风春') + list('风'.encode('utf-8')[:-1])
        scores = torch.zeros(1, tok.vocab_size)
        out = TextConstraints(tok, 1., 2)(scores, torch.tensor([history]), 10)
        self.assertFalse(torch.isfinite(out[0, '风'.encode('utf-8')[-1]]))
        self.assertTrue(torch.equal(scores, torch.zeros_like(scores)))

    def test_repetition_inside_merged_token(self):
        tok = BPETokenizer([(97, 97), (259, 259)])
        result = TextConstraints(tok, 1., 2)(torch.zeros(1, tok.vocab_size), torch.tensor([[98]]), 4)
        self.assertFalse(torch.isfinite(result[0, 260]))

    def test_generated_text_is_strict_utf8_with_cache(self):
        torch.manual_seed(42)
        tok = BPETokenizer()
        model = MiniGPT(tok.vocab_size, d_model=16, n_heads=2, n_layers=1, block_size=8).eval()
        prompt = torch.tensor([tok.encode('春')])
        processor = TextConstraints(tok)
        for length in [1, 2, 3, 13]:
            outputs = [model.generate(prompt, length, temperature=.8, top_p=.9,
                       generator=torch.Generator().manual_seed(42), use_cache=cache,
                       logits_processor=processor) for cache in [True, False]]
            self.assertTrue(torch.equal(*outputs))
            text = tok.decode(outputs[0][0].tolist())
            grams = [text[i:i+4] for i in range(len(text)-3)]
            self.assertEqual(len(grams), len(set(grams)))


if __name__ == '__main__':
    unittest.main()
