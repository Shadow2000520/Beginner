"""Regression tests for variable dimensions, padding and safe checkpoint reload."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from run_matching import CLASSES, load_checkpoint, save_checkpoint, select_threshold


class MatchingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.config = BertConfig(vocab_size=12, hidden_size=16, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=32, pad_token_id=0)
        self.ids = torch.tensor([[2, 5, 3], [2, 6, 3]])
        self.types = torch.zeros_like(self.ids)
        self.mask = torch.ones_like(self.ids)

    def test_all_four_forward_backward_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            vocab = Path(tmp)/'vocab.txt'
            vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '你', '好', '天', '气', '很', '棒', '吗']), encoding='utf-8')
            tokenizer = BertTokenizer(vocab_file=str(vocab))
            for name, cls in CLASSES.items():
                with self.subTest(method=name):
                    model = cls(BertModel(self.config))
                    if name == 'pointwise':
                        output = model(self.ids, self.types, attention_mask=self.mask)
                        self.assertEqual(tuple(output.shape), (2, 2))
                    elif name == 'simcse':
                        output = model(self.ids, self.types, self.ids, self.types)
                        self.assertGreater(output.item(), 0)
                    else:
                        kwargs = {f'{side}_{key}': value for side in ('query', 'doc') for key, value in
                                  {'input_ids': self.ids, 'token_type_ids': self.types, 'attention_mask': self.mask}.items()}
                        output = model.get_similarity(**kwargs) if name == 'dssm' else model.get_similarity_label(**kwargs)
                    output.sum().backward()
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))
                    path = Path(tmp)/name
                    save_checkpoint(path, model, tokenizer, {'method': name, 'max_length': 16, 'threshold': .5})
                    loaded, _, _ = load_checkpoint(path, torch.device('cpu'))
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, loaded.state_dict()[key])

    def test_simcse_padding_does_not_change_embedding(self):
        model = CLASSES['simcse'](BertModel(self.config)).eval()
        with torch.no_grad():
            a = model.get_pooled_embedding(self.ids)
            b = model.get_pooled_embedding(torch.nn.functional.pad(self.ids, (0, 5)))
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_threshold_uses_dev_scores_and_handles_ties(self):
        threshold = select_threshold([0, 0, 1, 1], [.1, .2, .6, .7])
        self.assertEqual(threshold, .6)
        self.assertTrue(np.isfinite(select_threshold([0, 1], [.5, .5])))


if __name__ == '__main__':
    unittest.main()
