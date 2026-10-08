"""Checks answer offsets, extraction decoding and checkpoint round trips."""
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from run_uie import encode_row, load_checkpoint, save_checkpoint, score_records, spans_from_probs
from model import UIE


class UIETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.tmp = tempfile.TemporaryDirectory()
        vocab = Path(self.tmp.name)/'vocab.txt'
        vocab.write_text('\n'.join(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','人','物','谭','孝','曾','是','元','寿','的','父','亲']),encoding='utf-8')
        self.tokenizer = BertTokenizerFast(vocab_file=str(vocab))
        self.config = BertConfig(vocab_size=16,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,pad_token_id=0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_gold_span_aligns_only_to_content_tokens(self):
        row={'prompt':'人物','content':'谭孝曾是谭元寿的父亲','result_list':[{'text':'谭元寿','start':4,'end':7}]}
        item=encode_row(self.tokenizer,row,32)
        self.assertEqual(item['omitted'],0)
        self.assertEqual(item['unaligned'],0)
        starts=item['start_labels'].nonzero().flatten().tolist()
        ends=item['end_labels'].nonzero().flatten().tolist()
        self.assertEqual(len(starts),1)
        self.assertEqual(len(ends),1)
        self.assertEqual(spans_from_probs(item['start_labels'],item['end_labels'],item['offsets'],item['content_mask'],.5),[(4,7)])
        rec={'row':row,'offsets':item['offsets'],'mask':item['content_mask'],'start_probs':item['start_labels'],'end_probs':item['end_labels']}
        self.assertEqual(score_records([rec],.5)[0]['all']['f1'],1.)

    def test_variable_width_model_and_safe_checkpoint(self):
        model=UIE(BertModel(self.config))
        item=encode_row(self.tokenizer,{'prompt':'人物','content':'谭孝曾','result_list':[]},24)
        start,end=model(item['input_ids'].unsqueeze(0),item['token_type_ids'].unsqueeze(0),item['attention_mask'].unsqueeze(0),return_logits=True)
        self.assertEqual(start.shape,(1,24))
        self.assertEqual(end.shape,(1,24))
        (start.sum()+end.sum()).backward()
        self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))
        path=Path(self.tmp.name)/'checkpoint'
        save_checkpoint(path,model,self.tokenizer,{'max_length':24,'threshold':.5})
        restored,_,info=load_checkpoint(path,torch.device('cpu'))
        self.assertEqual(info['max_length'],24)
        for key,val in model.state_dict().items():
            torch.testing.assert_close(val,restored.state_dict()[key])


if __name__=='__main__':
    unittest.main()
