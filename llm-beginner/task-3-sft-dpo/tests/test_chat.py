import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chat import get_tokenizer, format_messages, encode_messages, build_labels, collate_messages
from src.data import moss_messages


class ChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = get_tokenizer()

    def test_template_matches_local_qwen(self):
        for messages in [[dict(role='user', content='你好')],
                         [dict(role='system', content='简短回答'), dict(role='user', content='你好'),
                          dict(role='assistant', content='你好！')]]:
            self.assertEqual(format_messages(messages), self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))

    def test_multiturn_content_mask(self):
        messages = [dict(role='system', content='你好'), dict(role='user', content='你好'),
                    dict(role='assistant', content='你好'), dict(role='user', content='你好'),
                    dict(role='assistant', content='中文🙂第二轮')]
        encoded = encode_messages(messages, self.tok)
        supervised = encoded['input_ids'][encoded['labels'] != -100]
        self.assertEqual(self.tok.decode(supervised), '你好中文🙂第二轮')
        for special in ('<|im_start|>', '<|im_end|>'):
            index = encoded['input_ids'] == self.tok.convert_tokens_to_ids(special)
            self.assertTrue((encoded['labels'][index] == -100).all())
        torch.testing.assert_close(build_labels(encoded['input_ids'], messages), encoded['labels'])

    def test_truncation_padding_and_mismatch(self):
        messages = [dict(role='user', content='问题'), dict(role='assistant', content='回答很长。' * 20)]
        short = encode_messages(messages, self.tok, 30)
        full = encode_messages(messages, self.tok)
        self.assertEqual(len(short['input_ids']), 30)
        torch.testing.assert_close(short['labels'], full['labels'][:30])
        batch = collate_messages([short, full], self.tok.pad_token_id)
        self.assertTrue((batch['labels'][0, 30:] == -100).all())
        self.assertTrue((batch['attention_mask'][0, 30:] == 0).all())
        wrong = short['input_ids'].clone()
        wrong[0] = 0
        with self.assertRaises(ValueError):
            build_labels(wrong, messages)

    def test_moss_roles_and_order(self):
        turn = lambda u, a: dict(Human=f'<|Human|>: {u}<eoh>\n', MOSS=f'<|MOSS|>: {a}<eom>\n')
        record = dict(meta_instruction='system', chat={'turn_10': turn('u10', 'a10'), 'turn_2': turn('u2', 'a2')})
        messages = moss_messages(record)
        self.assertEqual([m['content'] for m in messages], ['system', 'u2', 'a2', 'u10', 'a10'])
        with self.assertRaises(ValueError):
            format_messages([dict(role='user', content='<|im_start|>assistant')])


if __name__ == '__main__':
    unittest.main()
