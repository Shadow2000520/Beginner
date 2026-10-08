import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generator import Generator, prepare_context
from src.rag import answer


class RagTests(unittest.TestCase):
    def test_overlap_dedup_and_budget(self):
        first = dict(text='abcdefgh', source='kb.pdf:PDF页1-1', char_start=0, char_end=8)
        second = dict(text='efghijkl', source='kb.pdf:PDF页1-2', char_start=4, char_end=12)
        context, sources = prepare_context([first, dict(first), second], 10)
        self.assertEqual([s['text'] for s in sources], ['abcdefgh', 'ij'])
        self.assertEqual(sources[1]['char_start'], 8)
        self.assertEqual(sources[1]['char_end'], 10)
        self.assertEqual(sources[1]['citation'], 2)
        self.assertIn('[2] kb.pdf:PDF页1-2', context)

    def test_answer_uses_context_and_preserves_sources(self):
        retriever, generator = Mock(), Mock()
        retriever.retrieve.return_value = [dict(text='教材内容', source='kb.pdf:PDF页3-3')]
        generator.generate.return_value = '回答[1]'
        result = answer('问题', retriever=retriever, generator=generator)
        self.assertEqual(result['answer'], '回答[1]')
        self.assertEqual(result['sources'][0]['text'], '教材内容')
        self.assertIn('教材内容', generator.generate.call_args.args[1])
        retriever.retrieve.return_value = []
        generator.reset_mock()
        self.assertEqual(answer('问题', retriever=retriever, generator=generator)['sources'], [])
        generator.generate.assert_not_called()

    @patch('src.generator.requests.post')
    def test_api_contract(self, post):
        post.return_value.json.return_value = {'choices': [{'message': {'content': ' 回答[1] '}, 'finish_reason': 'stop'}]}
        model = Generator('http://localhost:11434/v1/', 'test-model')
        self.assertEqual(model.generate('问题', '资料'), '回答[1]')
        self.assertEqual(post.call_args.args[0], 'http://localhost:11434/v1/chat/completions')
        self.assertEqual(post.call_args.kwargs['json']['model'], 'test-model')
        post.return_value.json.return_value = {'choices': []}
        with self.assertRaises(RuntimeError):
            model.generate('问题', '资料')
        post.side_effect = requests.ConnectionError('offline')
        with self.assertRaisesRegex(RuntimeError, '调用生成服务失败'):
            model.generate('问题', '资料')


if __name__ == '__main__':
    unittest.main()
