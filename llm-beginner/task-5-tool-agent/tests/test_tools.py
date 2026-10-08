import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tools import calculator, python_sandbox, file_search, wiki


class ToolTests(unittest.TestCase):
    def test_calculator(self):
        self.assertEqual(calculator.run({'expression': '(123+456)*789'}), '456831')
        self.assertAlmostEqual(float(calculator.run({'expression': 'sqrt(2026)'})), 45.0111097397)
        for expression in ['__import__("os")', '(1).__class__', '[1,2]', '2**1000000']:
            with self.assertRaises(ValueError):
                calculator.run({'expression': expression})
        with self.assertRaises(ZeroDivisionError):
            calculator.run({'expression': '1/0'})

    def test_python(self):
        code = 'def palindrome(s):\n    return s == s[::-1]\nprint(palindrome("level"), palindrome("world"))'
        self.assertEqual(python_sandbox.run({'code': code}).strip(), 'True False')
        self.assertEqual(python_sandbox.run({'code': 'import math\nprint(f"{math.sqrt(2026):.6f}")'}).strip(), '45.011110')
        for code in ['import os', 'print((()).__class__)']:
            with self.assertRaises(ValueError):
                python_sandbox.run({'code': code})
        with self.assertRaises(RuntimeError):
            python_sandbox.run({'code': 'open("secret.txt")'})
        with self.assertRaisesRegex(RuntimeError, '8000'):
            python_sandbox.run({'code': 'print("a"*9000)'})
        with patch.object(python_sandbox, 'TIMEOUT', .5):
            with self.assertRaises(TimeoutError):
                python_sandbox.run({'code': 'while True: pass'})

    def test_file_search(self):
        result = json.loads(file_search.run({'pattern': '*.md', 'dir': 'data/agent-fixtures'}))
        self.assertEqual(result['count'], 2)
        result = json.loads(file_search.run({'pattern': 'TODO', 'dir': 'data/agent-fixtures'}))
        self.assertTrue(any('todo_note.md' in x['path'] for x in result['matches']))
        result = json.loads(file_search.run({'pattern': '*', 'dir': 'data/agent-fixtures/README.md'}))
        self.assertIn('本地文件检索测试文件', result['matches'][0]['snippet'])
        with self.assertRaises(ValueError):
            file_search.run({'pattern': '*', 'dir': '..'})

    @patch('src.tools.wiki.request')
    def test_wiki_search_fallback(self, request):
        request.side_effect = [
            {'query': {'pages': {'-1': {'missing': ''}}}},
            {'query': {'search': [{'title': '图灵机'}]}},
            {'query': {'pages': {'1': dict(title='图灵机', fullurl='https://zh.wikipedia.org/wiki/图灵机', extract='图灵机简介')}}}
        ]
        result = json.loads(wiki.run({'query': '图灵机相关'}))
        self.assertEqual(result['title'], '图灵机')
        self.assertEqual(request.call_args_list[0].args[0], 'zh')

    def test_schemas(self):
        for module in [calculator, python_sandbox, file_search, wiki]:
            schema = module.TOOL_SCHEMA
            self.assertEqual(schema['type'], 'function')
            self.assertEqual(schema['function']['name'], module.__name__.split('.')[-1])
            self.assertTrue(schema['function']['parameters']['required'])


if __name__ == '__main__':
    unittest.main()
