import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chunker import chunk_text, chunk_spans, chunk_pdf


class ChunkerTests(unittest.TestCase):
    def test_coverage_and_overlap(self):
        text = '这是一段测试文本。' * 400
        spans = list(chunk_spans(text, 256, 32))
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(text))
        self.assertTrue(all(0 < end - start <= 256 for start, end in spans))
        for previous, current in zip(spans, spans[1:]):
            self.assertEqual(previous[1] - current[0], 32)
        self.assertTrue(all(text[end - 1] == '。' for _, end in spans[:-1]))

    def test_edge_cases(self):
        self.assertEqual(chunk_text('', 4, 0), [])
        self.assertEqual(chunk_text('abc', 4, 0), ['abc'])
        self.assertEqual(chunk_text('abcdef', 4, 3), ['abcd', 'bcde', 'cdef'])
        for size, overlap in [(0, 0), (4, -1), (4, 4)]:
            with self.assertRaises(ValueError):
                chunk_text('abc', size, overlap)

    def test_pdf_page_sources(self):
        pages = [{'page': 1, 'text': 'abc'}, {'page': 3, 'text': 'def'}]
        with patch('src.chunker.extract_pdf_pages', return_value=pages):
            chunks = chunk_pdf('book.pdf', 6, 1)
        self.assertEqual(chunks[0]['text'], 'abc\nde')
        self.assertEqual(chunks[0]['source'], 'book.pdf:PDF页1-3')
        self.assertEqual(chunks[-1]['page_start'], 3)


if __name__ == '__main__':
    unittest.main()
