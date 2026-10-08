"""Meaningful parser, comparison, and Streamlit smoke tests."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tokenizer_utils import compare_tokenizers, parse_token_ids, sorted_vocab, vocab_stats


class FakeTokenizer:
    def __init__(self, vocab, added=None):
        self.vocab = vocab
        self.added = added or {}

    def get_vocab(self):
        return self.vocab

    def get_added_vocab(self):
        return self.added


class TokenizerUtilsTests(unittest.TestCase):
    def test_decode_input_is_json_integer_array_only(self):
        self.assertEqual(parse_token_ids("[0, 42]"), [0, 42])
        for value in ("__import__('os').system('whoami')", "[true]", "[-1]", "[1.5]", '{"id": 1}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_token_ids(value)

    def test_comparison_is_sorted_and_tracks_changed_ids(self):
        first = FakeTokenizer({"a": 2, "b": 0, "shared": 3}, {"shared": 3})
        second = FakeTokenizer({"shared": 1, "c": 0, "a": 2})
        result = compare_tokenizers(first, second)
        self.assertEqual(vocab_stats(first), {"base_vocab": 2, "added_vocab": 1, "total_vocab": 3})
        self.assertEqual(result["only_first"], [{"id": 0, "token": "b"}])
        self.assertEqual(result["only_second"], [{"id": 0, "token": "c"}])
        self.assertEqual(result["different_ids"], [{"token": "shared", "first_id": 3, "second_id": 1}])
        self.assertEqual(sorted_vocab({"z": 2, "a": 0}), [{"id": 0, "token": "a"}, {"id": 2, "token": "z"}])


class StreamlitSmokeTests(unittest.TestCase):
    def test_viewer_and_comparison_with_local_tokenizers(self):
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[2]
        qwen = root / "LLM/models/Qwen2.5-0.5B"
        t5 = root / "answer_generation/models/t5-small-chinese-cluecorpussmall"
        if not (qwen.is_dir() and t5.is_dir()):
            self.skipTest("本地 smoke test 模型未下载")

        app = AppTest.from_file(str(Path(__file__).with_name("web_ui.py")), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        app.text_input(key="viewer_source").input(str(qwen)).run()
        app.button(key="load_viewer").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("151665" in value.value for value in app.markdown))

        app.text_input(key="search_完整词表").input("<|im_start|>").run()
        self.assertEqual(len(app.exception), 0)
        app.text_input(key="decode_ids").input("[true]").run()
        self.assertTrue(any("非负整数" in item.value for item in app.error))

        app.text_input(key="compare_first").input(str(qwen)).run()
        app.text_input(key="compare_second").input(str(t5)).run()
        app.button(key="load_comparison").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("150171" in value.value for value in app.subheader))


if __name__ == "__main__":
    unittest.main()
