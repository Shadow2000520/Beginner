import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tokenizer import BPETokenizer, merge_pair


class BPETests(unittest.TestCase):
    def test_highest_frequency_merge_and_overlap(self):
        tok = BPETokenizer.train("abababab", vocab_size=260)
        self.assertEqual(tok.merges, [(97, 98)])
        self.assertEqual(tok.encode("abab"), [259, 259])
        self.assertEqual(merge_pair([97] * 5, (97, 97), 259), [259, 259, 97])

    def test_rank_order_not_longest_match(self):
        tok = BPETokenizer([(97, 98), (98, 99), (97, 260)])
        self.assertEqual(tok.encode("abc"), [259, 99])

    def test_unicode_and_unseen_characters(self):
        tok = BPETokenizer.train("床前明月光" * 10, 280)
        rng = random.Random(0)
        samples = ["", "𠮷🙂 café\r\n\t\x00", "<bos>", "Hello, world!"]
        samples += ["".join(rng.choice("甲乙🙂𠮷a\né") for _ in range(30)) for _ in range(30)]
        for text in samples:
            self.assertEqual(tok.decode(tok.encode(text)), text)
        self.assertEqual(BPETokenizer().decode([0xE4, 0xB8, 0xAD]), "中")
        with self.assertRaises(UnicodeDecodeError):
            tok.decode([0xE4])

    def test_save_load_specials_and_determinism(self):
        tok = BPETokenizer.train("aaaa abc 中文" * 10, 290)
        self.assertEqual(tok.merges, BPETokenizer.train("aaaa abc 中文" * 10, 290).merges)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tokenizer.json"
            tok.save_pretrained(path)
            loaded = BPETokenizer.from_pretrained(path)
            self.assertEqual(tok.encode("中文🙂"), loaded.encode("中文🙂"))
            self.assertEqual(tok.vocab_size, loaded.vocab_size)
        ids = tok.encode("你好", add_bos=True, add_eos=True)
        self.assertEqual(tok.decode(ids), "你好")
        self.assertEqual(tok.decode(ids, skip_special_tokens=False), "<bos>你好<eos>")

    def test_invalid_and_small_corpus(self):
        self.assertEqual(BPETokenizer.train("a", 512).vocab_size, 259)
        with self.assertRaises(ValueError):
            BPETokenizer([(259, 1)])
        with self.assertRaises(ValueError):
            BPETokenizer.train("abc", 100)
        with self.assertRaises(ValueError):
            BPETokenizer().decode([999])


if __name__ == "__main__":
    unittest.main()
