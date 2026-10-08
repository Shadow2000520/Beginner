"""Leakage, target-shift and generation-metric checks for both T5 tasks."""

import unittest

from transformers import AutoTokenizer

from run_generation import DEFAULT_MODEL, char_bleu4, char_f1, load_splits, make_loader, normalize, parse_rows


class GenerationTests(unittest.TestCase):
    def test_qa_and_filling_splits_are_group_disjoint(self):
        for task in ("qa", "filling"):
            train, valid, test, counts = load_splits(task, max_train=64, max_valid=16, max_test=16)
            self.assertEqual((len(train), len(valid), len(test)), (64, 16, 16))
            self.assertGreater(counts["removed_dev_overlap"], 0)
            self.assertGreater(counts["invalid_rows"], 0)
            train_keys = {row["key"] for row in train}
            full_train_keys = {row["key"] for row in parse_rows(task)[0]}
            valid_keys = {row["key"] for row in valid}
            test_keys = {row["key"] for row in test}
            self.assertFalse(train_keys & valid_keys)
            self.assertFalse(train_keys & test_keys)
            self.assertFalse(full_train_keys & valid_keys)
            self.assertFalse(full_train_keys & test_keys)
            self.assertFalse(valid_keys & test_keys)
            self.assertEqual((train, valid, test, counts), load_splits(task, max_train=64, max_valid=16, max_test=16))

    def test_labels_end_with_sep_and_pad_with_ignore_id(self):
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, local_files_only=True)
        rows = [
            {"source": "问题：颜色？[SEP]原文：苹果是红色的。", "target": "答案：红色", "reference": "红色"},
            {"source": "问题：地点？[SEP]原文：北京是中国的首都。", "target": "答案：北京", "reference": "北京"},
        ]
        encoded, _, _ = next(iter(make_loader(rows, tokenizer, 2, 32, 16)))
        self.assertNotIn("token_type_ids", encoded)
        self.assertEqual(encoded["labels"].shape[0], 2)
        for row in encoded["labels"]:
            ids = row[row != -100].tolist()
            self.assertEqual(ids[-1], tokenizer.sep_token_id)
            self.assertNotEqual(ids[0], tokenizer.cls_token_id)

    def test_metrics_and_answer_prefix(self):
        self.assertEqual(normalize("答案： 北 京 ", "qa"), "北京")
        self.assertEqual(char_f1("北京", "北京"), 1.0)
        self.assertEqual(char_f1("北京", "上海"), 0.0)
        self.assertAlmostEqual(char_bleu4(["苹果很好吃"], ["苹果很好吃"]), 1.0)


if __name__ == "__main__":
    unittest.main()
