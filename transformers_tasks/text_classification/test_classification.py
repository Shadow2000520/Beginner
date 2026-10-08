"""Data and batching checks for the eight-class classifier."""

import unittest

from transformers import AutoTokenizer

from run_classification import DEFAULT_MODEL, load_splits, make_loader, read_labels


class ClassificationTests(unittest.TestCase):
    def test_mapping_and_disjoint_stratified_split(self):
        train, valid, test, mapping = load_splits()
        self.assertEqual((len(train), len(valid), len(test)), (322, 80, 63))
        self.assertEqual(mapping, read_labels())
        self.assertEqual(len(mapping), 8)
        texts = [{content for _, content in rows} for rows in (train, valid, test)]
        self.assertFalse(texts[0] & texts[1])
        self.assertFalse(texts[0] & texts[2])
        self.assertFalse(texts[1] & texts[2])
        self.assertEqual((train, valid, test, mapping), load_splits())
        self.assertEqual({label for label, _ in train}, set(mapping))
        self.assertEqual({label for label, _ in valid}, set(mapping))

    def test_batch_padding_attention_mask_and_labels(self):
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, local_files_only=True)
        batch = next(iter(make_loader([(1, "苹果很好吃"), (5, "酒店房间干净而且交通方便")], tokenizer, 2, 12)))
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertLessEqual(batch["input_ids"].shape[1], 12)
        self.assertEqual(batch["labels"].tolist(), [1, 5])
        self.assertEqual(batch["attention_mask"].sum(dim=1).tolist(), [len(tokenizer("苹果很好吃", max_length=12, truncation=True)["input_ids"]), 12])


if __name__ == "__main__":
    unittest.main()
