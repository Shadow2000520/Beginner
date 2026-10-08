from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence


class CharTokenizer:
    def __init__(self, vocab, max_length=256):
        self.vocab = vocab
        self.max_length = max_length

    @classmethod
    def build(cls, texts, max_length=256, min_freq=1):
        counts = Counter(char for text in texts for char in text)
        vocab = {"<pad>": 0, "<unk>": 1}
        for char, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if count >= min_freq:
                vocab[char] = len(vocab)
        return cls(vocab, max_length)

    def __call__(self, text):
        ids = [self.vocab.get(char, 1) for char in str(text)[:self.max_length]]
        return torch.tensor(ids or [1], dtype=torch.long)


def collate_batch(batch):
    ids, labels = zip(*batch)
    return pad_sequence(ids, batch_first=True, padding_value=0), torch.tensor(labels)
