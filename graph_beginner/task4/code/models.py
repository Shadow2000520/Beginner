import math

import torch
from torch import nn
from torch.nn import functional as F


MODELS = ("transe", "rotate", "conve")


class TransE(nn.Module):
    def __init__(self, entities, relations, dimension, gamma=12.0, **_):
        super().__init__()
        self.gamma = gamma
        self.entity = nn.Embedding(entities, dimension)
        self.relation = nn.Embedding(relations, dimension)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def score(self, triples):
        scores = self.score_all_tails(triples[:, :2])
        return scores.gather(1, triples[:, 2, None]).squeeze(1)

    def score_all_tails(self, pairs):
        head = self.entity(pairs[:, 0])
        relation = self.relation(pairs[:, 1])
        distance = head[:, None, :] + relation[:, None, :] - self.entity.weight[None, :, :]
        return self.gamma - torch.linalg.vector_norm(distance, ord=1, dim=-1)

    def post_step(self):
        with torch.no_grad():
            self.entity.weight.copy_(F.normalize(self.entity.weight, p=2, dim=-1))


class RotatE(nn.Module):
    def __init__(self, entities, relations, dimension, gamma=12.0, **_):
        super().__init__()
        self.gamma = gamma
        self.dimension = dimension
        self.entity = nn.Embedding(entities, dimension * 2)
        self.phase = nn.Embedding(relations, dimension)
        scale = (gamma + 2.0) / dimension
        nn.init.uniform_(self.entity.weight, -scale, scale)
        nn.init.uniform_(self.phase.weight, -math.pi, math.pi)

    def score(self, triples):
        scores = self.score_all_tails(triples[:, :2])
        return scores.gather(1, triples[:, 2, None]).squeeze(1)

    def score_all_tails(self, pairs):
        head_real, head_imag = self.entity(pairs[:, 0]).chunk(2, dim=-1)
        tail_real, tail_imag = self.entity.weight.chunk(2, dim=-1)
        phase = self.phase(pairs[:, 1])
        relation_real = torch.cos(phase)
        relation_imag = torch.sin(phase)
        rotated_real = head_real * relation_real - head_imag * relation_imag
        rotated_imag = head_real * relation_imag + head_imag * relation_real
        result_real = rotated_real[:, None, :] - tail_real[None, :, :]
        result_imag = rotated_imag[:, None, :] - tail_imag[None, :, :]
        distance = torch.sqrt(result_real.square() + result_imag.square() + 1e-9).sum(-1)
        return self.gamma - distance

    def post_step(self):
        pass


class ConvE(nn.Module):
    def __init__(self, entities, relations, dimension, conv_layers=1,
                 channels=16, dropout=0.2, **_):
        super().__init__()
        if dimension % 8:
            raise ValueError("ConvE dimension must be divisible by 8")
        self.height = 8
        self.width = dimension // self.height
        self.entity = nn.Embedding(entities, dimension)
        self.relation = nn.Embedding(relations, dimension)
        self.bias = nn.Parameter(torch.zeros(entities))
        blocks = []
        in_channels = 1
        for _ in range(conv_layers):
            blocks.extend((
                nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Dropout2d(dropout),
            ))
            in_channels = channels
        self.convolution = nn.Sequential(*blocks)
        self.projection = nn.Linear(channels * self.height * 2 * self.width, dimension)
        self.hidden_dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)
        nn.init.xavier_uniform_(self.projection.weight)

    def score(self, triples):
        scores = self.score_all_tails(triples[:, :2])
        return scores.gather(1, triples[:, 2, None]).squeeze(1)

    def score_all_tails(self, pairs):
        head = self.entity(pairs[:, 0]).view(-1, 1, self.height, self.width)
        relation = self.relation(pairs[:, 1]).view(-1, 1, self.height, self.width)
        features = torch.cat((head, relation), dim=2)
        features = self.convolution(features).flatten(1)
        features = self.hidden_dropout(F.relu(self.projection(features)))
        return features @ self.entity.weight.T + self.bias

    def post_step(self):
        pass


def build_model(name, entities, relations, dimension=64, gamma=12.0,
                conv_layers=1, channels=16, dropout=0.2):
    classes = {"transe": TransE, "rotate": RotatE, "conve": ConvE}
    if name not in classes:
        raise ValueError(f"Unknown model: {name}")
    return classes[name](
        entities, relations, dimension, gamma=gamma,
        conv_layers=conv_layers, channels=channels, dropout=dropout)
