import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    global_max_pool,
    global_mean_pool,
)


def global_min_pool(x, batch):
    return -global_max_pool(-x, batch)


class GraphClassifier(nn.Module):
    def __init__(self, model, pooling, in_channels, hidden_channels,
                 out_channels, num_layers=2, dropout=0.5, heads=4):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if model == "gat" and hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")

        self.model = model
        self.pooling = pooling
        self.dropout = dropout
        self.convs = nn.ModuleList()
        dims = [in_channels] + [hidden_channels] * num_layers

        for in_dim, out_dim in zip(dims, dims[1:]):
            if model == "gcn":
                conv = GCNConv(in_dim, out_dim)
            elif model == "gat":
                conv = GATConv(
                    in_dim, out_dim // heads, heads=heads,
                    concat=True, dropout=dropout)
            elif model == "sage":
                conv = SAGEConv(in_dim, out_dim, aggr="mean")
            elif model == "gin":
                conv = GINConv(nn.Sequential(
                    nn.Linear(in_dim, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, out_dim),
                ), train_eps=True)
            else:
                raise ValueError(f"Unknown model: {model}")
            self.convs.append(conv)

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == "mean":
            graph_features = global_mean_pool(x, batch)
        elif self.pooling == "max":
            graph_features = global_max_pool(x, batch)
        elif self.pooling == "min":
            graph_features = global_min_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.classifier(graph_features)
