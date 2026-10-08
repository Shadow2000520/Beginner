import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


class GNNEncoder(nn.Module):
    def __init__(self, model, in_channels, hidden_channels, num_layers=2,
                 dropout=0.5, heads=4):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if model == "gat" and num_layers > 1 and hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")

        self.name = model
        self.dropout = dropout
        dims = [in_channels] + [hidden_channels] * num_layers
        self.convs = nn.ModuleList()

        for i, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            final = i == num_layers - 1
            if model == "gcn":
                conv = GCNConv(in_dim, out_dim, cached=False)
            elif model == "gat":
                conv = GATConv(
                    in_dim,
                    out_dim if final else out_dim // heads,
                    heads=1 if final else heads,
                    concat=not final,
                    dropout=dropout,
                )
            elif model == "sage":
                conv = SAGEConv(in_dim, out_dim, aggr="mean")
            elif model == "gin":
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, out_dim),
                )
                conv = GINConv(mlp, train_eps=True)
            else:
                raise ValueError(f"Unknown model: {model}")
            self.convs.append(conv)

    def adjacency(self, edge_index, num_nodes):
        if self.name == "gat":
            return edge_index
        return torch.sparse_coo_tensor(
            edge_index.flip(0),
            torch.ones(edge_index.size(1), device=edge_index.device),
            (num_nodes, num_nodes),
        ).coalesce().to_sparse_csr()

    def forward(self, x, adjacency):
        for i, conv in enumerate(self.convs):
            x = conv(x, adjacency)
            if i < len(self.convs) - 1:
                x = F.elu(x) if self.name == "gat" else F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def edge_logits(z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
