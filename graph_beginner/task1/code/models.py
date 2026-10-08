"""Four PyG node classifiers; the final layer returns unnormalised logits."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


class NodeClassifier(nn.Module):
    def __init__(self, model, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, heads=4):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if model == "gat" and num_layers > 1 and hidden_channels % heads:
            raise ValueError("GAT hidden_channels must be divisible by heads")
        self.name, self.dropout = model, dropout
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList()
        for i, (inc, outc) in enumerate(zip(dims, dims[1:])):
            if model == "gcn":
                conv = GCNConv(inc, outc, cached=False)
            elif model == "gat":
                final = i == num_layers - 1
                conv = GATConv(inc, outc if final else outc // heads,
                               heads=1 if final else heads, concat=not final,
                               dropout=dropout)
            elif model == "sage":
                conv = SAGEConv(inc, outc, aggr="mean")
            elif model == "gin":
                conv = GINConv(nn.Sequential(nn.Linear(inc, hidden_channels),
                              nn.ReLU(), nn.Linear(hidden_channels, outc)),
                              train_eps=True)
            else:
                raise ValueError(f"Unknown model: {model}")
            self.convs.append(conv)

    def adjacency(self, edge_index, num_nodes):
        # GAT uses edge indices; other models use sparse adjacency.
        if self.name == "gat":
            return edge_index
        return torch.sparse_coo_tensor(
            edge_index.flip(0), torch.ones(edge_index.size(1), device=edge_index.device),
            (num_nodes, num_nodes)).coalesce().to_sparse_csr()

    def forward(self, x, adjacency):
        for i, conv in enumerate(self.convs):
            x = conv(x, adjacency)
            if i < len(self.convs) - 1:
                x = F.elu(x) if self.name == "gat" else F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
