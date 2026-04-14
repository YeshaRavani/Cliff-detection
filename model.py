import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool


class GNNEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 6,
        edge_feature_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=edge_feature_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        return global_mean_pool(x, batch)


class SiameseGNN(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 6,
        edge_feature_dim: int = 3,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GNNEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )

        pair_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 2, 1),
        )

    def encode(self, batch):
        return self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    def forward(self, graph1, graph2):
        h1 = self.encode(graph1)
        h2 = self.encode(graph2)

        pair_representation = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2], dim=-1)
        logits = self.classifier(pair_representation).squeeze(-1)
        return logits
