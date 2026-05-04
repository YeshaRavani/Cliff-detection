from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool


class MoleculeEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int = 9,
        edge_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=edge_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = torch.relu(h)
            h = self.dropout(h)

        return global_mean_pool(h, batch_index)


class MoleculePairClassifier(nn.Module):
    def __init__(
        self,
        node_dim: int = 9,
        edge_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_pair_features: bool = False,
    ):
        super().__init__()
        self.use_pair_features = use_pair_features
        self.encoder = MoleculeEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Keep only similarity as an optional pair-level feature.
        # delta_gap defines the target label and would leak the answer.
        pair_feature_dim = 1 if use_pair_features else 0
        classifier_input_dim = hidden_dim * 4 + pair_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _get_batch_index(batch, attr_name: str, node_tensor: torch.Tensor) -> torch.Tensor:
        follow_batch_name = f"{attr_name}_batch"
        if hasattr(batch, follow_batch_name):
            return getattr(batch, follow_batch_name)

        fallback_name = attr_name.replace("x_", "batch_")
        if hasattr(batch, fallback_name):
            return getattr(batch, fallback_name)

        return torch.zeros(node_tensor.size(0), dtype=torch.long, device=node_tensor.device)

    def forward(self, batch) -> torch.Tensor:
        batch_1 = self._get_batch_index(batch, "x_1", batch.x_1)
        batch_2 = self._get_batch_index(batch, "x_2", batch.x_2)

        h1 = self.encoder(batch.x_1, batch.edge_index_1, batch.edge_attr_1, batch_1)
        h2 = self.encoder(batch.x_2, batch.edge_index_2, batch.edge_attr_2, batch_2)

        combined = [h1, h2, torch.abs(h1 - h2), h1 * h2]

        if self.use_pair_features:
            combined.append(batch.similarity.view(-1, 1))

        pair_embedding = torch.cat(combined, dim=-1)
        logits = self.classifier(pair_embedding)
        return logits.view(-1)
