"""
benchmark_model.py — Dual-Tower Activity Cliff Classifier

Architecture
------------
Unlike the current MoleculePairClassifier (which uses a *single shared*
GINEConv encoder for both molecules), this model uses two fully *independent*
encoder towers — one dedicated to molecule 1, one to molecule 2.

Tower A (mol-1): node_proj_a → [GINEConv → BN → ReLU → Dropout] × N → global_mean_pool → h1
Tower B (mol-2): node_proj_b → [GINEConv → BN → ReLU → Dropout] × N → global_mean_pool → h2

Fusion: cat(h1, h2, |h1-h2|, h1*h2)  →  MLP  →  scalar logit

This is the benchmark: no weight sharing at all. The two towers can specialise
independently, which may (or may not) outperform the shared-weight baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool


class _Tower(nn.Module):
    """A single independent GNN encoder tower."""

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


class DualTowerClassifier(nn.Module):
    """
    Dual-Tower benchmark model.

    Two fully independent GNN towers (tower_a for mol-1, tower_b for mol-2)
    produce embeddings h1 and h2.  These are fused and passed through an MLP
    classifier to produce a single cliff-probability logit.

    Parameters
    ----------
    node_dim      : atom feature dimension (default 9, matches featurizer.py)
    edge_dim      : bond feature dimension (default 4)
    hidden_dim    : width of every hidden layer
    num_layers    : number of GINEConv message-passing steps per tower
    dropout       : dropout rate (applied after every conv block and in MLP)
    use_pair_features : if True, concatenates Tanimoto similarity as an extra
                        scalar input to the classifier head
    """

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

        # --- two independent towers ---
        self.tower_a = _Tower(node_dim, edge_dim, hidden_dim, num_layers, dropout)
        self.tower_b = _Tower(node_dim, edge_dim, hidden_dim, num_layers, dropout)

        # --- fusion + classifier ---
        # Fusion dim: h1 ‖ h2 ‖ |h1-h2| ‖ h1*h2  →  4 * hidden_dim
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

    # ------------------------------------------------------------------
    # Batch-index helpers (identical to MoleculePairClassifier)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_batch_index(batch, attr_name: str, node_tensor: torch.Tensor) -> torch.Tensor:
        follow_batch_name = f"{attr_name}_batch"
        if hasattr(batch, follow_batch_name):
            return getattr(batch, follow_batch_name)

        fallback_name = attr_name.replace("x_", "batch_")
        if hasattr(batch, fallback_name):
            return getattr(batch, fallback_name)

        return torch.zeros(node_tensor.size(0), dtype=torch.long, device=node_tensor.device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch) -> torch.Tensor:
        batch_1 = self._get_batch_index(batch, "x_1", batch.x_1)
        batch_2 = self._get_batch_index(batch, "x_2", batch.x_2)

        # Independent encodings
        h1 = self.tower_a(batch.x_1, batch.edge_index_1, batch.edge_attr_1, batch_1)
        h2 = self.tower_b(batch.x_2, batch.edge_index_2, batch.edge_attr_2, batch_2)

        combined = [h1, h2, torch.abs(h1 - h2), h1 * h2]

        if self.use_pair_features:
            combined.append(batch.similarity.view(-1, 1))

        pair_embedding = torch.cat(combined, dim=-1)
        logits = self.classifier(pair_embedding)
        return logits.view(-1)
