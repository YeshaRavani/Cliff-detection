from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from featurizer import smiles_to_graph


class MoleculePairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_1":
            return self.x_1.size(0)
        if key == "edge_index_2":
            return self.x_2.size(0)
        return super().__inc__(key, value, *args, **kwargs)


@dataclass
class PairMetadata:
    mol_id_1: int
    mol_id_2: int
    smiles_1: str
    smiles_2: str
    gap_1: float
    gap_2: float
    similarity: float
    delta_gap: float


class MoleculePairDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform=None, pre_transform=None):
        super().__init__(transform=transform, pre_transform=pre_transform)
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self._graph_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        required_cols = {
            "mol_id_1",
            "mol_id_2",
            "smiles_1",
            "smiles_2",
            "gap_1",
            "gap_2",
            "similarity",
            "delta_gap",
            "label",
        }
        missing = required_cols.difference(self.df.columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in {self.csv_path}: {missing_str}")

    def len(self) -> int:
        return len(self.df)

    def _get_graph(self, smiles: str) -> Dict[str, torch.Tensor]:
        graph = self._graph_cache.get(smiles)
        if graph is None:
            graph = smiles_to_graph(smiles)
            self._graph_cache[smiles] = graph
        return graph

    def get_metadata(self, idx: int) -> PairMetadata:
        row = self.df.iloc[idx]
        return PairMetadata(
            mol_id_1=int(row["mol_id_1"]),
            mol_id_2=int(row["mol_id_2"]),
            smiles_1=str(row["smiles_1"]),
            smiles_2=str(row["smiles_2"]),
            gap_1=float(row["gap_1"]),
            gap_2=float(row["gap_2"]),
            similarity=float(row["similarity"]),
            delta_gap=float(row["delta_gap"]),
        )

    def get(self, idx: int) -> MoleculePairData:
        row = self.df.iloc[idx]
        graph_1 = self._get_graph(str(row["smiles_1"]))
        graph_2 = self._get_graph(str(row["smiles_2"]))

        data = MoleculePairData(
            x_1=graph_1["x"].clone(),
            edge_index_1=graph_1["edge_index"].clone(),
            edge_attr_1=graph_1["edge_attr"].clone(),
            x_2=graph_2["x"].clone(),
            edge_index_2=graph_2["edge_index"].clone(),
            edge_attr_2=graph_2["edge_attr"].clone(),
            y=torch.tensor([float(row["label"])], dtype=torch.float),
            similarity=torch.tensor([float(row["similarity"])], dtype=torch.float),
            delta_gap=torch.tensor([float(row["delta_gap"])], dtype=torch.float),
        )

        data.mol_id_1 = int(row["mol_id_1"])
        data.mol_id_2 = int(row["mol_id_2"])
        data.smiles_1 = str(row["smiles_1"])
        data.smiles_2 = str(row["smiles_2"])
        return data


def load_pair_splits(data_dir: str | Path = "data") -> Dict[str, MoleculePairDataset]:
    base = Path(data_dir)
    return {
        "train": MoleculePairDataset(base / "train_pairs.csv"),
        "val": MoleculePairDataset(base / "val_pairs.csv"),
        "test": MoleculePairDataset(base / "test_pairs.csv"),
    }


def describe_sample(dataset: MoleculePairDataset, idx: int = 0) -> Optional[Dict[str, int]]:
    if len(dataset) == 0:
        return None

    sample = dataset[idx]
    return {
        "nodes_1": int(sample.x_1.size(0)),
        "edges_1": int(sample.edge_index_1.size(1)),
        "nodes_2": int(sample.x_2.size(0)),
        "edges_2": int(sample.edge_index_2.size(1)),
        "label": int(sample.y.item()),
    }
