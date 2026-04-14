from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.data import Batch, Data, Dataset

from featurizer import smiles_to_graph


class MoleculePairDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        graph_cache: bool = True,
        label_dtype: torch.dtype = torch.long,
    ):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        required_columns = {"smiles_1", "smiles_2", "label"}
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in {self.csv_path}: {sorted(missing)}"
            )

        self.graph_cache = {} if graph_cache else None
        self.label_dtype = label_dtype

    def len(self) -> int:
        return len(self.df)

    def _graph_from_smiles(self, smiles: str) -> Data:
        if self.graph_cache is None:
            return smiles_to_graph(smiles)

        if smiles not in self.graph_cache:
            self.graph_cache[smiles] = smiles_to_graph(smiles)
        return self.graph_cache[smiles].clone()

    def get(self, idx: int) -> Tuple[Data, Data, torch.Tensor]:
        row = self.df.iloc[idx]
        graph1 = self._graph_from_smiles(row["smiles_1"])
        graph2 = self._graph_from_smiles(row["smiles_2"])
        label = torch.tensor(row["label"], dtype=self.label_dtype)
        return graph1, graph2, label


def pair_collate_fn(batch):
    graphs1, graphs2, labels = zip(*batch)
    batch1 = Batch.from_data_list(list(graphs1))
    batch2 = Batch.from_data_list(list(graphs2))
    labels = torch.stack(list(labels))
    return batch1, batch2, labels


def build_pair_datasets(
    data_dir: str = "data",
    dataset_cls: Callable[..., MoleculePairDataset] = MoleculePairDataset,
    **dataset_kwargs,
) -> Dict[str, MoleculePairDataset]:
    data_path = Path(data_dir)
    return {
        "train": dataset_cls(data_path / "train_pairs.csv", **dataset_kwargs),
        "val": dataset_cls(data_path / "val_pairs.csv", **dataset_kwargs),
        "test": dataset_cls(data_path / "test_pairs.csv", **dataset_kwargs),
    }
