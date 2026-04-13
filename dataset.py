from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, HybridizationType
from torch_geometric.data import Batch, Data, Dataset


HYBRIDIZATION_TO_INT = {
    HybridizationType.UNSPECIFIED: 0,
    HybridizationType.S: 1,
    HybridizationType.SP: 2,
    HybridizationType.SP2: 3,
    HybridizationType.SP3: 4,
    HybridizationType.SP3D: 5,
    HybridizationType.SP3D2: 6,
}

BOND_TYPE_TO_FLOAT = {
    BondType.SINGLE: 1.0,
    BondType.DOUBLE: 2.0,
    BondType.TRIPLE: 3.0,
    BondType.AROMATIC: 1.5,
}


def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(
            [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                int(atom.GetHybridization() in HYBRIDIZATION_TO_INT),
                HYBRIDIZATION_TO_INT.get(atom.GetHybridization(), 0),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
            ]
        )

    x = torch.tensor(atom_features, dtype=torch.float)

    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        features = [
            BOND_TYPE_TO_FLOAT.get(bond.GetBondType(), 0.0),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            float(bond.GetStereo()),
        ]

        edge_indices.extend([[i, j], [j, i]])
        edge_features.extend([features, features])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


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
