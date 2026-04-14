import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, HybridizationType
from torch_geometric.data import Data


HYBRIDIZATION_TO_INT = {
    HybridizationType.UNSPECIFIED: 0,
    HybridizationType.S: 1,
    HybridizationType.SP: 2,
    HybridizationType.SP2: 3,
    HybridizationType.SP3: 4,
    HybridizationType.SP3D: 5,
    HybridizationType.SP3D2: 6,
}

BOND_TYPE_TO_INT = {
    BondType.SINGLE: 0,
    BondType.DOUBLE: 1,
    BondType.TRIPLE: 2,
    BondType.AROMATIC: 3,
}


def atom_to_feature_vector(atom) -> list[int]:
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        HYBRIDIZATION_TO_INT.get(atom.GetHybridization(), 0),
        atom.GetTotalNumHs(),
    ]


def bond_to_feature_vector(bond) -> list[int]:
    return [
        BOND_TYPE_TO_INT.get(bond.GetBondType(), -1),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]


def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    atom_features = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_features = bond_to_feature_vector(bond)

        edge_indices.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
        edge_features.extend([bond_features, bond_features])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
