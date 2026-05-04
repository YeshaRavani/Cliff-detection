from __future__ import annotations

from typing import Dict, List

import torch
from rdkit import Chem


ATOM_TYPES: List[int] = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
HYBRIDIZATION_MAP: Dict[Chem.rdchem.HybridizationType, int] = {
    Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
    Chem.rdchem.HybridizationType.S: 1,
    Chem.rdchem.HybridizationType.SP: 2,
    Chem.rdchem.HybridizationType.SP2: 3,
    Chem.rdchem.HybridizationType.SP3: 4,
    Chem.rdchem.HybridizationType.SP3D: 5,
    Chem.rdchem.HybridizationType.SP3D2: 6,
}
BOND_TYPE_MAP: Dict[Chem.rdchem.BondType, int] = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
STEREO_MAP: Dict[Chem.rdchem.BondStereo, int] = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ: 2,
    Chem.rdchem.BondStereo.STEREOE: 3,
    Chem.rdchem.BondStereo.STEREOCIS: 4,
    Chem.rdchem.BondStereo.STEREOTRANS: 5,
}
CHIRALITY_MAP: Dict[Chem.rdchem.ChiralType, int] = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER: 3,
}


def _atom_type_index(atomic_num: int) -> int:
    try:
        return ATOM_TYPES.index(atomic_num)
    except ValueError:
        return len(ATOM_TYPES)


def atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    return [
        float(_atom_type_index(atom.GetAtomicNum())),
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs(includeNeighbors=True)),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(HYBRIDIZATION_MAP.get(atom.GetHybridization(), 0)),
        float(CHIRALITY_MAP.get(atom.GetChiralTag(), 0)),
    ]


def bond_features(bond: Chem.rdchem.Bond) -> List[float]:
    return [
        float(BOND_TYPE_MAP.get(bond.GetBondType(), 0)),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(STEREO_MAP.get(bond.GetStereo(), 0)),
    ]


def smiles_to_graph(smiles: str) -> Dict[str, torch.Tensor]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feats = bond_features(bond)

        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(feats)
        edge_attrs.append(feats)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }
