import pandas as pd
import numpy as np
from collections import Counter

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import matplotlib.pyplot as plt


INFILE = "qm9_smiles_gap_50k.csv"
OUTCSV = "qm9_master_dataset.csv"


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def mol_features_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # invalid SMILES

    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())

    num_atoms = mol.GetNumAtoms()
    num_heavy_atoms = mol.GetNumHeavyAtoms()
    total_bonds = len(bonds)

    # --- Atom-level derived (aggregated) ---
    atomic_nums = [a.GetAtomicNum() for a in atoms]
    degrees = [a.GetDegree() for a in atoms]
    formal_charges = [a.GetFormalCharge() for a in atoms]
    total_hs = [a.GetTotalNumHs(includeNeighbors=True) for a in atoms]
    aromatic_flags = [1 if a.GetIsAromatic() else 0 for a in atoms]
    ring_atom_flags = [1 if a.IsInRing() else 0 for a in atoms]

    num_aromatic_atoms = int(sum(aromatic_flags))
    num_ring_atoms = int(sum(ring_atom_flags))

    avg_atomic_number = float(np.mean(atomic_nums)) if num_atoms else 0.0
    avg_degree = float(np.mean(degrees)) if num_atoms else 0.0
    avg_formal_charge = float(np.mean(formal_charges)) if num_atoms else 0.0
    avg_num_hydrogens = float(np.mean(total_hs)) if num_atoms else 0.0

    # Hybridization counts
    hyb_counts = Counter()
    for a in atoms:
        hyb = str(a.GetHybridization())  # e.g., 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED'
        hyb_counts[hyb] += 1

    num_sp = int(hyb_counts.get("SP", 0))
    num_sp2 = int(hyb_counts.get("SP2", 0))
    num_sp3 = int(hyb_counts.get("SP3", 0))
    num_sp3d = int(hyb_counts.get("SP3D", 0))
    num_sp3d2 = int(hyb_counts.get("SP3D2", 0))
    num_hyb_other = int(num_atoms - (num_sp + num_sp2 + num_sp3 + num_sp3d + num_sp3d2))

    # --- Bond-level derived (aggregated) ---
    bond_type_counts = Counter()
    num_conjugated_bonds = 0
    num_ring_bonds = 0

    for b in bonds:
        btype = str(b.GetBondType())  # 'SINGLE','DOUBLE','TRIPLE','AROMATIC'
        bond_type_counts[btype] += 1
        if b.GetIsConjugated():
            num_conjugated_bonds += 1
        if b.IsInRing():
            num_ring_bonds += 1

    num_single_bonds = int(bond_type_counts.get("SINGLE", 0))
    num_double_bonds = int(bond_type_counts.get("DOUBLE", 0))
    num_triple_bonds = int(bond_type_counts.get("TRIPLE", 0))
    num_aromatic_bonds = int(bond_type_counts.get("AROMATIC", 0))

    # Ring count (SSSR) – common simple ring measure
    num_rings = int(rdMolDescriptors.CalcNumRings(mol))

    # --- Ratio features (often more informative than raw counts) ---
    aromatic_atom_ratio = safe_div(num_aromatic_atoms, num_atoms)
    ring_atom_ratio = safe_div(num_ring_atoms, num_atoms)

    conjugation_ratio = safe_div(num_conjugated_bonds, total_bonds)
    ring_bond_ratio = safe_div(num_ring_bonds, total_bonds)
    aromatic_bond_ratio = safe_div(num_aromatic_bonds, total_bonds)

    return {
        # size/topology
        "num_atoms": num_atoms,
        "num_heavy_atoms": num_heavy_atoms,
        "total_bonds": total_bonds,
        "num_rings": num_rings,

        # atomic aggregates
        "num_aromatic_atoms": num_aromatic_atoms,
        "num_ring_atoms": num_ring_atoms,
        "avg_atomic_number": avg_atomic_number,
        "avg_degree": avg_degree,
        "avg_formal_charge": avg_formal_charge,
        "avg_num_hydrogens": avg_num_hydrogens,

        # hybridization counts
        "num_sp": num_sp,
        "num_sp2": num_sp2,
        "num_sp3": num_sp3,
        "num_sp3d": num_sp3d,
        "num_sp3d2": num_sp3d2,
        "num_hyb_other": num_hyb_other,

        # bond aggregates
        "num_single_bonds": num_single_bonds,
        "num_double_bonds": num_double_bonds,
        "num_triple_bonds": num_triple_bonds,
        "num_aromatic_bonds": num_aromatic_bonds,
        "num_conjugated_bonds": int(num_conjugated_bonds),
        "num_ring_bonds": int(num_ring_bonds),

        # ratios
        "aromatic_atom_ratio": aromatic_atom_ratio,
        "ring_atom_ratio": ring_atom_ratio,
        "conjugation_ratio": conjugation_ratio,
        "ring_bond_ratio": ring_bond_ratio,
        "aromatic_bond_ratio": aromatic_bond_ratio,
    }


def main():
    df = pd.read_csv(INFILE)
    if not {"smiles", "gap"}.issubset(df.columns):
        raise ValueError(f"Expected columns ['smiles','gap'] in {INFILE}, got {list(df.columns)}")

    # Standardize gap (target preprocessing)
    df = df.copy()
    df.rename(columns={"gap": "gap_raw"}, inplace=True)

    mean_gap = df["gap_raw"].mean()
    std_gap = df["gap_raw"].std()
    df["gap_std"] = (df["gap_raw"] - mean_gap) / std_gap

    # Compute features
    feats = []
    bad = 0
    for smi in df["smiles"]:
        f = mol_features_from_smiles(smi)
        if f is None:
            bad += 1
            feats.append(None)
        else:
            feats.append(f)

    feats_df = pd.DataFrame([f if f is not None else {} for f in feats])
    out = pd.concat([df[["smiles", "gap_raw", "gap_std"]], feats_df], axis=1)

    # Drop invalid rows if any
    if bad > 0:
        out = out.dropna(subset=["num_atoms"])
        print(f"Warning: dropped {bad} invalid SMILES rows")

    out.to_csv(OUTCSV, index=False)
    print(f"Saved: {OUTCSV}")
    print("Shape:", out.shape)

    # --- Quick plots for Part 3 ---
    # 1) Gap distribution
    plt.figure()
    plt.hist(out["gap_raw"], bins=50)
    plt.title("HOMO–LUMO Gap Distribution (raw)")
    plt.xlabel("gap_raw")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("plot_gap_raw_hist.png", dpi=200)

    # 2) Atom count distribution
    plt.figure()
    plt.hist(out["num_atoms"], bins=20)
    plt.title("Atoms per Molecule")
    plt.xlabel("num_atoms")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("plot_num_atoms_hist.png", dpi=200)

    # 3) Aromatic ratio distribution
    plt.figure()
    plt.hist(out["aromatic_atom_ratio"], bins=30)
    plt.title("Aromatic Atom Ratio")
    plt.xlabel("aromatic_atom_ratio")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("plot_aromatic_ratio_hist.png", dpi=200)

    # 4) Simple correlations with target (top 10 abs corr)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    corr = out[numeric_cols].corr(numeric_only=True)["gap_raw"].drop("gap_raw").sort_values(key=lambda s: s.abs(), ascending=False)
    corr.head(10).to_csv("top10_corr_with_gap_raw.csv")
    print("Saved plots: plot_gap_raw_hist.png, plot_num_atoms_hist.png, plot_aromatic_ratio_hist.png")
    print("Saved top correlations: top10_corr_with_gap_raw.csv")


if __name__ == "__main__":
    main()