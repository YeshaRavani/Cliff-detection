from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


PAIR_COLUMNS = [
    "mol_id_1",
    "mol_id_2",
    "smiles_1",
    "smiles_2",
    "gap_1",
    "gap_2",
    "similarity",
    "delta_gap",
    "label",
]


def scaffold_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaffold:
        return scaffold

    # Many small QM9 molecules are acyclic and have an empty Murcko scaffold.
    # Use canonical SMILES fallback so all acyclic molecules do not collapse
    # into one unusably large scaffold group.
    return Chem.MolToSmiles(mol, canonical=True)


def load_all_pairs(data_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}_pairs.csv"
        frame = pd.read_csv(path)
        frame["source_split"] = split
        frames.append(frame)

    pairs = pd.concat(frames, ignore_index=True)
    missing = set(PAIR_COLUMNS).difference(pairs.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required pair columns: {missing_str}")

    dedup_cols = ["mol_id_1", "mol_id_2"]
    pairs["_a"] = pairs[dedup_cols].min(axis=1)
    pairs["_b"] = pairs[dedup_cols].max(axis=1)
    pairs = pairs.drop_duplicates(subset=["_a", "_b"]).drop(columns=["_a", "_b"])
    return pairs


def build_molecule_table(pairs: pd.DataFrame) -> pd.DataFrame:
    left = pairs[["mol_id_1", "smiles_1"]].rename(columns={"mol_id_1": "mol_id", "smiles_1": "smiles"})
    right = pairs[["mol_id_2", "smiles_2"]].rename(columns={"mol_id_2": "mol_id", "smiles_2": "smiles"})
    molecules = pd.concat([left, right], ignore_index=True).drop_duplicates("mol_id")
    molecules["scaffold"] = molecules["smiles"].map(scaffold_from_smiles)
    return molecules


def assign_scaffolds(molecules: pd.DataFrame, seed: int, val_frac: float, test_frac: float) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    scaffold_sizes = molecules.groupby("scaffold").size().reset_index(name="size")
    scaffold_sizes = scaffold_sizes.sample(frac=1.0, random_state=seed).sort_values("size", ascending=False)

    total = int(scaffold_sizes["size"].sum())
    targets = {
        "test": total * test_frac,
        "val": total * val_frac,
        "train": total * (1.0 - val_frac - test_frac),
    }
    current = {"train": 0, "val": 0, "test": 0}
    assignment: dict[str, str] = {}

    for _, row in scaffold_sizes.iterrows():
        scaffold = str(row["scaffold"])
        size = int(row["size"])
        candidates = ["train", "val", "test"]
        rng.shuffle(candidates)
        split = min(candidates, key=lambda item: current[item] / max(targets[item], 1.0))
        assignment[scaffold] = split
        current[split] += size

    return assignment


def make_scaffold_split(pairs: pd.DataFrame, molecules: pd.DataFrame, scaffold_to_split: dict[str, str]) -> pd.DataFrame:
    mol_to_scaffold = dict(zip(molecules["mol_id"], molecules["scaffold"]))
    mol_to_split = {mol_id: scaffold_to_split[scaffold] for mol_id, scaffold in mol_to_scaffold.items()}

    pairs = pairs.copy()
    pairs["scaffold_1"] = pairs["mol_id_1"].map(mol_to_scaffold)
    pairs["scaffold_2"] = pairs["mol_id_2"].map(mol_to_scaffold)
    pairs["split_1"] = pairs["mol_id_1"].map(mol_to_split)
    pairs["split_2"] = pairs["mol_id_2"].map(mol_to_split)
    pairs["scaffold_split"] = np.where(pairs["split_1"] == pairs["split_2"], pairs["split_1"], "cross")
    return pairs


def validate_no_scaffold_overlap(split_pairs: dict[str, pd.DataFrame]) -> None:
    split_scaffolds: dict[str, set[str]] = {}
    for split, frame in split_pairs.items():
        split_scaffolds[split] = set(frame["scaffold_1"]).union(frame["scaffold_2"])

    for left in ["train", "val", "test"]:
        for right in ["train", "val", "test"]:
            if left >= right:
                continue
            overlap = split_scaffolds[left].intersection(split_scaffolds[right])
            if overlap:
                raise RuntimeError(f"Scaffold overlap between {left} and {right}: {len(overlap)}")


def write_split_outputs(pairs: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    split_pairs = {}
    for split in ["train", "val", "test"]:
        frame = pairs[pairs["scaffold_split"] == split].copy()
        split_pairs[split] = frame
        frame[PAIR_COLUMNS].to_csv(out_dir / f"{split}_pairs.csv", index=False)

    validate_no_scaffold_overlap(split_pairs)

    summary_rows = []
    for split, frame in split_pairs.items():
        labels = frame["label"].value_counts()
        scaffolds = set(frame["scaffold_1"]).union(frame["scaffold_2"])
        molecules = set(frame["mol_id_1"]).union(frame["mol_id_2"])
        summary_rows.append(
            {
                "split": split,
                "pairs": len(frame),
                "molecules": len(molecules),
                "scaffolds": len(scaffolds),
                "non_cliff": int(labels.get(0, 0)),
                "cliff": int(labels.get(1, 0)),
                "cliff_rate": float(frame["label"].mean()) if len(frame) else 0.0,
            }
        )

    cross = pairs[pairs["scaffold_split"] == "cross"]
    summary_rows.append(
        {
            "split": "dropped_cross_split",
            "pairs": len(cross),
            "molecules": len(set(cross["mol_id_1"]).union(cross["mol_id_2"])) if len(cross) else 0,
            "scaffolds": len(set(cross["scaffold_1"]).union(cross["scaffold_2"])) if len(cross) else 0,
            "non_cliff": int(cross["label"].value_counts().get(0, 0)) if len(cross) else 0,
            "cliff": int(cross["label"].value_counts().get(1, 0)) if len(cross) else 0,
            "cliff_rate": float(cross["label"].mean()) if len(cross) else 0.0,
        }
    )
    pd.DataFrame(summary_rows).to_csv(out_dir / "scaffold_split_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create strict scaffold train/val/test pair splits.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--out_dir", type=Path, default=Path("data_scaffold"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.val_frac <= 0 or args.test_frac <= 0 or args.val_frac + args.test_frac >= 1:
        raise ValueError("--val_frac and --test_frac must be positive and sum to less than 1")

    pairs = load_all_pairs(args.data_dir)
    molecules = build_molecule_table(pairs)
    scaffold_to_split = assign_scaffolds(molecules, args.seed, args.val_frac, args.test_frac)
    scaffold_pairs = make_scaffold_split(pairs, molecules, scaffold_to_split)
    write_split_outputs(scaffold_pairs, args.out_dir)

    summary = pd.read_csv(args.out_dir / "scaffold_split_summary.csv")
    print(summary.to_string(index=False))
    print(f"Saved scaffold split CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
