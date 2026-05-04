from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from featurizer import smiles_to_graph


NODE_FEATURES = [
    "atom_type_index",
    "atomic_num",
    "degree",
    "formal_charge",
    "total_hs",
    "is_aromatic",
    "is_in_ring",
    "hybridization",
    "chirality",
]

BOND_FEATURES = [
    "bond_type",
    "is_conjugated",
    "is_in_ring",
    "stereo",
]


def tensor_to_list(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def graph_to_payload(smiles: str, graph: dict[str, torch.Tensor], include_names: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "smiles": smiles,
        "num_nodes": int(graph["x"].size(0)),
        "num_directed_edges": int(graph["edge_index"].size(1)),
        "x": tensor_to_list(graph["x"]),
        "edge_index": tensor_to_list(graph["edge_index"]),
        "edge_attr": tensor_to_list(graph["edge_attr"]),
    }
    if include_names:
        payload["node_feature_names"] = NODE_FEATURES
        payload["edge_feature_names"] = BOND_FEATURES
    return payload


def load_smiles_from_dataset(csv_path: Path, row_index: int, molecule: int) -> str:
    df = pd.read_csv(csv_path)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"Row index {row_index} is outside dataset length {len(df)}")

    column = f"smiles_{molecule}"
    if column not in df.columns:
        raise ValueError(f"{csv_path} does not contain column {column}")

    return str(df.iloc[row_index][column])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the intermediate graph representation produced from a SMILES string."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles", help="SMILES string to convert into a graph")
    source.add_argument("--csv", type=Path, help="Pair dataset CSV to read a SMILES string from")
    parser.add_argument("--row", type=int, default=0, help="Dataset row index when using --csv")
    parser.add_argument(
        "--molecule",
        type=int,
        choices=(1, 2),
        default=1,
        help="Use smiles_1 or smiles_2 when reading from a pair dataset CSV",
    )
    parser.add_argument("--output", type=Path, help="Optional path to write JSON output")
    parser.add_argument("--no-feature-names", action="store_true", help="Omit feature-name metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smiles = args.smiles
    if smiles is None:
        smiles = load_smiles_from_dataset(args.csv, args.row, args.molecule)

    graph = smiles_to_graph(smiles)
    payload = graph_to_payload(smiles, graph, include_names=not args.no_feature_names)
    output = json.dumps(payload, indent=2)

    if args.output:
        args.output.write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
