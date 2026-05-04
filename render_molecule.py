from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def load_smiles_from_dataset(csv_path: Path, row_index: int, molecule: int) -> str:
    df = pd.read_csv(csv_path)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"Row index {row_index} is outside dataset length {len(df)}")

    column = f"smiles_{molecule}"
    if column not in df.columns:
        raise ValueError(f"{csv_path} does not contain column {column}")

    return str(df.iloc[row_index][column])


def parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Size must use WIDTHxHEIGHT, for example 800x600")

    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive")

    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a molecule structure image from a SMILES string.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles", help="SMILES string to render")
    source.add_argument("--csv", type=Path, help="Pair dataset CSV to read a SMILES string from")
    parser.add_argument("--row", type=int, default=0, help="Dataset row index when using --csv")
    parser.add_argument(
        "--molecule",
        type=int,
        choices=(1, 2),
        default=1,
        help="Use smiles_1 or smiles_2 when reading from a pair dataset CSV",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .png, .jpg, or .jpeg path")
    parser.add_argument("--size", type=parse_size, default=(600, 450), help="Image size, for example 800x600")
    parser.add_argument("--legend", default="", help="Optional text shown below the molecule")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smiles = args.smiles
    if smiles is None:
        smiles = load_smiles_from_dataset(args.csv, args.row, args.molecule)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = Draw.MolToImage(mol, size=args.size, legend=args.legend or smiles)

    suffix = args.output.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image = image.convert("RGB")
        image.save(args.output, format="JPEG", quality=95)
    elif suffix == ".png":
        image.save(args.output, format="PNG")
    else:
        raise ValueError("Output path must end with .png, .jpg, or .jpeg")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
