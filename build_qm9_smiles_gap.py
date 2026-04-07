import argparse
import csv
import random
from pathlib import Path
from rdkit import Chem


def extract_gap(line2):
    parts = line2.strip().split()

    floats = []
    for p in parts:
        try:
            floats.append(float(p))
        except:
            continue

    # From your dataset, GAP is the 3rd float after HOMO & LUMO
    # Example: ... -0.3877 0.1171 0.5048 ...
    # So gap = floats[7]
    return floats[7]


def extract_smiles_or_inchi(lines):
    smiles = None
    inchi = None

    for line in reversed(lines):
        line = line.strip()

        if not line:
            continue

        # If SMILES explicitly present
        if line.startswith("SMILES"):
            smiles = line.split()[-1]
            break

        # If InChI present
        if "InChI=" in line:
            parts = line.split()
            for p in parts:
                if p.startswith("InChI="):
                    inchi = p
                    break
            break

    if smiles:
        return smiles

    if inchi:
        mol = Chem.MolFromInchi(inchi)
        if mol:
            return Chem.MolToSmiles(mol)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--out", default="qm9_smiles_gap_50k.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = list(data_dir.glob("qm9_*.xyz"))

    random.seed(args.seed)
    if len(files) > args.n:
        files = random.sample(files, args.n)

    rows = []

    for file in files:
        try:
            lines = file.read_text().splitlines()
            gap = extract_gap(lines[1])
            smiles = extract_smiles_or_inchi(lines)

            if smiles is None:
                continue

            rows.append([smiles, gap])

        except:
            continue

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "gap"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} molecules to {args.out}")


if __name__ == "__main__":
    main()
