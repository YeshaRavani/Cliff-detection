import pandas as pd
import heapq
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

INPUT = "qm9_master_dataset.csv"
OUTPUT = "qm9_pair_candidates.csv"

TOP_K = 20
FP_RADIUS = 2
FP_BITS = 2048
CHUNK_SIZE = 5000  # tune if needed


def add_to_topk(heap, sim, j, k):
    """
    Maintain a min-heap of size <= k.
    Stores tuples (similarity, neighbor_index).
    """
    item = (sim, j)
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        if sim > heap[0][0]:
            heapq.heapreplace(heap, item)


def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT).copy()

    # Ensure mol_id exists and is aligned with row index
    if "mol_id" not in df.columns:
        df["mol_id"] = range(len(df))

    smiles = df["smiles"].tolist()
    gaps = df["gap_raw"].tolist()
    mol_ids = df["mol_id"].tolist()

    print("Building RDKit molecules...")
    mols = []
    valid_rows = []

    for idx, smi in enumerate(tqdm(smiles)):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_rows.append(idx)

    # If any invalid rows exist, filter dataframe
    if len(valid_rows) != len(df):
        print(f"Dropping {len(df) - len(valid_rows)} invalid molecules")
        df = df.iloc[valid_rows].reset_index(drop=True)
        smiles = df["smiles"].tolist()
        gaps = df["gap_raw"].tolist()
        mol_ids = df["mol_id"].tolist()
        mols = [Chem.MolFromSmiles(s) for s in smiles]

    n = len(mols)
    print(f"Valid molecules: {n}")

    print("Computing Morgan fingerprints...")
    fps = []
    for mol in tqdm(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_BITS)
        fps.append(fp)

    # One top-K heap per molecule
    topk_heaps = [[] for _ in range(n)]

    print("Computing exact top-K neighbors using BulkTanimotoSimilarity...")
    for i in tqdm(range(n)):
        fp_i = fps[i]

        # Compare only with j > i to avoid duplicate work
        for start in range(i + 1, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            sims = DataStructs.BulkTanimotoSimilarity(fp_i, fps[start:end])

            for offset, sim in enumerate(sims):
                j = start + offset

                # Update i's heap
                add_to_topk(topk_heaps[i], sim, j, TOP_K)

                # Update j's heap
                add_to_topk(topk_heaps[j], sim, i, TOP_K)

    print("Collecting unique candidate pairs...")
    pair_dict = {}

    for i in range(n):
        for sim, j in topk_heaps[i]:
            a, b = sorted((i, j))
            key = (a, b)

            # keep one unique entry
            if key not in pair_dict or sim > pair_dict[key]["similarity"]:
                pair_dict[key] = {
                    "mol_id_a": int(mol_ids[a]),
                    "mol_id_b": int(mol_ids[b]),
                    "smiles_a": smiles[a],
                    "smiles_b": smiles[b],
                    "similarity": float(sim),
                    "gap_a": float(gaps[a]),
                    "gap_b": float(gaps[b]),
                    "delta_gap": float(abs(gaps[a] - gaps[b]))
                }

    pairs_df = pd.DataFrame(pair_dict.values())

    # Sort for convenience
    pairs_df = pairs_df.sort_values(
        by=["similarity", "delta_gap"],
        ascending=[False, False]
    ).reset_index(drop=True)

    pairs_df.to_csv(OUTPUT, index=False)

    print(f"Saved: {OUTPUT}")
    print(f"Unique candidate pairs: {len(pairs_df)}")
    print(pairs_df.head())


if __name__ == "__main__":
    main()