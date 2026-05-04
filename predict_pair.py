from __future__ import annotations

import argparse

import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from featurizer import smiles_to_graph
from model import MoleculePairClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_1", required=True, help="SMILES for the first molecule")
    parser.add_argument("--smiles_2", required=True, help="SMILES for the second molecule")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def compute_similarity(smiles_1: str, smiles_2: str) -> float | None:
    mol_1 = Chem.MolFromSmiles(smiles_1)
    mol_2 = Chem.MolFromSmiles(smiles_2)
    if mol_1 is None or mol_2 is None:
        return None

    fp_1 = AllChem.GetMorganFingerprintAsBitVect(mol_1, radius=2, nBits=2048)
    fp_2 = AllChem.GetMorganFingerprintAsBitVect(mol_2, radius=2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fp_1, fp_2))


def build_pair_data(smiles_1: str, smiles_2: str) -> Data:
    graph_1 = smiles_to_graph(smiles_1)
    graph_2 = smiles_to_graph(smiles_2)

    return Data(
        x_1=graph_1["x"],
        edge_index_1=graph_1["edge_index"],
        edge_attr_1=graph_1["edge_attr"],
        x_2=graph_2["x"],
        edge_index_2=graph_2["edge_index"],
        edge_attr_2=graph_2["edge_attr"],
        x_1_batch=torch.zeros(graph_1["x"].size(0), dtype=torch.long),
        x_2_batch=torch.zeros(graph_2["x"].size(0), dtype=torch.long),
        similarity=torch.tensor([compute_similarity(smiles_1, smiles_2) or 0.0], dtype=torch.float),
    )


def build_model_from_checkpoint(checkpoint_payload, device: str):
    ckpt_args = checkpoint_payload.get("args", {})
    model = MoleculePairClassifier(
        hidden_dim=ckpt_args.get("hidden_dim", 128),
        num_layers=ckpt_args.get("num_layers", 3),
        dropout=ckpt_args.get("dropout", 0.2),
        use_pair_features=ckpt_args.get("use_pair_features", False),
    ).to(device)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.eval()
    return model


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    threshold = float(args.threshold) if args.threshold is not None else float(checkpoint.get("best_threshold", 0.5))

    pair_data = build_pair_data(args.smiles_1, args.smiles_2).to(args.device)
    model = build_model_from_checkpoint(checkpoint, args.device)

    with torch.no_grad():
        logit = model(pair_data).item()
        probability = float(torch.sigmoid(torch.tensor(logit)).item())

    prediction = 1 if probability >= threshold else 0
    label_name = "cliff" if prediction == 1 else "non-cliff"
    similarity = compute_similarity(args.smiles_1, args.smiles_2)

    print(f"SMILES 1: {args.smiles_1}")
    print(f"SMILES 2: {args.smiles_2}")
    if similarity is not None:
        print(f"Tanimoto similarity: {similarity:.4f}")
    print(f"Cliff probability: {probability:.4f}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Prediction: {prediction} ({label_name})")


if __name__ == "__main__":
    main()
