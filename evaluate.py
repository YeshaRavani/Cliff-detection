import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import build_pair_datasets, pair_collate_fn
from model import SiameseGNN


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Siamese GNN checkpoint.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pt")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def safe_roc_auc(y_true, y_score):
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def compute_metrics(targets, probs, threshold):
    preds = [1 if p >= threshold else 0 for p in probs]
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    return {
        "pr_auc": average_precision_score(targets, probs),
        "roc_auc": safe_roc_auc(targets, probs),
        "f1": f1_score(targets, preds, zero_division=0),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall": recall_score(targets, preds, zero_division=0),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    model = SiameseGNN(
        hidden_dim=ckpt_args.get("hidden_dim", 128),
        num_gnn_layers=ckpt_args.get("num_gnn_layers", 3),
        mlp_hidden_dim=ckpt_args.get("mlp_hidden_dim", 256),
        dropout=ckpt_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate(model, loader, device, threshold):
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for graph1, graph2, labels in loader:
            graph1 = graph1.to(device)
            graph2 = graph2.to(device)
            labels = labels.to(device).float()

            logits = model(graph1, graph2)
            probs = torch.sigmoid(logits)

            all_targets.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return compute_metrics(all_targets, all_probs, threshold)


def main():
    args = parse_args()

    datasets = build_pair_datasets(args.data_dir)
    loader = DataLoader(
        datasets[args.split],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pair_collate_fn,
    )

    model, checkpoint = load_model_from_checkpoint(args.checkpoint, args.device)
    metrics = evaluate(model, loader, args.device, args.threshold)

    result = {
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint)),
        "threshold": args.threshold,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "metrics": metrics,
    }

    print(f"Split: {result['split']}")
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Epoch: {result['checkpoint_epoch']}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print("Confusion Matrix:")
    print(
        f"TN={metrics['confusion_matrix']['tn']} "
        f"FP={metrics['confusion_matrix']['fp']} "
        f"FN={metrics['confusion_matrix']['fn']} "
        f"TP={metrics['confusion_matrix']['tp']}"
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(result, indent=2))
        print(f"Saved evaluation to: {output_path}")


if __name__ == "__main__":
    main()
