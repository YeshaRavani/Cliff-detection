from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from model import MoleculePairClassifier
from train import build_loaders, collect_predictions, evaluate_from_predictions, find_best_threshold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tune_on_val", action="store_true")
    parser.add_argument("--threshold_metric", choices=["f1", "precision", "recall"], default="f1")
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


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


def resolve_threshold(args, checkpoint_payload, model, loaders, device: str) -> float:
    if args.threshold is not None:
        return float(args.threshold)

    if args.tune_on_val:
        val_probs, val_y_true = collect_predictions(model, loaders["val"], device)
        threshold, _ = find_best_threshold(val_y_true, val_probs, args.threshold_metric)
        return float(threshold)

    if "best_threshold" in checkpoint_payload:
        return float(checkpoint_payload["best_threshold"])

    ckpt_args = checkpoint_payload.get("args", {})
    return float(ckpt_args.get("threshold", 0.5))


def plot_pr_curve(y_true, probs, out_path: Path):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion_matrix(metrics, out_path: Path):
    matrix = np.array(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4.5))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint_payload = torch.load(checkpoint_path, map_location=args.device)

    ckpt_args = checkpoint_payload.get("args", {})
    batch_size = args.batch_size if args.batch_size is not None else ckpt_args.get("batch_size", 64)
    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parent / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, loaders = build_loaders(args.data_dir, batch_size, args.num_workers)
    model = build_model_from_checkpoint(checkpoint_payload, args.device)
    threshold = resolve_threshold(args, checkpoint_payload, model, loaders, args.device)

    probs, y_true = collect_predictions(model, loaders[args.split], args.device)
    metrics = evaluate_from_predictions(probs, y_true, threshold)
    metrics["split"] = args.split
    metrics["checkpoint"] = str(checkpoint_path)

    pr_curve_path = out_dir / f"{args.split}_pr_curve.png"
    confusion_matrix_path = out_dir / f"{args.split}_confusion_matrix.png"
    metrics_path = out_dir / f"{args.split}_metrics.json"

    plot_pr_curve(y_true, probs, pr_curve_path)
    plot_confusion_matrix(metrics, confusion_matrix_path)
    save_json(metrics_path, metrics)

    print(f"Split: {args.split}")
    print(
        f"PR-AUC={metrics['pr_auc']:.4f} | "
        f"ROC-AUC={metrics['roc_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"Threshold={metrics['threshold']:.2f}"
    )
    print(
        f"Confusion Matrix: TN={metrics['tn']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TP={metrics['tp']}"
    )
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved PR curve: {pr_curve_path}")
    print(f"Saved confusion matrix: {confusion_matrix_path}")


if __name__ == "__main__":
    main()
