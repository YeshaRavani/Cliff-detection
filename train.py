from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from dataset import load_pair_splits
from model import MoleculePairClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune_threshold", action="store_true")
    parser.add_argument("--threshold_metric", choices=["f1", "precision", "recall"], default="f1")
    parser.add_argument("--use_pair_features", action="store_true")
    parser.add_argument("--loss_type", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="plateau")
    parser.add_argument("--scheduler_patience", type=int, default=3)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="outputs")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalWithLogitsLoss(torch.nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - pt).pow(self.gamma)
        return (focal_weight * bce).mean()


def build_loaders(data_dir: str, batch_size: int, num_workers: int):
    splits = load_pair_splits(data_dir)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            follow_batch=["x_1", "x_2"],
        )
        for split, dataset in splits.items()
    }
    return splits, loaders


def compute_pos_weight(dataset) -> torch.Tensor:
    labels = torch.tensor(dataset.df["label"].values, dtype=torch.float)
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0:
        return torch.tensor(1.0, dtype=torch.float)
    return negatives / positives


def train_one_epoch(model, loader, optimizer, criterion, device: str, grad_clip: float | None = None) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)
        targets = batch.y.view(-1)
        loss = criterion(logits, targets)

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def collect_predictions(model, loader, device: str):
    model.eval()
    logits_all = []
    labels_all = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        logits_all.append(logits.cpu())
        labels_all.append(batch.y.view(-1).cpu())

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    probs = torch.sigmoid(logits).numpy()
    y_true = labels.numpy()
    return probs, y_true


def find_best_threshold(y_true, probs, metric_name: str = "f1"):
    best_threshold = 0.5
    best_score = float("-inf")

    for threshold in np.linspace(0.05, 0.95, 37):
        y_pred = (probs >= threshold).astype(int)
        if metric_name == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric_name == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_score


def evaluate_from_predictions(probs, y_true, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (probs >= threshold).astype(int)

    metrics = {
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    )
    return metrics


@torch.no_grad()
def evaluate(model, loader, device: str, threshold: float = 0.5) -> Dict[str, float]:
    probs, y_true = collect_predictions(model, loader, device)
    return evaluate_from_predictions(probs, y_true, threshold)


def build_criterion(args, pos_weight: torch.Tensor):
    if args.loss_type == "focal":
        return FocalWithLogitsLoss(pos_weight=pos_weight, gamma=args.focal_gamma)
    return BCEWithLogitsLoss(pos_weight=pos_weight)


def build_scheduler(args, optimizer):
    if args.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_lr,
        )
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    return None


def save_json(path: Path, payload: Dict):
    path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, loaders = build_loaders(args.data_dir, args.batch_size, args.num_workers)
    pos_weight = compute_pos_weight(loaders["train"].dataset).to(args.device)

    model = MoleculePairClassifier(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_pair_features=args.use_pair_features,
    ).to(args.device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = build_criterion(args, pos_weight)
    scheduler = build_scheduler(args, optimizer)

    best_val_pr_auc = float("-inf")
    best_state_path = out_dir / "best_model.pt"
    history = []
    best_threshold = args.threshold
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, args.device, args.grad_clip)
        val_probs, val_y_true = collect_predictions(model, loaders["val"], args.device)

        threshold = args.threshold
        if args.tune_threshold:
            threshold, _ = find_best_threshold(val_y_true, val_probs, args.threshold_metric)

        val_metrics = evaluate_from_predictions(val_probs, val_y_true, threshold)
        val_metrics["train_loss"] = float(train_loss)
        val_metrics["epoch"] = epoch
        val_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(val_metrics)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_pr_auc={val_metrics['pr_auc']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_thr={val_metrics['threshold']:.2f} | "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_metrics["pr_auc"])
            else:
                scheduler.step()

        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics["pr_auc"]
            best_threshold = val_metrics["threshold"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "best_threshold": best_threshold,
                },
                best_state_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch:03d}")
            break

    checkpoint = torch.load(best_state_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_threshold = float(checkpoint.get("best_threshold", best_threshold))
    test_metrics = evaluate(model, loaders["test"], args.device, test_threshold)

    print("\nBest validation checkpoint loaded for test evaluation.")
    print(
        f"Test PR-AUC={test_metrics['pr_auc']:.4f} | "
        f"ROC-AUC={test_metrics['roc_auc']:.4f} | "
        f"F1={test_metrics['f1']:.4f} | "
        f"Precision={test_metrics['precision']:.4f} | "
        f"Recall={test_metrics['recall']:.4f} | "
        f"Threshold={test_metrics['threshold']:.2f}"
    )

    summary = {
        "best_val_pr_auc": best_val_pr_auc,
        "best_threshold": test_threshold,
    }
    save_json(out_dir / "train_history.json", {"history": history, "summary": summary})
    save_json(out_dir / "test_metrics.json", test_metrics)
    print(f"Saved checkpoint: {best_state_path}")
    print(f"Saved history: {out_dir / 'train_history.json'}")
    print(f"Saved test metrics: {out_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
