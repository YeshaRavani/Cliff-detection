"""
compare_models.py — Side-by-side comparison of the current shared-tower model
vs. the dual-tower benchmark model.

Usage:
    # After training both models:
    python compare_models.py \
        --current_ckpt  outputs/best_model.pt \
        --benchmark_ckpt outputs_benchmark/best_model.pt

    # Optionally re-tune thresholds on the validation split:
    python compare_models.py --tune_on_val

    # Choose a specific split:
    python compare_models.py --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

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

from dataset import load_pair_splits
from model import MoleculePairClassifier
from benchmark_model import DualTowerClassifier
from torch_geometric.loader import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Compare shared-tower vs dual-tower models")
    parser.add_argument("--current_ckpt", default="outputs/best_model.pt",
                        help="Checkpoint for the current shared-tower model")
    parser.add_argument("--benchmark_ckpt", default="outputs_benchmark/best_model.pt",
                        help="Checkpoint for the dual-tower benchmark model")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tune_on_val", action="store_true",
                        help="Re-tune classification threshold on val set before evaluating")
    parser.add_argument("--threshold_metric", choices=["f1", "precision", "recall"], default="f1")
    parser.add_argument("--out_dir", default=None,
                        help="If set, saves comparison JSON here")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_loader(data_dir: str, split: str, batch_size: int, num_workers: int):
    splits = load_pair_splits(data_dir)
    return DataLoader(
        splits[split],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        follow_batch=["x_1", "x_2"],
    ), splits["val"]


def load_shared_tower(ckpt_path: str, device: str) -> torch.nn.Module:
    payload = torch.load(ckpt_path, map_location=device)
    ckpt_args = payload.get("args", {})
    model = MoleculePairClassifier(
        hidden_dim=ckpt_args.get("hidden_dim", 128),
        num_layers=ckpt_args.get("num_layers", 3),
        dropout=ckpt_args.get("dropout", 0.2),
        use_pair_features=ckpt_args.get("use_pair_features", False),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def load_dual_tower(ckpt_path: str, device: str) -> torch.nn.Module:
    payload = torch.load(ckpt_path, map_location=device)
    ckpt_args = payload.get("args", {})
    model = DualTowerClassifier(
        hidden_dim=ckpt_args.get("hidden_dim", 128),
        num_layers=ckpt_args.get("num_layers", 3),
        dropout=ckpt_args.get("dropout", 0.2),
        use_pair_features=ckpt_args.get("use_pair_features", False),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def collect_predictions(model, loader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        logits_all.append(logits.cpu())
        labels_all.append(batch.y.view(-1).cpu())
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    return torch.sigmoid(logits).numpy(), labels.numpy()


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray, metric_name: str = "f1") -> float:
    best_threshold, best_score = 0.5, float("-inf")
    for threshold in np.linspace(0.05, 0.95, 37):
        y_pred = (probs >= threshold).astype(int)
        if metric_name == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric_name == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score, best_threshold = float(score), float(threshold)
    return best_threshold


def compute_metrics(probs: np.ndarray, y_true: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (probs >= threshold).astype(int)
    metrics: Dict[str, float] = {
        "pr_auc":    float(average_precision_score(y_true, probs)),
        "roc_auc":   float(roc_auc_score(y_true, probs)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def resolve_threshold(args, payload, model, val_dataset, device: str) -> float:
    if args.tune_on_val:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, follow_batch=["x_1", "x_2"],
        )
        val_probs, val_y_true = collect_predictions(model, val_loader, device)
        return find_best_threshold(val_y_true, val_probs, args.threshold_metric)
    return float(payload.get("best_threshold", payload.get("args", {}).get("threshold", 0.5)))


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

METRIC_NAMES = ["pr_auc", "roc_auc", "f1", "precision", "recall", "threshold", "tp", "fp", "tn", "fn"]
HIGHER_IS_BETTER = {"pr_auc", "roc_auc", "f1", "precision", "recall", "tp", "tn"}


def _winner(m: str, v_cur: float, v_bench: float) -> str:
    """Return a simple indicator showing which model wins on a given metric."""
    if m not in HIGHER_IS_BETTER:
        return ""
    diff = v_bench - v_cur
    if abs(diff) < 1e-6:
        return "tie"
    return "◀ benchmark" if diff > 0 else "◀ current"


def print_comparison(current_metrics: Dict, benchmark_metrics: Dict, n_params_cur: int, n_params_bench: int):
    col_w = 16
    header_cur   = "Shared-Tower (current)"
    header_bench = "Dual-Tower (benchmark)"

    print()
    print("=" * 72)
    print("  MODEL COMPARISON")
    print("=" * 72)
    print(f"  {'Metric':<18} {header_cur:<{col_w}} {header_bench:<{col_w}}  Winner")
    print("-" * 72)

    for m in METRIC_NAMES:
        v_cur   = current_metrics.get(m, float("nan"))
        v_bench = benchmark_metrics.get(m, float("nan"))
        if isinstance(v_cur, float):
            s_cur   = f"{v_cur:.4f}"
            s_bench = f"{v_bench:.4f}"
        else:
            s_cur   = str(v_cur)
            s_bench = str(v_bench)
        winner = _winner(m, float(v_cur), float(v_bench))
        print(f"  {m:<18} {s_cur:<{col_w}} {s_bench:<{col_w}}  {winner}")

    print("-" * 72)
    print(f"  {'Parameters':<18} {n_params_cur:<{col_w},} {n_params_bench:<{col_w},}")
    print("=" * 72)
    print()

    # Delta summary for headline metrics
    print("  Δ (benchmark − current):")
    for m in ["pr_auc", "roc_auc", "f1", "precision", "recall"]:
        delta = benchmark_metrics[m] - current_metrics[m]
        sign  = "+" if delta >= 0 else ""
        print(f"    {m:<12} {sign}{delta:+.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = args.device

    print(f"\nLoading data ({args.split} split) from '{args.data_dir}' …")
    eval_loader, val_dataset = build_loader(args.data_dir, args.split, args.batch_size, args.num_workers)

    # ── load models ───────────────────────────────────────────────────────────
    print(f"Loading shared-tower model from '{args.current_ckpt}' …")
    model_cur, payload_cur   = load_shared_tower(args.current_ckpt, device)

    print(f"Loading dual-tower model    from '{args.benchmark_ckpt}' …")
    model_bench, payload_bench = load_dual_tower(args.benchmark_ckpt, device)

    n_params_cur   = sum(p.numel() for p in model_cur.parameters()   if p.requires_grad)
    n_params_bench = sum(p.numel() for p in model_bench.parameters() if p.requires_grad)

    # ── resolve thresholds ────────────────────────────────────────────────────
    thr_cur   = resolve_threshold(args, payload_cur,   model_cur,   val_dataset, device)
    thr_bench = resolve_threshold(args, payload_bench, model_bench, val_dataset, device)

    # ── run inference ─────────────────────────────────────────────────────────
    print(f"\nRunning inference on '{args.split}' split …")
    probs_cur,   y_true = collect_predictions(model_cur,   eval_loader, device)
    probs_bench, _      = collect_predictions(model_bench, eval_loader, device)

    # ── compute metrics ───────────────────────────────────────────────────────
    metrics_cur   = compute_metrics(probs_cur,   y_true, thr_cur)
    metrics_bench = compute_metrics(probs_bench, y_true, thr_bench)

    print_comparison(metrics_cur, metrics_bench, n_params_cur, n_params_bench)

    # ── optionally save comparison ────────────────────────────────────────────
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        result = {
            "split": args.split,
            "shared_tower":  metrics_cur,
            "dual_tower":    metrics_bench,
            "n_params_shared": n_params_cur,
            "n_params_dual":   n_params_bench,
        }
        save_path = out / f"{args.split}_comparison.json"
        save_path.write_text(json.dumps(result, indent=2))
        print(f"Saved comparison JSON → {save_path}")


if __name__ == "__main__":
    main()
