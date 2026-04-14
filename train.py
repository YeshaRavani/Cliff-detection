import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import build_pair_datasets, pair_collate_fn
from model import SiameseGNN


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Siamese GNN for molecular cliff classification.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-gnn-layers", type=int, default=3)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--loss",
        type=str,
        choices=["bce", "weighted_bce"],
        default="weighted_bce",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        choices=["val_pr_auc", "val_f1"],
        default="val_pr_auc",
    )
    return parser.parse_args()


def move_batch_to_device(batch, device):
    graph1, graph2, labels = batch
    return graph1.to(device), graph2.to(device), labels.to(device)


def compute_pos_weight(labels):
    num_pos = int((labels == 1).sum().item())
    num_neg = int((labels == 0).sum().item())
    if num_pos == 0:
        raise ValueError("Training set has no positive samples; cannot compute pos_weight.")
    return torch.tensor(num_neg / num_pos, dtype=torch.float)


def build_loss(args, train_dataset):
    if args.loss == "bce":
        return nn.BCEWithLogitsLoss(), None

    train_labels = torch.tensor(train_dataset.df["label"].to_numpy(), dtype=torch.float)
    pos_weight = compute_pos_weight(train_labels)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion, pos_weight.item()


def safe_roc_auc(y_true, y_score):
    unique = len(set(y_true))
    if unique < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def compute_metrics(targets, probs, threshold=0.5):
    preds = [1 if p >= threshold else 0 for p in probs]
    return {
        "pr_auc": average_precision_score(targets, probs),
        "roc_auc": safe_roc_auc(targets, probs),
        "f1": f1_score(targets, preds, zero_division=0),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall": recall_score(targets, preds, zero_division=0),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_probs = []

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            graph1, graph2, labels = move_batch_to_device(batch, device)
            labels = labels.float()

            logits = model(graph1, graph2)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.sigmoid(logits)
            all_targets.extend(labels.detach().cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())

    metrics = compute_metrics(all_targets, all_probs)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def save_checkpoint(path, model, optimizer, epoch, metrics, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def append_metrics_row(csv_path, row):
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_pair_datasets(args.data_dir)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pair_collate_fn,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pair_collate_fn,
    )

    model = SiameseGNN(
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        dropout=args.dropout,
    ).to(args.device)

    criterion, pos_weight = build_loss(args, datasets["train"])
    if isinstance(criterion, nn.BCEWithLogitsLoss) and criterion.pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=criterion.pos_weight.to(args.device))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics_csv = output_dir / "metrics.csv"
    history_json = output_dir / "metrics_history.json"
    checkpoint_path = output_dir / "best_model.pt"

    best_score = float("-inf")
    history = []

    run_config = {
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "num_gnn_layers": args.num_gnn_layers,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "dropout": args.dropout,
        "loss": args.loss,
        "monitor": args.monitor,
        "device": args.device,
        "pos_weight": pos_weight,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, args.device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, args.device)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_pr_auc": train_metrics["pr_auc"],
            "train_roc_auc": train_metrics["roc_auc"],
            "train_f1": train_metrics["f1"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }

        history.append(epoch_metrics)
        append_metrics_row(metrics_csv, epoch_metrics)
        history_json.write_text(json.dumps(history, indent=2))

        monitor_value = epoch_metrics[args.monitor]
        if monitor_value > best_score:
            best_score = monitor_value
            save_checkpoint(checkpoint_path, model, optimizer, epoch, epoch_metrics, args)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={epoch_metrics['train_loss']:.4f} "
            f"train_pr_auc={epoch_metrics['train_pr_auc']:.4f} "
            f"train_f1={epoch_metrics['train_f1']:.4f} | "
            f"val_loss={epoch_metrics['val_loss']:.4f} "
            f"val_pr_auc={epoch_metrics['val_pr_auc']:.4f} "
            f"val_f1={epoch_metrics['val_f1']:.4f}"
        )

    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"Metrics saved to: {metrics_csv}")


if __name__ == "__main__":
    main()
