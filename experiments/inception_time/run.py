#!/usr/bin/env python3
"""
AMBC Experiment: InceptionTime Baseline (PyTorch)
================================================
PyTorch implementation via tsai, avoiding TF/cuDNN compatibility issues.

Usage:
    python experiments/inception_time/run.py --epochs 10

Key Arguments:
    --epochs      Training epochs. Default: 200 (matches official TF notebook).
    --lr          Learning rate. Default: 1e-4
    --batch_size  Default: 128

Examples:
    # Quick sanity check (10 epochs, ~2 minutes)
    python experiments/inception_time/run.py --epochs 10

    # Full convergence (200 epochs, official TF setting)
    python experiments/inception_time/run.py
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tsai.models.InceptionTime import InceptionTime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import compute_classification_metrics


def load_fold(processed_dir: str, fold: int):
    fold_dir = Path(processed_dir) / f"fold_{fold}"
    X_train = np.load(fold_dir / "X_train.npy").astype(np.float32)
    y_train = np.load(fold_dir / "y_train.npy").astype(np.int64)
    X_val = np.load(fold_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(fold_dir / "y_val.npy").astype(np.int64)
    X_train = np.transpose(X_train, (0, 2, 1))  # (N, 23, 4096)
    X_val = np.transpose(X_val, (0, 2, 1))
    return X_train, y_train, X_val, y_val


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("AMBC Experiment: InceptionTime (PyTorch)")
    print("=" * 60)
    print(f"Device: {device}")
    print("=" * 60)

    fold_results = []

    for fold in range(5):
        print(f"\n[Fold {fold}] Loading data...")
        X_train, y_train, X_val, y_val = load_fold(cfg.processed_dir, fold)
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

        model = InceptionTime(
            c_in=23, c_out=2, seq_len=4096,
            nf=32, ks=40, bottleneck=True,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val)
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size)

        print(f"[Fold {fold}] Training {cfg.epochs} epochs...")
        best_val_acc = 0.0
        best_state = None

        for epoch in range(cfg.epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    preds = logits.argmax(dim=1)
                    all_preds.append(preds.cpu())
                    all_labels.append(yb)

            y_pred = torch.cat(all_preds).numpy()
            y_true = torch.cat(all_labels).numpy()
            val_acc = (y_pred == y_true).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best
        if best_state:
            model.load_state_dict(best_state)
            model.to(device)

        # Final eval with probabilities
        model.eval()
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)
                all_preds.append(logits.argmax(dim=1).cpu())
                all_probs.append(probs[:, 1].cpu())

        y_pred = torch.cat(all_preds).numpy()
        y_prob = torch.cat(all_probs).numpy()

        metrics = compute_classification_metrics(
            y_true=y_val, y_pred=y_pred, y_prob=y_prob, pos_label=1,
        )

        # Latency
        print(f"[Fold {fold}] Measuring latency...")
        dummy = torch.from_numpy(X_val[:1]).float().to(device)
        for _ in range(10):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(cfg.n_repeat):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0 / cfg.n_repeat

        peak_mem = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            _ = model(dummy)
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

        n_params = sum(p.numel() for p in model.parameters())

        result = {
            "fold": fold,
            "val_acc": metrics["accuracy"],
            "val_f1": metrics["f1"],
            "latency_ms": float(latency_ms),
            "peak_memory_mb": float(peak_mem) if peak_mem else None,
            "n_params": int(n_params),
        }
        fold_results.append(result)
        print(f"[Fold {fold}] Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
              f"Latency={latency_ms:.3f}ms")

    accs = [r["val_acc"] for r in fold_results]
    f1s = [r["val_f1"] for r in fold_results]
    lats = [r["latency_ms"] for r in fold_results]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Latency:  {np.mean(lats):.3f} ± {np.std(lats):.3f} ms/flight")
    print("=" * 60)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "inceptiontime",
            "summary": {
                "mean_acc": float(np.mean(accs)),
                "std_acc": float(np.std(accs)),
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s)),
                "mean_latency_ms": float(np.mean(lats)),
                "std_latency_ms": float(np.std(lats)),
            },
            "fold_results": fold_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="experiments/inception_time/outputs")
    parser.add_argument("--epochs", type=int, default=200, help="Official setting: 200 epochs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_repeat", type=int, default=100)
    args = parser.parse_args()
    main(args)