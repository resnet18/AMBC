#!/usr/bin/env python3
"""
AMBC Experiment: Logistic Regression Baseline (Statistical Features, CPU only)
=====================================================================
Lctong1021's Stage 1 equivalent under AMBC Standard Protocol.

Usage:
    python experiments\logistic_regression\run.py
"""

import gc
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import compute_classification_metrics


def extract_stats(X):
    """X: (N, 4096, 23) -> (N, 161)"""
    return np.concatenate([
        X.mean(axis=1),
        X.std(axis=1),
        X.max(axis=1),
        X.min(axis=1),
        np.quantile(X, 0.25, axis=1),
        np.quantile(X, 0.75, axis=1),
        X[:, -1, :] - X[:, 0, :],
    ], axis=1).astype(np.float32)


def main(cfg):
    print("=" * 60)
    print("AMBC Experiment: Logistic Regression (Statistical Features)")
    print("=" * 60)

    fold_results = []
    for fold in range(5):
        fold_dir = Path(cfg.processed_dir) / f"fold_{fold}"
        X_train = np.load(fold_dir / "X_train.npy").astype(np.float32)
        y_train = np.load(fold_dir / "y_train.npy").astype(np.int64)
        X_val = np.load(fold_dir / "X_val.npy").astype(np.float32)
        y_val = np.load(fold_dir / "y_val.npy").astype(np.int64)

        X_train_f = extract_stats(X_train)
        X_val_f = extract_stats(X_val)

        scaler = StandardScaler()
        X_train_f = scaler.fit_transform(X_train_f)
        X_val_f = scaler.transform(X_val_f)

        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(X_train_f, y_train)

        y_prob = clf.predict_proba(X_val_f)[:, 1]
        y_pred = clf.predict(X_val_f)

        metrics = compute_classification_metrics(
            y_true=y_val, y_pred=y_pred, y_prob=y_prob, pos_label=1,
        )

        # Latency (CPU)
        dummy = X_val_f[:1]
        t0 = time.perf_counter()
        for _ in range(100):
            _ = clf.predict_proba(dummy)
        latency_ms = (time.perf_counter() - t0) * 1000.0 / 100

        result = {
            "fold": fold,
            "val_acc": metrics["accuracy"],
            "val_f1": metrics["f1"],
            "latency_ms": float(latency_ms),
            "n_params": int(np.prod([c.coef_.shape for c in [clf]])),
        }
        fold_results.append(result)
        print(f"[Fold {fold}] Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f}")
        
        del X_train, y_train, X_val, y_val
        gc.collect()

    accs = [r["val_acc"] for r in fold_results]
    f1s = [r["val_f1"] for r in fold_results]
    lats = [r["latency_ms"] for r in fold_results]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print("=" * 60)

    out_path = Path(cfg.output_dir) / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "lr_stats",
            "summary": {
                "mean_acc": float(np.mean(accs)),
                "std_acc": float(np.std(accs)),
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s)),
                "mean_latency_ms": float(np.mean(lats)),
            },
            "fold_results": fold_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="experiments/logistic_regression/outputs")
    args = parser.parse_args()
    main(args)