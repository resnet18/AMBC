#!/usr/bin/env python3
"""
AMBC Experiment: MiniRocket Baseline
=====================================
Standard Protocol reproduction of the MiniRocket baseline from
Yang et al., "A Large-Scale Annotated Multivariate Time Series Aviation
Maintenance Dataset from the NGAFID".

Input:  data/processed/fold_{i}/X_{train,val}.npy
Output: experiments/minirocket/outputs/results.json

Notes:
- Pure CPU experiment. No GPU required.
- Uses sktime MiniRocketMultivariate + RidgeClassifierCV.
- Inference time measured on CPU (per-flight average).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

# sktime MiniRocket
from sktime.transformations.panel.rocket import MiniRocketMultivariate  # type: ignore

# Import AMBC unified metrics
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import compute_classification_metrics


def load_fold(processed_dir: str, fold: int):
    """Load a single fold's train/val split."""
    fold_dir = Path(processed_dir) / f"fold_{fold}"
    X_train = np.load(fold_dir / "X_train.npy").astype(np.float32)  # (N, 4096, 23)
    y_train = np.load(fold_dir / "y_train.npy").astype(np.int64)
    X_val = np.load(fold_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(fold_dir / "y_val.npy").astype(np.int64)
    return X_train, y_train, X_val, y_val


def main(cfg):
    processed_dir = Path(cfg.processed_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AMBC Experiment: MiniRocket Baseline")
    print("=" * 60)
    print(f"Processed dir: {processed_dir}")
    print(f"Output dir:    {output_dir}")
    print("=" * 60)

    fold_results = []

    for fold in range(5):
        print(f"\n[Fold {fold}] Loading data...")
        X_train, y_train, X_val, y_val = load_fold(processed_dir, fold)
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

        # --- Feature extraction: MiniRocket ---
        print(f"[Fold {fold}] Fitting MiniRocket...")
        rocket = MiniRocketMultivariate(
            num_kernels=cfg.num_kernels,
            max_dilations_per_kernel=cfg.max_dilations,
            n_jobs=cfg.n_jobs,
        )
        # sktime expects (N, T, C) which matches our .npy format
        X_train_features = rocket.fit_transform(X_train)
        X_val_features = rocket.transform(X_val)
        print(f"  Feature shape: {X_train_features.shape}")

        # --- Classifier: RidgeClassifierCV ---
        print(f"[Fold {fold}] Training RidgeClassifier...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        X_val_scaled = scaler.transform(X_val_features)

        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        clf.fit(X_train_scaled, y_train)

        # --- Validation ---
        y_pred = clf.predict(X_val_scaled)
        # RidgeClassifierCV has no predict_proba; use decision_function
        try:
            y_scores = clf.decision_function(X_val_scaled)
            # Normalize scores to [0,1] roughly for AUC
            y_prob = 1.0 / (1.0 + np.exp(-y_scores))
        except Exception:
            y_prob = None

        metrics = compute_classification_metrics(
            y_true=y_val,
            y_pred=y_pred,
            y_prob=y_prob,
            pos_label=1,
        )

        # --- Efficiency: per-flight latency ---
        # Measure full pipeline: transform + predict on single sample
        print(f"[Fold {fold}] Measuring inference latency...")
        dummy_single = X_val[:1]
        for _ in range(10):  # warmup
            _ = rocket.transform(dummy_single)
            _ = clf.predict(scaler.transform(_))

        t0 = time.perf_counter()
        for _ in range(100):
            _ = rocket.transform(dummy_single)
            _ = clf.predict(scaler.transform(_))
        latency_ms = (time.perf_counter() - t0) * 1000.0 / 100.0

        # --- Model stats ---
        n_params = clf.coef_.size + clf.intercept_.size if hasattr(clf, "coef_") else 0

        result = {
            "fold": fold,
            "val_acc": metrics["accuracy"],
            "val_f1": metrics["f1"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_auc": metrics["auc"],
            "latency_ms": float(latency_ms),
            "peak_memory_mb": None,  # CPU-only
            "n_params": int(n_params),
        }
        fold_results.append(result)

        print(f"[Fold {fold}] Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
              f"Latency={latency_ms:.3f}ms")

    # --- Summary ---
    accs = [r["val_acc"] for r in fold_results]
    f1s = [r["val_f1"] for r in fold_results]
    lats = [r["latency_ms"] for r in fold_results]

    summary = {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "mean_latency_ms": float(np.mean(lats)),
        "std_latency_ms": float(np.std(lats)),
        "mean_peak_memory_mb": None,
        "n_params": fold_results[0]["n_params"],
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Accuracy: {summary['mean_acc']:.4f} ± {summary['std_acc']:.4f}")
    print(f"F1:       {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"Latency:  {summary['mean_latency_ms']:.3f} ± {summary['std_latency_ms']:.3f} ms/flight")
    print("=" * 60)

    # --- Save results ---
    results = {
        "model": "minirocket",
        "protocol": "AMBC-Standard-v1.0",
        "config": {
            "num_kernels": cfg.num_kernels,
            "max_dilations": cfg.max_dilations,
            "classifier": "RidgeClassifierCV",
        },
        "fold_results": fold_results,
        "summary": summary,
    }

    out_path = output_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMBC MiniRocket Baseline")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="experiments/minirocket/outputs")
    parser.add_argument("--num_kernels", type=int, default=10000, help="MiniRocket num_kernels")
    parser.add_argument("--max_dilations", type=int, default=32, help="Max dilations per kernel")
    parser.add_argument("--n_jobs", type=int, default=-1, help="sktime n_jobs")
    args = parser.parse_args()
    main(args)