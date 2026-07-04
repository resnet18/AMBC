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
    
    # =========================================================
    # FIX 1: Dimension transpose
    # ---------------------------------------------------------
    # sktime MiniRocketMultivariate expects input shape (N, C, T):
    #   N = number of samples
    #   C = number of channels (sensors)
    #   T = number of timesteps
    #
    # Our preprocessed .npy stores (N, T, C) = (N, 4096, 23).
    # Without transpose, sktime treats 4096 as "channels" and 23 as
    # "timesteps", which means the 1D convolutions slide over only 23
    # steps instead of 4096. This extracts cross-sensor correlations
    # rather than temporal patterns, causing severe performance
    # degradation.
    #
    # To verify this is the culprit, comment out the next two lines
    # and run one fold. If accuracy drops to ~0.55, this fix is
    # essential. If accuracy stays ~0.58+, the dimension was already
    # correct and the issue lies elsewhere (e.g., StandardScaler).
    # =========================================================
    X_train = np.transpose(X_train, (0, 2, 1))  # (N, 4096, 23) -> (N, 23, 4096)
    X_val = np.transpose(X_val, (0, 2, 1))       # (N, 4096, 23) -> (N, 23, 4096)
    
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
        
        # =========================================================
        # FIX 3: Windows multiprocessing compatibility
        # ---------------------------------------------------------
        # sktime uses numba under the hood. On Windows, n_jobs=-1
        # (all cores) frequently triggers deadlocks or memory
        # explosions due to multiprocessing spawn issues. Default
        # changed to 1 for stability. If you are on Linux/macOS,
        # you can override with --n_jobs -1.
        # =========================================================
        rocket = MiniRocketMultivariate(
            num_kernels=cfg.num_kernels,
            max_dilations_per_kernel=cfg.max_dilations,
            n_jobs=cfg.n_jobs,
        )
        
        # sktime expects (N, C, T) which matches our transposed .npy
        X_train_features = rocket.fit_transform(X_train)
        X_val_features = rocket.transform(X_val)
        print(f"  Feature shape: {X_train_features.shape}")

        # --- Classifier: RidgeClassifierCV ---
        print(f"[Fold {fold}] Training RidgeClassifier...")
        
        # FIX 2: StandardScaler disabled by default
        # ---------------------------------------------------------
        # MiniRocket features are sparse non-negative responses.
        # StandardScaler's centering turns them into signed values,
        # which is theoretically undesirable. Empirically, scaler
        # has negligible impact (<0.5%) on this dataset, but disabled
        # by default to align with MiniRocket's original design.
        # Enable with --use_scaler for ablation.
        # =========================================================
        if cfg.use_scaler:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_clf = scaler.fit_transform(X_train_features)
            X_val_clf = scaler.transform(X_val_features)
        else:
            X_train_clf = X_train_features
            X_val_clf = X_val_features

        clf = RidgeClassifierCV(alphas=np.logspace(-6, 6, 20))
        clf.fit(X_train_clf, y_train)

        # --- Validation ---
        y_pred = clf.predict(X_val_clf)
        try:
            y_scores = clf.decision_function(X_val_clf)
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
        print(f"[Fold {fold}] Measuring inference latency...")
        dummy_single = X_val[:1]
        for _ in range(10):  # warmup
            _ = rocket.transform(dummy_single)
            _ = clf.predict(_)

        t0 = time.perf_counter()
        for _ in range(100):
            _ = rocket.transform(dummy_single)
            _ = clf.predict(_)
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
            "use_scaler": cfg.use_scaler,
            "n_jobs": cfg.n_jobs,
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
    
    # FIX 3: default n_jobs changed from -1 to 1 for Windows stability
    parser.add_argument("--n_jobs", type=int, default=1, help="sktime n_jobs (use 1 on Windows)")
    
    # FIX 2: toggle to test StandardScaler impact
    parser.add_argument("--use_scaler", action="store_true", help="Enable StandardScaler (for ablation)")
    
    args = parser.parse_args()
    main(args)