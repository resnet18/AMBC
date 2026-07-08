#!/usr/bin/env python3
"""
AMBC Experiment: MiniRocket Baseline (Official tsai/fastai)
===========================================================
Faithful reproduction of the official MiniRocket baseline from
Yang et al. using tsai + fastai, matching the official Colab
notebook implementation as closely as possible.

Input:  data/processed/fold_{i}/X_{train,val}.npy
Output: experiments/minirocket_tsai/outputs/results.json

Requirements:
    pip install tsai fastai

Note:
- Requires GPU (CUDA) for practical training speed. 200 epochs on
  CPU is prohibitively slow for 5-fold CV.
- The official notebook uses fit_one_cycle(200, 2.5e-5) with
  MiniRocketHead (neural network classifier) on top of
  MiniRocketFeatures (PyTorch kernel extraction).
- The notebook we audited appears to run multi-class (19 classes)
  by default; we override to binary (before/after) for AMBC.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch

# tsai / fastai imports matching official notebook
from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures, MiniRocketHead
from tsai.models.utils import build_ts_model
from tsai.models.MINIROCKET_Pytorch import get_minirocket_features
from tsai.data.core import get_ts_dls
from tsai.data.transforms import TSClassification
from tsai.data.preprocessing import TSStandardize
from fastai.vision import *
from fastai.callback.progress import CSVLogger
from fastai.learner import Learner
from fastai.metrics import accuracy

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
    # Dimension transpose for tsai
    # ---------------------------------------------------------
    # tsai MiniRocketFeatures expects (N, C, T) = (N, 23, 4096).
    # Our preprocessed .npy stores (N, T, C) = (N, 4096, 23).
    # Commenting out the following two lines reproduces the
    # official dimension bug (c_in=4096, seq_len=23), yielding
    # ~0.59 Acc. This matches the reported 59.8% in Yang et al.
    # and all derived reproductions.
    # =========================================================
    X_train = np.transpose(X_train, (0, 2, 1))  # (N, 23, 4096)
    X_val = np.transpose(X_val, (0, 2, 1))       # (N, 23, 4096)
    
    return X_train, y_train, X_val, y_val


def main(cfg):
    processed_dir = Path(cfg.processed_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("AMBC Experiment: MiniRocket Baseline (tsai/fastai)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Processed dir: {processed_dir}")
    print(f"Output dir:    {output_dir}")
    print("=" * 60)
    
    fold_results = []
    
    for fold in range(5):
        print(f"\n[Fold {fold}] Loading data...")
        X_train, y_train, X_val, y_val = load_fold(processed_dir, fold)
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
        
        # --- MiniRocket feature extraction (PyTorch GPU) ---
        print(f"[Fold {fold}] Fitting MiniRocketFeatures...")
        mrf = MiniRocketFeatures(
            c_in=X_train.shape[1],      # 23 channels
            seq_len=X_train.shape[2],   # 4096 timesteps
            num_features=cfg.num_features,
            max_dilations_per_kernel=cfg.max_dilations,
        ).to(device)
        
        chunksize = 64
        mrf.fit(X_train, chunksize=chunksize)
        
        # Extract features for all data (train + val)
        X_all = np.concatenate([X_train, X_val])
        X_feat = get_minirocket_features(X_all, mrf, chunksize=chunksize, to_np=True)
        print(f"  Feature shape: {X_feat.shape}")
        
        # --- fastai data loader ---
        n_train = len(X_train)
        n_val = len(X_val)
        splits = [list(range(n_train)), list(range(n_train, n_train + n_val))]
        
        tfms = [None, TSClassification()]
        batch_tfms = TSStandardize(by_sample=True)  # matches official notebook
        dls = get_ts_dls(
            X_feat,
            np.concatenate([y_train, y_val]),
            splits=splits,
            tfms=tfms,
            batch_tfms=batch_tfms,
            bs=cfg.batch_size,
        )
        
        # --- Build classifier head (neural network) ---
        model = build_ts_model(MiniRocketHead, dls=dls)
        
        # --- fastai Learner ---
        learn = Learner(dls, model, metrics=accuracy)
        
        print(f"[Fold {fold}] Training fit_one_cycle({cfg.epochs} epochs, lr={cfg.lr})...")
        learn.fit_one_cycle(cfg.epochs, cfg.lr)
        
        # --- Validation predictions ---
        preds, targs = learn.get_preds()
        y_prob = preds[:, 1].cpu().numpy() if preds.shape[1] == 2 else preds.cpu().numpy()
        y_pred = preds.argmax(dim=1).cpu().numpy()
        
        metrics = compute_classification_metrics(
            y_true=y_val,
            y_pred=y_pred,
            y_prob=y_prob,
            pos_label=1,
        )
        
        # --- Latency measurement (GPU sync required) ---
        print(f"[Fold {fold}] Measuring inference latency (GPU sync)...")
        dummy = torch.from_numpy(X_val[:1]).float().to(device)
        
        # Warmup
        for _ in range(10):
            _ = mrf(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Timed run
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(cfg.n_repeat):
            _ = mrf(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0 / cfg.n_repeat
        
        # Peak GPU memory during single inference
        peak_mem_mb = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            _ = mrf(dummy)
            torch.cuda.synchronize()
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        
        # Model stats
        n_params = sum(p.numel() for p in model.parameters())
        
        result = {
            "fold": fold,
            "val_acc": metrics["accuracy"],
            "val_f1": metrics["f1"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_auc": metrics["auc"],
            "latency_ms": float(latency_ms),
            "peak_memory_mb": float(peak_mem_mb) if peak_mem_mb is not None else None,
            "n_params": int(n_params),
        }
        fold_results.append(result)
        
        print(f"[Fold {fold}] Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
              f"Latency={latency_ms:.3f}ms")
    
    # --- Summary ---
    accs = [r["val_acc"] for r in fold_results]
    f1s = [r["val_f1"] for r in fold_results]
    lats = [r["latency_ms"] for r in fold_results]
    mems = [r["peak_memory_mb"] for r in fold_results if r["peak_memory_mb"] is not None]
    
    summary = {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "mean_latency_ms": float(np.mean(lats)),
        "std_latency_ms": float(np.std(lats)),
        "mean_peak_memory_mb": float(np.mean(mems)) if mems else None,
        "n_params": fold_results[0]["n_params"],
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Accuracy: {summary['mean_acc']:.4f} ± {summary['std_acc']:.4f}")
    print(f"F1:       {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"Latency:  {summary['mean_latency_ms']:.3f} ± {summary['std_latency_ms']:.3f} ms/flight")
    if summary["mean_peak_memory_mb"]:
        print(f"GPU Mem:  {summary['mean_peak_memory_mb']:.1f} MB")
    print("=" * 60)
    
    # --- Save results ---
    results = {
        "model": "minirocket_tsai",
        "protocol": "AMBC-Standard-v1.0",
        "config": {
            "num_features": cfg.num_features,
            "max_dilations": cfg.max_dilations,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "classifier": "MiniRocketHead (fastai NN)",
        },
        "fold_results": fold_results,
        "summary": summary,
    }
    
    out_path = output_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMBC MiniRocket Baseline (tsai/fastai)")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="experiments/minirocket/outputs_tsai")
    parser.add_argument("--num_features", type=int, default=10000)
    parser.add_argument("--max_dilations", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200, help="fit_one_cycle epochs (official notebook uses 200)")
    parser.add_argument("--lr", type=float, default=2.5e-5, help="fastai max_lr (official notebook uses 2.5e-5)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_repeat", type=int, default=100, help="latency measurement repeats")
    args = parser.parse_args()
    main(args)