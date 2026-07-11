#!/usr/bin/env python3
"""
AMBC Experiment: SongX-1 CNN+BiMamba
=====================================
Standard Protocol reproduction of SongX-1's CNN+BiMamba under
AMBC unified evaluation protocol.

Input:  data/processed/fold_{i}/X_{train,val}.npy
Output: experiments/cnn_bimamba/outputs/results.json

Modes:
    --cnn_only    Skip Mamba blocks, pure CNN backbone (CPU-friendly, for debug)
    (default)     Full CNN+BiMamba (requires CUDA + mamba_ssm)
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.ambc_dataset import AMBCSequenceDataset
from src.models.cnn_bimamba import CNNBiMambaClassifier
from src.evaluation.metrics import (
    compute_classification_metrics,
    measure_latency,
    reset_peak_memory,
    get_peak_memory_mb,
    count_params,
)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_y = []
    all_pred = []
    all_prob = []

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        static = batch["static"].to(device)
        y = batch["y"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x, mask, static)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        total_loss += loss.item() * y.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()

        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(preds.tolist())
        all_prob.extend(probs.tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_classification_metrics(
        y_true=np.array(all_y),
        y_pred=np.array(all_pred),
        y_prob=np.array(all_prob),
        pos_label=1,
    )
    metrics["loss"] = avg_loss
    return metrics


def main(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not cfg.cnn_only and device.type == "cpu":
        print("[Warning] Full BiMamba requires CUDA + mamba_ssm. Running on CPU will fail.")
        print("[Hint] Add --cnn_only to run CNN-only mode on CPU.")
        raise RuntimeError("Full BiMamba requires GPU. Use --cnn_only for CPU debug.")

    print("=" * 60)
    print("AMBC Experiment: SongX-1 CNN+BiMamba")
    print("=" * 60)
    print(f"Device:      {device}")
    print(f"CNN-only:    {cfg.cnn_only}")
    print(f"Processed:   {cfg.processed_dir}")
    print(f"Output:      {cfg.output_dir}")
    print("=" * 60)

    processed_dir = Path(cfg.processed_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []

    for fold in range(5):
        print(f"\n[Fold {fold}] Loading data...")
        fold_dir = processed_dir / f"fold_{fold}"
        train_ds = AMBCSequenceDataset(
            str(fold_dir / "X_train.npy"),
            str(fold_dir / "y_train.npy"),
        )
        val_ds = AMBCSequenceDataset(
            str(fold_dir / "X_val.npy"),
            str(fold_dir / "y_val.npy"),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # --- Model ---
        mamba_layers = 0 if cfg.cnn_only else cfg.mamba_layers
        model = CNNBiMambaClassifier(
            seq_in_dim=23,
            static_dim=0,
            cnn_dim=cfg.cnn_dim,
            stem_kernel=cfg.stem_kernel,
            stem_stride=cfg.stem_stride,
            mamba_layers=mamba_layers,
            mamba_d_state=cfg.mamba_d_state,
            mamba_d_conv=cfg.mamba_d_conv,
            mamba_expand=cfg.mamba_expand,
            static_hidden=cfg.static_hidden,
            static_out=cfg.static_out,
            num_classes=2,
            dropout=cfg.dropout,
        ).to(device)

        print(f"[Fold {fold}] Model: CNNBiMamba(mamba_layers={mamba_layers})")
        print(f"  Total params: {count_params(model):,}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        # --- Training ---
        best_f1 = -1.0
        best_state = None
        history = []

        for epoch in range(1, cfg.epochs + 1):
            train_m = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
            val_m = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

            history.append({
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "train_acc": train_m["accuracy"],
                "train_f1": train_m["f1"],
                "val_loss": val_m["loss"],
                "val_acc": val_m["accuracy"],
                "val_f1": val_m["f1"],
            })

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"[Epoch {epoch:02d}] "
                    f"train_loss={train_m['loss']:.4f} train_acc={train_m['accuracy']:.4f} | "
                    f"val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']:.4f} val_f1={val_m['f1']:.4f}"
                )

            if val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                best_state = model.state_dict().copy()

        # Load best
        model.load_state_dict(best_state)
        final_val = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        # --- Efficiency ---
        dummy_x = torch.randn(1, 4096, 23).to(device)
        dummy_mask = torch.ones(1, 4096, dtype=torch.bool).to(device)
        dummy_static = torch.zeros(1, 0).to(device)

        reset_peak_memory(device)
        latency_ms = measure_latency(
            model, (dummy_x, dummy_mask, dummy_static), device,
            n_warmup=10, n_repeat=100,
        )
        peak_mem_mb = get_peak_memory_mb(device)

        result = {
            "fold": fold,
            "val_acc": final_val["accuracy"],
            "val_f1": final_val["f1"],
            "val_precision": final_val["precision"],
            "val_recall": final_val["recall"],
            "val_auc": final_val["auc"],
            "latency_ms": float(latency_ms),
            "peak_memory_mb": float(peak_mem_mb) if peak_mem_mb is not None else None,
            "n_params": count_params(model),
        }
        fold_results.append(result)

        print(f"[Fold {fold}] Best: val_acc={final_val['accuracy']:.4f} val_f1={final_val['f1']:.4f}")

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

    # --- Save ---
    results = {
        "model": "cnn_bimamba" if not cfg.cnn_only else "cnn_only",
        "protocol": "AMBC-Standard-v1.0",
        "config": {
            "cnn_dim": cfg.cnn_dim,
            "mamba_layers": 0 if cfg.cnn_only else cfg.mamba_layers,
            "mamba_d_state": cfg.mamba_d_state,
            "mamba_d_conv": cfg.mamba_d_conv,
            "mamba_expand": cfg.mamba_expand,
            "dropout": cfg.dropout,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
        },
        "fold_results": fold_results,
        "summary": summary,
    }

    out_path = output_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMBC SongX-1 CNN+BiMamba")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="experiments/cnn_bimamba/outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # Model architecture
    parser.add_argument("--cnn_dim", type=int, default=128)
    parser.add_argument("--stem_kernel", type=int, default=7)
    parser.add_argument("--stem_stride", type=int, default=2)
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--mamba_d_state", type=int, default=16)
    parser.add_argument("--mamba_d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--static_hidden", type=int, default=128)
    parser.add_argument("--static_out", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)

    # Mode switch
    parser.add_argument("--cnn_only", action="store_true", help="Skip Mamba blocks, pure CNN (CPU-friendly)")

    args = parser.parse_args()
    main(args)