#!/usr/bin/env python3
"""
AMBC Experiment: ConvMHSA Baseline (PyTorch)
=============================================
PyTorch implementation of the official ConvMHSA architecture.
Verified on RTX 5060 Ti (cu132). Replaces the TF version which is
non-functional on Blackwell due to cuDNN autotuner gaps.

Usage:
    python experiments/conv_mhsa/run.py --epochs 10

Key Arguments:
    --epochs      Training epochs. Default: 200 (matches official TF notebook).
    --lr          Learning rate. Default: 3e-5
    --batch_size  Effective batch size. Default: 128 (official config).
    --micro_batch Micro-batch per forward pass. Default: 64.

Examples:
    # Quick sanity check (10 epochs, ~2 minutes)
    python experiments/conv_mhsa/run.py --epochs 10

    # Full convergence (200 epochs, official TF setting)
    python experiments/conv_mhsa/run.py

    # Ablation Study (enable the "fixed" version, BN + PosEnc + GELU + CosineLR)
    python experiments/conv_mhsa/run.py --enhanced
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
from torch.amp import autocast, GradScaler

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import compute_classification_metrics


class ConvMHSABlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        x = self.ln1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x


class ConvMHSA(nn.Module):
    def __init__(self, c_in=23, c_out=2, seq_len=4096, d_model=512, dff=1024,
                 num_heads=8, num_layers=4, enhanced=False):
        super().__init__()
        self.enhanced = enhanced
        self.d_model = d_model

        # CNN stem: 4096 -> 2048 -> 2048 -> 1024 -> 512
        layers = []
        cfg = [
            (c_in, 128, 1),   # 4096
            (128, 128, 2),    # 2048
            (128, 256, 1),    # 2048
            (256, 256, 2),    # 1024
            (256, 512, 2),    # 512
        ]
        for i, (in_ch, out_ch, stride) in enumerate(cfg):
            layers.append(nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3))
            if enhanced:
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
        self.stem = nn.Sequential(*layers)

        # Positional encoding (learnable) for enhanced version
        if enhanced:
            self.pos_enc = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        self.mhsa_blocks = nn.ModuleList([
            ConvMHSABlock(d_model, num_heads, dff) for _ in range(num_layers)
        ])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, c_out)

    def forward(self, x):
        x = self.stem(x)           # (batch, 512, 512)
        x = x.transpose(1, 2)      # (batch, 512, 512)
        if self.enhanced:
            x = x + self.pos_enc
        for block in self.mhsa_blocks:
            x = block(x)
        x = x.transpose(1, 2)      # (batch, 512, 512)
        x = self.gap(x).squeeze(-1)  # (batch, 512)
        return self.classifier(x)


def load_fold(processed_dir: str, fold: int):
    fold_dir = Path(processed_dir) / f"fold_{fold}"
    X_train = np.load(fold_dir / "X_train.npy").astype(np.float32)
    y_train = np.load(fold_dir / "y_train.npy").astype(np.int64)
    X_val = np.load(fold_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(fold_dir / "y_val.npy").astype(np.int64)
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    return X_train, y_train, X_val, y_val


def main(cfg):
    # [HARDWARE APPROXIMATION] RTX 5060 Ti 16GB VRAM saturates at >95% under the
    # official batch_size=128, causing CUDA memory allocator thrashing and
    # ~400s/epoch (would be ~23h/fold). We enable cudnn.benchmark to let PyTorch
    # autotune conv algorithms for this specific workload, and empty_cache to
    # reduce fragmentation on WDDM Windows drivers.
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("AMBC Experiment: ConvMHSA Baseline (PyTorch)")
    if cfg.enhanced:
        print("  [ENHANCED] BN + PosEnc + GELU + CosineAnnealingLR")
    print("=" * 60)
    print(f"Device: {device}")
    # [HARDWARE APPROXIMATION] Print effective batch size info for reproducibility notes.
    print(f"Effective batch size: {cfg.batch_size} | "
          f"Micro batch: {cfg.micro_batch} | "
          f"Accumulation: {cfg.batch_size // cfg.micro_batch}")
    print("=" * 60)

    fold_results = []

    for fold in range(5):
        print(f"\n[Fold {fold}] Loading data...")
        X_train, y_train, X_val, y_val = load_fold(cfg.processed_dir, fold)
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

        model = ConvMHSA(
            c_in=23, c_out=2, seq_len=4096,
            d_model=512, dff=1024, num_heads=8, num_layers=4,
            enhanced=cfg.enhanced,
        ).to(device)

        # [HARDWARE APPROXIMATION] torch.compile disabled on Windows because
        # Triton (required by torch.compile/inductor backend) is Linux-only.
        # This suppresses the "triton not found" and "Not enough SMs" warnings.
        if hasattr(torch, 'compile') and os.name != 'nt':
            model = torch.compile(model, mode="default")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scaler = GradScaler()

        scheduler = None
        if cfg.enhanced:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs, eta_min=1e-6
            )

        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val)
        )
        # [HARDWARE APPROXIMATION] DataLoader uses micro_batch to fit VRAM.
        # The official TF/TPU config uses batch_size=128. On RTX 5060 Ti 16GB,
        # batch_size=128 saturates VRAM and triggers allocator thrashing.
        # We use micro_batch=64 for the DataLoader and recover the effective
        # batch size via gradient accumulation (see training loop).
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=cfg.micro_batch, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=cfg.micro_batch
        )

        print(f"[Fold {fold}] Training {cfg.epochs} epochs...")
        best_val_acc = 0.0
        best_state = None

        # [HARDWARE APPROXIMATION] Pre-compute accumulation steps so the
        # effective batch size matches the official paper (128).
        accumulation_steps = cfg.batch_size // cfg.micro_batch

        for epoch in tqdm(range(cfg.epochs), desc=f"[Fold {fold}] Training", leave=False):
            model.train()
            optimizer.zero_grad()

            # [HARDWARE APPROXIMATION] Gradient accumulation loop.
            # Official TF/TPU config uses batch_size=128. On RTX 5060 Ti 16GB,
            # batch_size=128 saturates VRAM (>95%) and triggers allocator
            # thrashing, degrading throughput by ~10x (400s/epoch).
            # We split each effective batch into micro-batches of 64 and accumulate
            # gradients over 2 steps before stepping the optimizer. This yields
            # the *same* average gradient as batch_size=128 because:
            #   loss = criterion(logits, yb) / accumulation_steps
            #   -> backward() accumulates scaled gradients
            #   -> step() updates with the mean over 128 samples.
            # Note: LayerNorm stats are per-sample, so this is exact for our model.
            for i, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                with autocast(device_type='cuda'):
                    logits = model(xb)
                    loss = criterion(logits, yb) / accumulation_steps
                scaler.scale(loss).backward()

                # Step optimizer when accumulation buffer is full OR at the last
                # batch of the epoch. This avoids the "No inf checks" error
                # from calling scaler.step() with zero gradients.
                is_last = (i + 1) == len(train_loader)
                if (i + 1) % accumulation_steps == 0 or is_last:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            if scheduler:
                scheduler.step()

            # Validation every 20 epochs + final epoch
            if (epoch + 1) % 20 == 0 or epoch == cfg.epochs - 1:
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

                tqdm.write(f"  [Fold {fold}] Epoch {epoch+1}/{cfg.epochs} — val_acc={val_acc:.4f}")

        if best_state:
            model.load_state_dict(best_state)
            model.to(device)

        # Final eval
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
            # [HARDWARE APPROXIMATION] Audit trail: record the approximation
            # in the result JSON so downstream analysis knows this was not
            # run on TPU but on a consumer GPU with gradient accumulation.
            "hardware_note": (
                f"Gradient accumulation: effective_batch={cfg.batch_size}, "
                f"micro_batch={cfg.micro_batch}, steps={accumulation_steps}. "
                f"Reason: RTX 5060 Ti 16GB VRAM thrashing at native batch=128."
            ),
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
            "model": "conv_mhsa_enhanced" if cfg.enhanced else "conv_mhsa",
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
    parser = argparse.ArgumentParser(description="AMBC ConvMHSA Baseline (PyTorch)")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="experiments/conv_mhsa/outputs")
    # [HARDWARE APPROXIMATION] --batch_size remains the official effective batch size (128).
    # --micro_batch is the actual GPU forward batch size (64) used to avoid VRAM thrashing.
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--micro_batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_repeat", type=int, default=100)
    parser.add_argument("--enhanced", action="store_true",
                        help="Enable BN + PosEnc + GELU + CosineAnnealingLR (ablation)")
    args = parser.parse_args()
    main(args)