#!/usr/bin/env python3
"""
AMBC Data Preprocessing Pipeline
=================================
Standard Protocol for NGAFID 2days Benchmark Subset.

Input:  ./data/raw/2days/flight_data.pkl, flight_header.csv, stats.csv
Output: ./data/processed/ (numpy arrays, fixed 4096, MinMax, NaN->0, flight-level)

Protocol Rules:
- Fixed length: 4096 (truncate tail if longer, pad zeros at end if shorter)
- Normalization: MinMax using global stats.csv (per-channel)
- NaN handling: fill with 0 AFTER normalization
- Evaluation unit: flight-level (one flight = one sample)
- Fold: use official 'fold' column in flight_header.csv
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from compress_pickle import load

# =========================================================
# Config
# =========================================================

DEFAULT_DATA_DIR = "./data/raw/2days"
DEFAULT_OUTPUT_DIR = "./data/processed"
MAX_SEQ_LEN = 4096
NUM_CHANNELS = 23
TARGET_COL = "before_after"
FOLD_COL = "fold"
INDEX_COL = "Master Index"


def load_raw_data(data_dir: str) -> Dict:
    """Load raw files from 2days directory."""
    data_dir = Path(data_dir)
    
    flight_data_path = data_dir / "flight_data.pkl"
    header_path = data_dir / "flight_header.csv"
    stats_path = data_dir / "stats.csv"
    
    if not flight_data_path.exists():
        raise FileNotFoundError(f"Missing {flight_data_path}")
    if not header_path.exists():
        raise FileNotFoundError(f"Missing {header_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {stats_path}")
    
    print(f"[Load] flight_data.pkl ...")
    flight_data = load(str(flight_data_path))
    
    print(f"[Load] flight_header.csv ...")
    header_df = pd.read_csv(header_path, index_col=INDEX_COL)
    
    print(f"[Load] stats.csv ...")
    stats_df = pd.read_csv(stats_path)
    
    # stats.csv format: two rows, one for max, one for min
    # We infer by taking the element-wise max/min of both rows
    row0 = stats_df.iloc[0, 1:1+NUM_CHANNELS].to_numpy(dtype=np.float32)
    row1 = stats_df.iloc[1, 1:1+NUM_CHANNELS].to_numpy(dtype=np.float32)
    
    global_max = np.maximum(row0, row1)
    global_min = np.minimum(row0, row1)
    
    return {
        "flight_data": flight_data,
        "header_df": header_df,
        "global_min": global_min,
        "global_max": global_max,
    }


def preprocess_flight(
    arr: np.ndarray,
    global_min: np.ndarray,
    global_max: np.ndarray,
    max_len: int = MAX_SEQ_LEN,
    num_channels: int = NUM_CHANNELS,
) -> np.ndarray:
    """
    Preprocess a single flight sequence to fixed length.
    
    Steps:
    1. Ensure shape is (T, C) and C == num_channels
    2. Truncate to last max_len if T > max_len
    3. Pad with zeros at the end if T < max_len
    4. MinMax normalize using global stats
    5. NaN -> 0
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")
    
    T, C = arr.shape
    if C < num_channels:
        raise ValueError(f"Expected at least {num_channels} channels, got {C}")
    
    # Take first num_channels channels
    arr = arr[:, :num_channels].copy().astype(np.float32)
    
    # Truncate or pad to max_len
    if T > max_len:
        # Official protocol: truncate to LAST max_len timesteps
        arr = arr[-max_len:]
    elif T < max_len:
        pad_len = max_len - T
        pad = np.zeros((pad_len, num_channels), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    
    # MinMax normalization
    denom = global_max - global_min
    denom[denom == 0] = 1.0  # avoid division by zero
    arr = (arr - global_min) / denom
    
    # NaN -> 0 (after normalization)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    return arr


def build_dataset(raw: Dict, max_len: int = MAX_SEQ_LEN) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Build fixed-length dataset aligned with header_df.
    
    Returns:
        X: (N, max_len, NUM_CHANNELS) float32
        y: (N,) int64
        folds: (N,) int64
        meta: dict with ids, lengths, etc.
    """
    flight_data = raw["flight_data"]
    header_df = raw["header_df"]
    global_min = raw["global_min"]
    global_max = raw["global_max"]
    
    # Get common IDs
    common_ids = [idx for idx in header_df.index if idx in flight_data]
    print(f"[Build] {len(common_ids)} flights found in both header and data")
    
    # Filter to rows with valid fold and target
    valid_mask = header_df[FOLD_COL].notna() & header_df[TARGET_COL].notna()
    valid_ids = [idx for idx in common_ids if valid_mask.loc[idx]]
    print(f"[Build] {len(valid_ids)} flights with valid fold and target")
    
    # Prepare arrays
    N = len(valid_ids)
    X = np.zeros((N, max_len, NUM_CHANNELS), dtype=np.float32)
    y = np.zeros(N, dtype=np.int64)
    folds = np.zeros(N, dtype=np.int64)
    original_lengths = np.zeros(N, dtype=np.int64)
    
    bad_ids = []
    
    for i, idx in enumerate(valid_ids):
        arr = flight_data[idx]
        
        if arr.ndim != 2:
            bad_ids.append((idx, f"ndim={arr.ndim}"))
            continue
        
        T, C = arr.shape
        original_lengths[i] = T
        
        if C < NUM_CHANNELS:
            bad_ids.append((idx, f"channels={C}<{NUM_CHANNELS}"))
            continue
        
        try:
            X[i] = preprocess_flight(arr, global_min, global_max, max_len, NUM_CHANNELS)
        except Exception as e:
            bad_ids.append((idx, str(e)))
            continue
        
        # Label mapping
        label_val = header_df.loc[idx, TARGET_COL]
        if isinstance(label_val, str):
            label_str = label_val.strip().lower()
            if label_str == "before":
                y[i] = 1
            elif label_str == "after":
                y[i] = 0
            else:
                bad_ids.append((idx, f"unknown_label={label_val}"))
                continue
        else:
            y[i] = int(label_val)
        
        folds[i] = int(header_df.loc[idx, FOLD_COL])
    
    # Remove bad entries
    if bad_ids:
        print(f"[Warning] {len(bad_ids)} bad flights skipped:")
        for idx, reason in bad_ids[:5]:
            print(f"  - {idx}: {reason}")
        if len(bad_ids) > 5:
            print(f"  ... and {len(bad_ids)-5} more")
    
    good_mask = np.array([idx not in {b[0] for b in bad_ids} for idx in valid_ids])
    X = X[good_mask]
    y = y[good_mask]
    folds = folds[good_mask]
    original_lengths = original_lengths[good_mask]
    valid_ids = [valid_ids[i] for i in range(len(valid_ids)) if good_mask[i]]
    
    meta = {
        "ids": valid_ids,
        "original_lengths": original_lengths.tolist(),
        "num_bad": len(bad_ids),
        "N": len(valid_ids),
    }
    
    print(f"[Build] Final dataset: X={X.shape}, y={y.shape}, folds={folds.shape}")
    print(f"[Build] Fold distribution: {dict(zip(*np.unique(folds, return_counts=True)))}")
    print(f"[Build] Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y, folds, meta


def save_splits(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    meta: Dict,
    output_dir: str,
):
    """Save per-fold train/val splits and unified dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save unified dataset
    np.savez(
        output_dir / "data.npz",
        X=X,
        y=y,
        folds=folds,
        ids=np.array(meta["ids"]),
        original_lengths=np.array(meta["original_lengths"]),
    )
    print(f"[Save] Unified dataset -> {output_dir / 'data.npz'}")
    
    # Save per-fold splits
    fold_values = sorted(np.unique(folds).tolist())
    for fold in fold_values:
        train_mask = folds != fold
        val_mask = folds == fold
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(fold_dir / "X_train.npy", X_train)
        np.save(fold_dir / "y_train.npy", y_train)
        np.save(fold_dir / "X_val.npy", X_val)
        np.save(fold_dir / "y_val.npy", y_val)
        
        print(f"[Save] Fold {fold}: train={len(y_train)}, val={len(y_val)} -> {fold_dir}")
    
    # Save metadata
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Save] Metadata -> {output_dir / 'meta.json'}")


def main():
    parser = argparse.ArgumentParser(description="AMBC Data Preprocessing")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Path to 2days/ directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max_len", type=int, default=MAX_SEQ_LEN, help="Fixed sequence length")
    args = parser.parse_args()
    
    print("=" * 60)
    print("AMBC Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"Data dir:  {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Max length: {args.max_len}")
    print("=" * 60)
    
    raw = load_raw_data(args.data_dir)
    X, y, folds, meta = build_dataset(raw, max_len=args.max_len)
    save_splits(X, y, folds, meta, args.output_dir)
    
    print("=" * 60)
    print("Preprocessing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()