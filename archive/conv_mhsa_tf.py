#!/usr/bin/env python3
"""
ARCHIVED: ConvMHSA Baseline (TensorFlow)
========================================
Faithful reproduction of the official ConvMHSA architecture from Yang et al.

STATUS: NON-FUNCTIONAL on RTX 5060 Ti (Blackwell / sm_120)
REASON: TensorFlow's cuDNN backend lacks autotuned configs for
        Conv2DBackpropFilter / ConvForward on CUDA capability 12.0.
        Training fails with `DEVICE_TYPE_INVALID` on WSL2 + CUDA 13.2.

WORKAROUND: Use `experiments\conv_mhsa\run.py` (PyTorch implementation) for GPU execution.
            or run this script on CPU (prohibitively slow for 5-fold CV).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import compute_classification_metrics


def load_fold(processed_dir: str, fold: int):
    fold_dir = Path(processed_dir) / f"fold_{fold}"
    X_train = np.load(fold_dir / "X_train.npy").astype(np.float32)
    y_train = np.load(fold_dir / "y_train.npy").astype(np.int64)
    X_val = np.load(fold_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(fold_dir / "y_val.npy").astype(np.int64)
    return X_train, y_train, X_val, y_val


def build_conv_mhsa(input_shape=(4096, 23), num_classes=2, d_model=512, dff=1024):
    inputs = tf.keras.Input(shape=input_shape, name="data")
    x = inputs

    x = tf.keras.layers.Conv1D(128, 7, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(128, 7, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(256, 7, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(256, 7, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(512, 7, strides=2, padding="same", activation="relu")(x)

    for _ in range(4):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model // 8)(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ffn = tf.keras.layers.Dense(dff, activation="relu")(x)
        ffn = tf.keras.layers.Dense(d_model)(ffn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # FIXED: softmax unconditionally to match sparse_categorical_crossentropy
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="before_after")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def measure_tf_latency(model, dummy_input, n_warmup=10, n_repeat=100):
    for _ in range(n_warmup):
        _ = model(dummy_input, training=False)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        _ = model(dummy_input, training=False)
    return (time.perf_counter() - t0) * 1000.0 / n_repeat


def main(cfg):
    processed_dir = Path(cfg.processed_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ARCHIVED: ConvMHSA Baseline (TF)")
    print("=" * 60)
    print(f"Device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
    print("=" * 60)

    fold_results = []

    for fold in range(5):
        print(f"\n[Fold {fold}] Loading data...")
        X_train, y_train, X_val, y_val = load_fold(processed_dir, fold)

        model = build_conv_mhsa()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        print(f"[Fold {fold}] Training...")
        history = model.fit(
            X_train, y_train,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            validation_data=(X_val, y_val),
            verbose=0,
        )

        print(f"[Fold {fold}] Evaluating...")
        # FIXED: proper extraction for softmax output
        y_prob_full = model.predict(X_val, batch_size=cfg.batch_size, verbose=0)
        y_prob = y_prob_full[:, 1]
        y_pred = y_prob_full.argmax(axis=1)

        metrics = compute_classification_metrics(y_val, y_pred, y_prob, pos_label=1)

        dummy = np.zeros((1, 4096, 23), dtype=np.float32)
        latency_ms = measure_tf_latency(model, dummy)
        n_params = model.count_params()

        result = {
            "fold": fold,
            "val_acc": metrics["accuracy"],
            "val_f1": metrics["f1"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_auc": metrics["auc"],
            "latency_ms": float(latency_ms),
            "peak_memory_mb": None,
            "n_params": int(n_params),
        }
        fold_results.append(result)

        print(f"[Fold {fold}] Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
              f"Latency={latency_ms:.3f}ms")

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

    results = {
        "model": "conv_mhsa_tf",
        "protocol": "AMBC-Standard-v1.0",
        "config": {"epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr},
        "fold_results": fold_results,
        "summary": summary,
    }

    out_path = output_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMBC ConvMHSA Baseline (TF) — ARCHIVED")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="experiments/conv_mhsa/outputs_tf")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_classes", type=int, default=2)
    args = parser.parse_args()
    main(args)