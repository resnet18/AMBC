#!/usr/bin/env python3
"""
ARCHIVED: InceptionTime Baseline (TensorFlow)
=============================================
This script implements the official InceptionTime architecture in pure TF/Keras,
faithful to Yang et al.'s original notebook.

STATUS: NON-FUNCTIONAL on RTX 5060Ti (Blackwell / sm_120)
REASON: TensorFlow's cuDNN backend does not provide autotuned configs for
        Conv2DBackpropFilter / ConvForward on CUDA capability 12.0.
        Training fails with `DEVICE_TYPE_INVALID` regardless of TF version
        (tested TF 2.15-2.21 on WSL2 + CUDA 13.2).

WORKAROUND: Use `experiments\inception_time\run.py` (PyTorch implementation) for GPU,
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


def _inception_module(input_tensor, nb_filters=32, bottleneck_size=32, kernel_size=41, use_bottleneck=True, use_residual=True):
    """Inception module from the official paper."""
    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        x = tf.keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1, padding="same", use_bias=False)(input_tensor)
    else:
        x = input_tensor

    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
    conv_list = []
    for k in kernel_size_s:
        conv_list.append(
            tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=k, strides=1, padding="same", use_bias=False)(x)
        )

    max_pool = tf.keras.layers.MaxPool1D(pool_size=3, strides=1, padding="same")(input_tensor)
    conv_6 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=1, padding="same", use_bias=False)(max_pool)
    conv_list.append(conv_6)

    x = tf.keras.layers.Concatenate(axis=2)(conv_list)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def _shortcut_layer(input_tensor, out_tensor):
    shortcut = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1, padding="same", use_bias=False)(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([shortcut, out_tensor])
    x = tf.keras.layers.Activation("relu")(x)
    return x


def build_inception_time(input_shape=(4096, 23), nb_classes=2, nb_filters=32, depth=6, kernel_size=41, use_residual=True, use_bottleneck=True):
    """InceptionTime architecture from the official paper."""
    input_layer = tf.keras.layers.Input(input_shape)
    x = input_layer
    input_res = input_layer

    for d in range(depth):
        x = _inception_module(x, nb_filters=nb_filters, kernel_size=kernel_size, use_bottleneck=use_bottleneck)
        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap = tf.keras.layers.GlobalAveragePooling1D()(x)
    output = tf.keras.layers.Dense(nb_classes, activation="softmax", name="before_after")(gap)
    model = tf.keras.Model(inputs=input_layer, outputs=output)
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
    print("AMBC Experiment: InceptionTime Baseline (TF)")
    print("=" * 60)
    print(f"Device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
    print("=" * 60)

    fold_results = []

    for fold in range(5):
        print(f"\n[Fold {fold}] Loading data...")
        X_train, y_train, X_val, y_val = load_fold(processed_dir, fold)
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

        model = build_inception_time(
            nb_filters=cfg.nb_filters,
            depth=cfg.depth,
            kernel_size=cfg.kernel_size,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
            loss="sparse_categorical_crossentropy" if cfg.nb_classes >= 2 else "binary_crossentropy",
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
        y_prob_full = model.predict(X_val, batch_size=cfg.batch_size, verbose=0)  # (N, 2)
        y_prob = y_prob_full[:, 1]  # 正类概率，给 compute_classification_metrics
        y_pred = y_prob_full.argmax(axis=1)  # 预测类别

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
        "model": "inception_time",
        "protocol": "AMBC-Standard-v1.0",
        "config": {"depth": cfg.depth, "nb_filters": cfg.nb_filters, "kernel_size": cfg.kernel_size, "epochs": cfg.epochs},
        "fold_results": fold_results,
        "summary": summary,
    }

    out_path = output_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMBC InceptionTime Baseline (TF)")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="experiments/inception_time/outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--nb_filters", type=int, default=32)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--kernel_size", type=int, default=41)
    parser.add_argument("--nb_classes", type=int, default=2)
    args = parser.parse_args()
    main(args)
