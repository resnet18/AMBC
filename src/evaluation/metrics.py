#!/usr/bin/env python3
"""
AMBC Unified Evaluation Metrics
=================================
All experiments import from here. No experiment script should define its own
metric computation to ensure cross-model comparability.
"""

import time
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


# =========================================================
# 1. Classification Metrics (Flight-Level)
# =========================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    pos_label: int = 1,
) -> Dict[str, float]:
    """
    Compute flight-level classification metrics.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Ground-truth labels (0 or 1).
    y_pred : np.ndarray, shape (N,)
        Hard predictions (0 or 1).
    y_prob : np.ndarray, shape (N,), optional
        Probability of the positive class. Required for AUC.
    pos_label : int, default 1
        Label treated as positive (before=1, after=0 in AMBC).

    Returns
    -------
    dict
        {
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float,
            "auc": float or None,
        }
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0
    )
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": None,
    }

    if y_prob is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            # e.g., only one class present in y_true
            metrics["auc"] = None

    return metrics


# =========================================================
# 2. Efficiency Metrics
# =========================================================

def measure_latency(
    model: torch.nn.Module,
    dummy_input: tuple,
    device: torch.device,
    n_warmup: int = 10,
    n_repeat: int = 100,
) -> float:
    """
    Measure mean per-sample inference latency in milliseconds.

    Parameters
    ----------
    model : torch.nn.Module
    dummy_input : tuple
        (x, mask, static) ready for model(*dummy_input).
    device : torch.device
    n_warmup : int
    n_repeat : int

    Returns
    -------
    float
        Mean latency per forward pass (ms).
    """
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(*dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(n_repeat):
            _ = model(*dummy_input)
        ender.record()
        torch.cuda.synchronize()
        total_ms = starter.elapsed_time(ender)
    else:
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            _ = model(*dummy_input)
        total_ms = (time.perf_counter() - t0) * 1000.0

    return total_ms / n_repeat


def reset_peak_memory(device: torch.device):
    """Reset CUDA peak memory counter. Call before training or inference."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_mb(device: torch.device) -> Optional[float]:
    """Return peak allocated GPU memory in MB, or None if CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    return None


# =========================================================
# 3. Model Statistics
# =========================================================

def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters.

    Parameters
    ----------
    trainable_only : bool
        If True, count only parameters with requires_grad=True.

    Returns
    -------
    int
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# =========================================================
# 4. Aggregation Helpers
# =========================================================

def aggregate_flight_predictions(
    flight_ids: np.ndarray,
    y_prob: np.ndarray,
    y_true: np.ndarray,
    method: str = "mean",
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Aggregate fragment-level predictions to flight-level.
    Required when a model internally produces multiple outputs per flight.

    Parameters
    ----------
    flight_ids : np.ndarray, shape (M,)
    y_prob : np.ndarray, shape (M,)
        Probability of positive class for each fragment.
    y_true : np.ndarray, shape (M,)
        Ground-truth label for each fragment (all fragments of a flight share it).
    method : {"mean", "vote"}
        "mean" -> average probability then threshold.
        "vote" -> majority vote on hard predictions.
    threshold : float

    Returns
    -------
    dict
        {"flight_ids": ..., "y_true": ..., "y_pred": ..., "y_prob": ...}
    """
    from collections import defaultdict

    flight_to_probs = defaultdict(list)
    flight_to_true = {}

    for fid, prob, true in zip(flight_ids, y_prob, y_true):
        flight_to_probs[fid].append(prob)
        flight_to_true[fid] = true

    ids = []
    y_true_agg = []
    y_prob_agg = []
    y_pred_agg = []

    for fid in sorted(flight_to_probs.keys()):
        probs = flight_to_probs[fid]
        true = flight_to_true[fid]

        if method == "mean":
            agg_prob = float(np.mean(probs))
        elif method == "vote":
            preds = [1 if p >= threshold else 0 for p in probs]
            agg_prob = float(np.mean(preds))  # fraction of votes for positive
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        agg_pred = 1 if agg_prob >= threshold else 0

        ids.append(fid)
        y_true_agg.append(true)
        y_prob_agg.append(agg_prob)
        y_pred_agg.append(agg_pred)

    return {
        "flight_ids": np.array(ids),
        "y_true": np.array(y_true_agg, dtype=np.int64),
        "y_prob": np.array(y_prob_agg, dtype=np.float32),
        "y_pred": np.array(y_pred_agg, dtype=np.int64),
    }


# =========================================================
# 5. Convenience: Full Evaluation Pass
# =========================================================

def evaluate_model(
    model: torch.nn.Module,
    loader,  # torch DataLoader
    device: torch.device,
    criterion=None,
) -> Dict[str, float]:
    """
    Full evaluation pass returning classification + loss metrics.
    Does NOT measure latency or memory (call those separately).

    Returns
    -------
    dict
        {
            "loss": float or None,
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float,
            "auc": float or None,
        }
    """
    model.eval()
    all_y = []
    all_pred = []
    all_prob = []
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            static = batch.get("static", torch.zeros(x.size(0), 0)).to(device)
            y = batch["y"].to(device)

            logits = model(x, mask, static)
            if criterion is not None:
                loss = criterion(logits, y)
                total_loss += loss.item() * y.size(0)
                n_samples += y.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_y.extend(y.cpu().numpy().tolist())
            all_pred.extend(preds.tolist())
            all_prob.extend(probs.tolist())

    metrics = compute_classification_metrics(
        y_true=np.array(all_y),
        y_pred=np.array(all_pred),
        y_prob=np.array(all_prob) if all_prob else None,
    )

    if criterion is not None and n_samples > 0:
        metrics["loss"] = total_loss / n_samples
    else:
        metrics["loss"] = None

    return metrics