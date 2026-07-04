```markdown
# AMBC Evaluation Protocol
## Air Maintenance Binary Classification — Standard Evaluation Protocol

> **Version:** 1.0  
> **Dataset:** NGAFID 2days maintenance event detection subset  
> **Task:** Flight-level binary classification (`before` vs. `after`)

---

## 1. Dataset & Task Definition

- **Source:** NGAFID (National General Aviation Flight Information Database) — *A Large-Scale Annotated Multivariate Time Series Aviation Maintenance Dataset from the NGAFID*
- **Subset:** `2days` benchmark subset
- **Unit of Evaluation:** **Flight-level** (one prediction per `Master Index`). Fragment-level evaluation is **prohibited** in the Standard Track.
- **Input:** Multivariate time series per flight, 23 sensor channels.
- **Label:** `before_after` column in `flight_header.csv`.
  - `before` → class `1`
  - `after`  → class `0`

---

## 2. Data Preprocessing Protocol (Mandatory)

All models in the Standard Track **must** consume data produced by the following unified preprocessing pipeline. No model is allowed to access raw `flight_data.pkl` directly with custom truncation or windowing.

| Step | Rule | Implementation |
|------|------|----------------|
| **Length Alignment** | Fixed length `T = 4096`. | Longer flights: **truncate to the last 4096 timesteps**. Shorter flights: **zero-pad at the end** to 4096. |
| **Channel Selection** | First 23 channels only. | `arr[:, :23]` |
| **Normalization** | Global MinMax per channel. | Use `stats.csv` (global min / max per channel). `x = (x - min) / (max - min)`. |
| **NaN Handling** | Fill with `0` **after** normalization. | `np.nan_to_num(x, nan=0.0)` |
| **Train/Val Split** | Official 5-fold split. | Use `fold` column in `flight_header.csv` (values 1–5). |

**Output Format:**  
Preprocessed data is saved as NumPy arrays per fold:


processed/
├── fold{i}/
│   ├── X_train.npy   # shape: (N_train, 4096, 23), dtype: float32
│   ├── y_train.npy   # shape: (N_train,), dtype: int64
│   ├── X_val.npy     # shape: (N_val, 4096, 23), dtype: float32
│   └── y_val.npy     # shape: (N_val,), dtype: int64


---

## 3. Training Protocol (Standard Track)

| Rule | Specification |
|------|---------------|
| **Sliding Window** | **Forbidden.** One flight = one training sample. No temporal augmentation via windowing. |
| **Input Format** | `(batch, 4096, 23)` |
| **Label Leakage** | Prohibited. The same `Master Index` must not appear in both train and validation sets. |
| **Static Features** | Optional. If used, must be derived solely from `flight_header.csv` and processed independently. |

### Data Augmentation Prohibition

The following **training-time** data augmentation techniques are **strictly prohibited** in the Standard Track:

| Technique | Rationale |
|---|---|
| **Sliding Window** | Splits a single flight into multiple fragments, artificially inflating the training set and breaking the one-flight-one-sample contract. |
| **Window Slice / Random Crop** | Simulates variable-length flights by randomly cropping and stretching sub-windows; alters the original temporal distribution. |
| **TimeWarp** | Applies non-linear temporal stretching/compression that may erase sharp transient features critical for fault detection. |
| **Gaussian Noise / Scaling** | Injects synthetic noise or rescales sensor magnitudes, changing the raw sensor distribution. |
| **Any label-altering per-sample augmentation** | Destroys the flight-level label correspondence required for evaluation. |

**Permitted preprocessing** (already completed in the data-preprocessing stage):
- MinMax normalization
- NaN imputation
- Fixed-length truncation / zero-padding

**Extended Track** may explore data augmentation, but results must be explicitly tagged `[Training Augmentation]` and must not be directly compared with Standard Track entries.

---

## 4. Evaluation Protocol (Flight-Level)

The evaluation **must** be performed at the flight level, even if the model internally processes sub-fragments during training.

**Procedure:**
1. For each validation flight, produce **one** prediction.
2. If the model architecture inherently produces fragment-level outputs (e.g., patch or window predictions), **aggregate** them to a single flight-level prediction via majority voting or mean probability before thresholding.
3. Compute metrics against the single ground-truth label per flight.

---

## 5. Metrics (Mandatory Reporting)

### 5.1 Performance
| Metric | Priority | Description |
|--------|----------|-------------|
| **F1-score** | **Primary** | `sklearn.metrics.f1_score(..., average='binary')` |
| **Accuracy** | Secondary | `sklearn.metrics.accuracy_score` |

### 5.2 Efficiency
| Metric | Priority | Description |
|--------|----------|-------------|
| **Inference Latency** | **Primary** | Mean per-flight forward-pass time (ms) on target hardware, averaged over `n_repeat ≥ 100` after `n_warmup ≥ 10`. |
| **GPU Memory** | Reference | Peak allocated memory (`torch.cuda.max_memory_allocated()` for PyTorch; `tf.config.experimental.get_memory_info()` for TF). Reported in table only; **not** used in Pareto dominance. |

---

## 6. Result Presentation

### 6.1 Standard Track (Main Table)
All results listed here are directly comparable. Every entry must be reproduced under the exact protocol above.

| Model | F1 | Acc. | Latency (ms) | GPU Mem (MB) | Notes |
|-------|----|------|--------------|--------------|-------|
| ... | ... | ... | ... | ... | ... |

### 6.2 Extended Track (Appendix)
For methods that cannot conform to the Standard Protocol (e.g., variable-length-only architectures, fragment-level training), results may be listed here with explicit disclaimers.

| Source | Claimed | Protocol | Audit Note |
|--------|---------|----------|------------|
| ... | ... | ... | `[Incomparable]` / `[Architecture-Protocol Mismatch]` / `[Bug]` |

### 6.3 Pareto Frontier
- **X-axis:** Inference Latency (log scale recommended).
- **Y-axis:** F1-score.
- **Bubble size (optional):** Accuracy.
- **Dominance rule:** A model dominates another iff it has **higher F1** and **lower latency**. Non-dominated points form the frontier.

---

## 7. Audit Principles

1. **Architecture Agnostic:** The benchmark maintainer does **not** modify model architectures. We only standardize the data and evaluation protocol.
2. **Fail Gracefully:** If a model fails to converge or exhibits severe degradation under the Standard Protocol, the result is marked `N/A` in the main table with an audit note in the appendix.
3. **No Retroactive Fixes:** We do not patch authors' code to "make it work." We report what the architecture produces under a fair protocol.

---

## 8. References

- Yang, H., et al. *A Large-Scale Annotated Multivariate Time Series Aviation Maintenance Dataset from the NGAFID.* (2021)
- NGAFID 2days subset: [Zenodo](https://doi.org/10.5281/zenodo.6624956) / [Kaggle](https://www.kaggle.com/datasets/hooong/aviation-maintenance-dataset-from-the-ngafid)
- Official reproduction notebooks: `hyang0129/NGAFIDDATASET`

---

*Protocol maintained by AMBC. For inclusion in the Standard Track, submit results reproduced under the above protocol with reproducible training scripts.*
```