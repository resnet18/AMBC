# AMBC

**A**viation **M**aintenance **B**inary **C**lassification

一个基于 NGAFID 数据集的非正式 benchmark，聚焦航空维修前/后航班的二分类检测。

---

## 中文

**任务**：维护前 (before) vs 维护后 (after) 二分类  
**数据**：NGAFID Section 3.2 基准子集（19 类维护事件 → 二分类标签）  
**目标**：
1. 审计已有复现结果，建立可复现的收录标准
2. 测试已有工作未覆盖的轻量级架构的可用性边界
3. 寻找性能-效率的 Pareto frontier

具体的测评原则和标准参见`evaluation_protocol.md`。

---

## English

**Task**: Binary classification of pre-maintenance vs post-maintenance flights  
**Data**: NGAFID Section 3.2 benchmark subset (19 maintenance event types → binary labels)  
**Goals**:
1. Audit existing reproductions with reproducibility standards
2. Test the availability of lightweight architectures not verified before
3. Find the Pareto frontier of performance and efficiency

See detailed evaluation standards and principles at `evaluation_protocol.md`.

---

## 速查 / Cheat Sheet

| 项目 | 内容 |
|------|------|
| 数据集 | NGAFID Aviation Maintenance Dataset |
| 官方源 | Zenodo DOI: 10.5281/zenodo.6624956 |
| 传感器 | 23 个，1Hz 采样 |
| 规模 | ~28k 航班，4.3GB |
| 目标 | 建立收录标准 + 轻量架构探索 |
| 评估协议 | `evaluation_protocol.md` |

## 初步结论

| # | 方法 | 结果（预实验） |
|---|------|------|
| 1 | Logistic Regression | Acc ~0.63 / F1 ~0.63 |
| 2 | MiniRocket (sktime/RidgeClassifierCV) | Acc ~0.72 / F1 ~0.71 |
| 3 | MiniRocket (tsai/MiniRocketHeadNN) | Acc ~0.73 / F1 ~0.72 |
| 4 | CNN+BiMamba (WSL2) | Acc ~0.71 / F1 ~0.68 |
| 5 | InceptionTime (PyTorch) | Acc ~0.75 / F1 ~0.75 |
| 6 | ConvMHSA (PyTorch) | Acc ~0.76 / F1 ~0.76 |
| 6 | ConvMHSA (PyTorch/Fixed) | Acc ~0.76 / F1 ~0.76 |
| 7 | MiniRocket (sktime) + 统计特征 (LR) | Acc ~0.71 / F1 ~0.71 |

---

*Created for the lab qualifying task. Not an official benchmark — yet.*
