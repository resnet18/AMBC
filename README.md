# AMBC

**A**viation **M**aintenance **B**inary **C**lassification

一个基于 NGAFID 数据集的非正式 benchmark，聚焦航空维修前/后航班的二分类检测。

---

## 中文

**任务**：维护前 (before) vs 维护后 (after) 二分类  
**数据**：NGAFID Section 3.2 基准子集（19 类维护事件 → 二分类标签）  
**目标**：
1. 审计已有复现结果，建立可复现的收录标准
2. 测试轻量级架构（1D ResNet / Micro-Transformer）在资源受限场景下的可用性边界
3. 绘制性能-参数量-推理延迟的 Pareto frontier

**收录分级**：
- **Verified**：已审计，可复现，无数据泄露
- **Pending**：代码在，待跑/待审
- **Unverified**：信息不足，不收录

---

## English

**Task**: Binary classification of pre-maintenance vs post-maintenance flights  
**Data**: NGAFID Section 3.2 benchmark subset (19 maintenance event types → binary labels)  
**Goals**:
1. Audit existing reproductions with reproducibility standards
2. Test lightweight architectures (1D ResNet / Micro-Transformer) under resource constraints
3. Plot Pareto frontier of performance vs parameters vs inference latency

**Inclusion Tiers**:
- **Verified**: Audited, reproducible, no data leakage
- **Pending**: Code available, pending execution/audit
- **Unverified**: Insufficient information, excluded

---

## 速查 / Cheat Sheet

| 项目 | 内容 |
|------|------|
| 数据集 | NGAFID Aviation Maintenance Dataset |
| 官方源 | Zenodo DOI: 10.5281/zenodo.6624956 |
| 传感器 | 23 个，1Hz 采样 |
| 规模 | ~28k 航班，4.3GB |
| 当前目标 | 建立收录标准 + 轻量架构探索 |

---

*Created for the lab qualifying task. Not an official benchmark — yet.*
