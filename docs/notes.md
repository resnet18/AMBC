## AMBC 项目关键工作记录（2026-05-23）
### 0. 动机升级轨迹（前置背景）

**原始计划**：仿照自己之前的一个独立研究项目模式，自己跑一堆常用模型作为基线，统一对比，画 Pareto frontier。目标是把 NGAFID 二分类做成有客观对比表的"小 benchmark"。

**触发转变**：搜索发现已有大量复现，但质量参差不齐且任务较难，容易失败，比如：
- 欠拟合（如MiniRocket）
- 过拟合（如某些LSTM/Transformer架构）
- 数据泄露（有人准确率虚高到 76%+，后诊断出是划分方式错误）
- 有人只跑单模型、无对比框架
- 目前尚未有人系统测试轻量架构（<<1M 参数）

**结论**：单纯"自己跑基线"会重复造轮子，且无法解决现有复现的**可复现性危机**。必须引入**收录标准 + 审计流程**，把 AMBC 从"个人实验汇总"升级为"可验证的评估基础设施"。

### 1. 任务定位升级
- 原目标：完成导师测试题（NGAFID 维护前/后二分类复现+改进）
- 升级后：将数据集转化为非正式 benchmark（AMBC），建立可复现收录标准与 Pareto frontier
- 文献调研：关于如何建立benchmark的教程

### 2. 收录标准确立
- 硬性门槛：代码可下载、任务一致（3.2节二分类）、CV策略透明、指标≥2个（F1+AUC+Recall）、变长处理说明
- 一票否决：数据泄露嫌疑、无法复现、任务黑箱
- 分级标签：Verified / Pending / Unverified

### 3. 指标体系确立
- 三维度：性能（F1 / ROC-AUC / Recall）/ 算力（参数量）/ 速度（单航班推理延迟 ms）
- 不加可解释性维度（超出当前能力边界，避免硬凑）

### 4. 第一轮学术引用搜索（agent）
- 原论文被引 11 次，2026 年集中爆发
- 发现目标导师胡杨（Yang Hu）的 LiteInception（81.92%，代码未公开）与已撤稿 LMSD
- **关键纠偏**：`hyang0129` = 数据集原作者杨鸿（Hong Yang，RIT），非导师

### 5. 第二轮代码/社区搜索（agent）
- 筛选出 21 个结果，排除重复后 6 个 Verified 候选
- 重点审计两个"科班标答"：
  - `ASpeacialTest-20260417`：Jupyter Notebook 风，MiniRocket，归一化+长度消融
  - `0221lele/ngafid-minirocket`：Python 脚本风，MiniRocket，工程化封装
- 两人结果完全一致（Acc 0.5894±0.0148），验证"AI 标答收敛"现象

### 6. 审计发现：软泄露
- 来源：`0221lele` 的 AI 协作日志
- 问题：官方 `flight_header.csv` 含固定 `fold` 列，但复现者常用 `StratifiedKFold(shuffle=True)` 重新随机划分，导致同机型/同维护事件信息泄露至验证集，准确率虚高
- 影响：直接成为 AMBC 收录标准中"防泄露"条款的实证依据

### 7. 项目基础设施
- 命名：AMBC（Aviation Maintenance Binary Classification）
- 目录：`D:\Projects\ambc\`，含 `audit/`（已 clone 两个标答仓库）
- README：中英双语，已撰写

### 8. 待执行
- clone 并审计更多 Verified 候选（SongX-1 81% Mamba、Lctong1021 三阶段复现等）
- 搭建统一 pipeline（config-driven runner + monitor）
- 实现魔改版：1D ResNet-8/10、micro-Transformer（处理变长序列：Adaptive Pooling / Patching）
- 绘制 Pareto frontier（性能 vs 参数量 vs 延迟）

## 标签体系（两层）

### 对外（benchmark 表格）
- **Verified**：已审计可信，进 Pareto frontier
- **Pending**：待审/在审，不进图，放附录
- **Unverified**：不收录

### 对内（审计优先级）
- **直接收录**：来源可信，多方交叉验证，直接标 Verified
- **待审计**：工作完整但有疑点，审完决定 Verified/Pending/Unverified
- **待补充**：缺数字或任务不对齐，补完再审

## 2026-06-10 夜聊审计手记

### 1. 中文复现者的同质化倾向

检索结果显示，2026年4-5月集中出现3-4个独立中文复现项目，特征高度收敛：
- **方法同质化**：全部以MiniRocket为起点，无人尝试其他架构（LSTM/GRU/Transformer/ResNet）
- **评估口径混乱**：CNN+BiMamba使用Sliding Window将航班级任务转为片段级，报告81%准确率，与官方76%不可直接比较
- **时间窗口重合**：全部集中在招新复试前后

**结论**：无语，本来以为有两个人看起来像是做题已经很神奇了，结果四个中文复现看起来全是？？？代码和依赖一个比一个复杂，你们是爽了，做benchmark的人的命就不是命了？（尖锐爆鸣.mp4）

---

### 2. 审计工作的实际痛点

**代码形态灾难**
- 人均Jupyter Notebook（.ipynb），模型定义、数据加载、训练循环、可视化混在一个文件里
- 代码与博客分离：GitHub放代码，CSDN/GitCode放文字说明，两边对不上
- 无requirements.txt，但是里面一大堆奇奇怪怪的依赖

**评估口径陷阱**
- 官方基准：航班级（一个航班一个样本），截断4096，MinMax归一化，五折CV
- 实际复现：什么乱七八糟的都有，还得我自己搞统一
- **结果**：同样数据集，数字从59%到81%都有，但无法直接比较

**环境管理成本**
- 工位电脑base环境被历史作业（就是ResCat）污染（哎我的问题，手滑了），需新建隔离conda环境
- 复杂架构（如BiMamba）依赖C++扩展（mamba-ssm），编译门槛高，不适合benchmark快速迭代
- 结论：轻量架构优先，重模型延后

---

### 3. Benchmarking的元认知

原本以为benchmarking是"跑一堆模型画张表"，实际上（呵呵）：
- **70%工作量在数据对齐**：确保每个人的预处理、划分、评估口径一致
- **20%在环境管理**：让不同来源的代码能在同一个pipeline里跑通
- **10%才是模型本身**：（请输入文本）

**结论**：我以后再也不会笑了.jpg，看来benchmarking这碗饭也不是谁都能吃的，但是谁让我别的啥都不会，就擅长这个呢

---

### 4. 明日优先级

1. 工位电脑新建conda环境（ambc），验证CUDA可用
2. 写统一data_loader.py：官方口径（航班级、4096截断、fold列划分、MinMax归一化）
3. 跑通MiniRocket baseline（~59%），验证pipeline正确性
4. 有余力则收编第一个"插件"（Lctong1021或Kaggle notebook）

**结论**：没有统一pipeline之前，任何"对比"都是伪命题。明天的首要任务不是跑模型，而是把官方口径的dataloader和CV框架搭出来。

## 2026-7-3 重新启动

### 1. 关于数据集
我发现官方给的下载链接都已经凉了，但是好在复现者 SongX-1 不知道从哪弄到了另一个 Google Drive 镜像链接。经过验证，该数据集与论文 Section 3.2 的统计完全吻合（11446 航班，19 类维护事件，before/after 标签分布 5602/5844），确认是官方 2days 子集的忠实副本。算是这个人做了点好事（笑）

### 2. 评估协议定义（AMBC Standard Protocol v1.0）
与 AI 讨论后，我主导定义了统一评估协议，核心原则：
- **输入对齐**：固定 4096 长度（长截尾、短补零），MinMax 归一化，NaN→0
- **训练对齐**：禁止 Sliding Window，一个航班 = 一个训练样本
- **评估对齐**：必须航班级（Flight-level），禁止片段级（Fragment-level）直接算 acc
- **指标对齐**：性能层 = F1（主）+ Accuracy（辅）；效率层 = 推理时间（主）+ 显存（参考）

**我的判断**：之前认为 SongX-1 的 0.810 不可比，根源不是模型强弱，而是"分母变了"（片段数 vs 航班数）。统一协议后，所有模型拉回同一起跑线。

### 3. 目录结构与工程规范
我要求拆分为 `src/`（基础设施）、`experiments/`（实验战场）、`data/`（不提交）、`audit/`（第三方只读）。AI 建议用 `notebooks/`，我拒绝（不用 .ipynb），改为纯脚本。

### 4. 预处理脚本
AI 写了 `src/preprocessing/preprocess.py`，我验证通过。输出统一为 `.npy` 格式，所有模型共用。

### 5. 评估模块
AI 写了 `src/evaluation/metrics.py`，我要求：
- 必须支持航班级聚合（`aggregate_flight_predictions`）
- 必须统一测量推理时间（`measure_latency`）
- Pareto frontier 只取 F1 和 Latency，其他进表格

### 6. MiniRocket 预实验
笔记本 CPU 跑通，num_kernels=10000 结果：Acc=0.5523±0.012, F1=0.5432±0.012。与官方论文 0.598 有差距，但**流水线验证通过**（数据流、评估、结果 JSON 格式全对）。

**我的判断**：预实验目标不是复现别人的数据，而是证明"统一协议下的横向对比机制"能跑通，因此暂时搁置这个问题。

### 7. 关于 BiMamba 的 Bug
AI 最初指出 SongX-1 的 `BiMambaBlock` 有 Padding 污染（`torch.flip` 把 padding 翻到序列前端）。但我发现：**标准协议下固定 4096 无 padding，bug 自动消失**。因此主实验不改架构，保留原版代码。

**我的决策**：不修 bug。标准协议本身就是最干净的修复。如果后续做变长扩展，再考虑逐样本 Flip 的修复方案。

### 8. Git 管理
遇到 `.gitignore` 对嵌套 Git 仓库（`audit/` 下的 clone 目录）不生效的问题。AI 建议用 `git submodule`，我拒绝（增加复杂度）。最终方案：手动逐个 `git add` 避开 `audit/` 和 `data/`，并重写 `.gitignore` 确保编码正确。

### 9. 官方基线脚本
AI 写了 ConvMHSA 和 InceptionTime 的 TF 脚本，待工位电脑（CUDA 环境）统一跑。笔记本不跑 TF（无 GPU 太慢）。

### 10. 下一步
- 笔记本：写 SongX-1 CNN-only（`mamba_layers=0`），验证 PyTorch 流水线
- 工位电脑：统一跑 TF 基线 + BiMamba 完整版
- 汇总：Pareto 图 + 审计附表
