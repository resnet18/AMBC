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
- 无requirements.txt里面一大堆奇奇怪怪的依赖

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