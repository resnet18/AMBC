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
- 发现LiteInception（81.92%，代码未公开）与已撤稿 LMSD

### 5. 第二轮代码/社区搜索（agent）
- 筛选出 21 个结果，排除重复后 6 个 Verified 候选
- 重点审计两个"标准答案"：
  - `ASpeacialTest-20260417`：Jupyter Notebook 风，MiniRocket，归一化+长度消融
  - `0221lele/ngafid-minirocket`：Python 脚本风，MiniRocket，工程化封装
- 两人结果完全一致（Acc 0.5894±0.0148），验证"AI标答收敛"现象

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
**结论**：无语，本来以为有两个人看起来像是做题已经很神奇了，结果四个中文复现看起来全是？代码和依赖一个比一个复杂，你们是爽了，做benchmark的人的命就不是命了？（尖锐爆鸣.mp4）

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

### 3. Benchmarking的元认知
原本以为benchmarking是"跑一堆模型画张表"，实际上（呵呵）：
- **70%工作量在数据对齐**：确保每个人的预处理、划分、评估口径一致
- **20%在环境管理**：让不同来源的代码能在同一个pipeline里跑通
- **10%才是模型本身**：（请输入文本）

**结论**：我以后再也不会笑了.jpg，看来benchmarking这碗饭也不是谁都能吃的，但是谁让我别的啥都不会，就擅长这个呢（？）

## 2026-7-3 重新启动

### 1. 关于数据集
我发现官方给的下载链接都已经凉了，但是好在复现者 SongX-1 自己弄了一个Google Drive镜像链接。经过验证，该数据集与论文 Section 3.2 的统计完全吻合（11446 航班，19 类维护事件，before/after 标签分布 5602/5844），确认是官方2days子集的忠实副本。算是这个人做了点好事（笑）
*（注：之前没注意，实际上这个人在readme写了就是下载了2days子集然后上传到自己的网盘，而且官方在zenodo上放的数据集本来下载就很方便，有2days子集这个数据包，那没事了）*

### 2. 评估协议定义（AMBC Standard Protocol v1.0）
与 AI 讨论后，我主导定义了统一评估协议，核心原则：
- **输入对齐**：固定 4096 长度（长截尾、短补零），MinMax 归一化，NaN→0（按照原数据集论文的处理方式）
- **训练对齐**：禁止 Sliding Window，一个航班 = 一个训练样本（同样，**其他任何形式的数据增强也是不被允许的**）
- **评估对齐**：必须航班级（Flight-level），禁止片段级（Fragment-level）直接算 acc
- **指标对齐**：性能层 = F1（主）+ Accuracy（辅）；效率层 = 推理时间（主）+ 显存（参考）
之前认为 SongX-1 的 0.810 不可比，根源不是模型强弱，而是"分母变了"（片段数 vs 航班数）。统一协议后，所有模型拉回同一起跑线。

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

**判断**：预实验目标不是复现别人的数据，而是证明"统一协议下的横向对比机制"能跑通，因此暂时搁置这个问题。

### 7. 关于 BiMamba 的 Bug
AI 最初指出 SongX-1 的 `BiMambaBlock` 有 Padding 污染（`torch.flip` 把 padding 翻到序列前端）。但是后来又发现：**标准协议下固定 4096 无 padding，bug 自动消失**。

**决策**：不修 bug，如果后续做变长扩展，再考虑逐样本 Flip 的修复方案。

### 8. Git 管理
遇到 `.gitignore` 对嵌套 Git 仓库（`audit/` 下的 clone 目录）不生效的问题。AI 建议用 `git submodule`，我拒绝（增加复杂度）。最终方案：手动逐个 `git add` 避开 `audit/` 和 `data/`，并重写 `.gitignore` 确保编码正确。

### 9. 官方基线脚本
AI 写了 ConvMHSA 和 InceptionTime 的 TF 脚本，待工位电脑（CUDA 环境）统一跑。笔记本不跑 TF（没GPU，太慢）。

### 10. 下一步
- 笔记本：写 SongX-1 CNN-only（`mamba_layers=0`），验证 PyTorch 流水线
- 工位电脑：统一跑 TF 基线 + BiMamba 完整版

## 2026-7-4 至 2026-7-8 进度

### 1. 社区复现补充
我在网络搜索时偶然发现之前的AI搜索遗漏了两个来源：
- **知乎 aq-void**：分析质量挺高，就是代码一个字都没有，没法收录（这很讨厌）。指出官方 ConvMHSA 存在四大工程缺陷（无 BN、无位置编码、无动态学习率、ReLU 而非 GELU），并报告了规范实现下的结果 **Acc 0.7755**（但是没法复现/交叉验证的结果我一个字都不会信）。
- **gitcode m0_64843793**：MiniRocket **0.7185**，与自己的 0.7187 几乎一致。这证明了 sktime 路线在正确实现下的准确率就是 0.72 左右，不是异常。

**判断**：官方 MiniRocket（tsai 路线）和 sktime MiniRocket 是两个不同的算法实现，~~收敛值差 20%（0.60 vs 0.72），~~*（注：关于此处删去的原因见后续）*这一点是时序特有的，跟之前的NLP区别很大，需要注意。

**决策**：在 AMBC 协议中明确标注"实现差异"，同时收录sktime与tsai实现。

### 2. MiniRocket代码里的维度Bug
**AI 最初没发现这个问题。** 跑通 MiniRocket 后得到 ~0.55 的结果，与官方 0.598 差距明显。AI 首先怀疑是 StandardScaler 的问题，建议去掉 scaler 试试。但是我不是很相信，就先让ai重新review我的代码。我发现 sktime 的 `MiniRocketMultivariate` 文档里写的输入格式是 `(N, C, T)`，而我们的预处理输出是 `(N, T, C)`。这意味着 **4096 个时间步被当成了"通道"，23 个传感器被当成了"时间步"**——卷积核在 23 长度的伪序列上滑动，提取的不是时序模式，而是传感器间的瞬时相关性。

**修复**：在 `load_fold` 里加 `np.transpose(X_train, (0, 2, 1))`，把 `(N, 4096, 23)` 转成 `(N, 23, 4096)`。

**结果**：Acc 从 0.5523 直接跳到 **0.7146**。这是整个项目中最关键的 bug 修复，没有之一。

**后续**：AI 建议我把 StandardScaler 也去掉，说 MiniRocket 特征不需要缩放。我照做了，但发现加不加 scaler 差距只有 0.4%（0.6754 vs 0.6793），影响很小。这说明 AI 的"StandardScaler 有害"假设是过度归因，真正的原因是维度问题。

**教训**：不要轻信 AI 的"建议"，先检查代码有没有bug。

### 3. 一个诚实的失败案例
**AI 搜索发现的第三个"标答"**，但结果与前两个完全不同：
- MiniRocket 0.5251，ConvMHSA 0.5240，InceptionTime 0.4887
- 原因：Colab 免费版 GPU OOM，被迫缩减 batch_size 和 epochs

**判断**：在有限算力下，深度学习baseline性能崩溃至随机水平。

**决策**：不纳入有效结果，但作为附录观察记录。这成为 AMBC 必须包含"效率/算力"维度的实证依据。

### 4.关于禁止数据增强
出于控制变量的考虑，我最开始就决定禁止任何形式的数据增强，一律用数据集论文的预处理方式，审计过程中看到社区复现没少做数据增强（Window Slide之类），感觉这个选择还是挺明智的（笑）

### 5. 关于部分实现的环境问题
由于 mamba_ssm 和 TensorFlow 的兼容性问题，决定部分基线使用 WSL（已确认对推理时间和算力测量无影响）。具体而言：
- **mamba-ssm**：官方仅提供 Linux 预编译 wheel，Windows 需从源码编译 CUDA 扩展，成功率极低。WSL2 Ubuntu 环境已搭建完成，conda 虚拟环境 `ambc` 已配置 PyTorch cu132。
- **TensorFlow**：TF >= 2.11 在 Windows 原生环境下彻底移除 GPU 支持，DirectML 插件性能差且兼容性随缘。5060 Ti 的 Blackwell 架构（sm_120）也不支持回退到 TF 2.10 + CUDA 11.2 的方案。
- **决策**：InceptionTime、ConvMHSA 等 TF 基线，以及 BiMamba，全部统一迁移至 WSL2 的独立 conda 环境中运行。Windows 环境仅保留 PyTorch（cu132）基线。

### 6. 一个重大发现（笑死我了）
之前一直困惑于官方 MiniRocket 59.8% 与 AMBC 自用的 72% 之间的巨大差距。AI 曾推测是"tsai 的 NN 头 vs sktime 的 Ridge 分类器"导致，直到自己用tsai试了一遍发现还是约72%，以为代码出了什么bug，就下载了官方Colab的完整 `.py` 文件进行逐行审计，发现官方代码中：
```python
mrf = MiniRocketFeatures(train_X.shape[1], train_X.shape[2])
```
输入数据为 `(N, 4096, 23)`，但官方**未进行通道-时间步转置**，导致 `c_in=4096, seq_len=23`。MiniRocket 的 1D 卷积核在 4096 个"伪通道"上、每个通道仅 23 个时间步运行，dilation 和感受野逻辑完全失效。

**验证**：做了一个消融实验，在自己的 tsai 实现中**故意注释掉 `np.transpose`**，完全复刻官方配置，结果准确率为 **0.5895 ± 0.0094**，与官方 59.8%、标答 A/B 的 0.5894 几乎完全一致。修正后（`c_in=23, seq_len=4096`）结果回到 **0.7318**，差距约 13 个百分点。

**影响范围**：官方 Colab、标答 A、标答 B 三方复制同一错误。交叉验证在此案例中不仅未发现问题，反而成了错误传播的放大器。

**决策**：AMBC v2.0 的"直接收录"分类被证伪，全部降级为"审计中"。MiniRocket 基线以 AMBC 自用正确实现（72-73%）为准，官方 59.8% 标记为"已确认缺陷"。

**教训**：交叉验证一致 ≠ 实现正确。对于基准测试来说，必须默认所有的来源都不可靠，亲自复现过才算，审计必须包含代码审查，不能因为任何原因而在实现细节的验证方面偷懒。

#### 影响范围与实现路线对比

| # | 来源 | 实现路线 | 维度处理 | 结果 | 状态 | 备注 |
|---|------|----------|----------|------|------|------|
| 1 | 官方 Colab (Yang et al.) | tsai | 未转置 `(N, 4096, 23)` | ~0.598 | 缺陷源头 | `c_in=4096, seq_len=23`，卷积核在伪通道上运行 |
| 2 | 标答 A | tsai | 未转置 | ~0.59 | 复制缺陷 | 直接复制官方代码 |
| 3 | 标答 B | tsai | 未转置 | ~0.59 | 复制缺陷 | 同上，另发现软泄露并自行修正 |
| 4 | SongX-1 | tsai | 未转置 | ~0.59 | 复制缺陷 | README 中记录 MiniRocket 基线，未审计 |
| 5 | ccDT2022 | tsai | 未转置 | ~0.53 | 复制缺陷+缩参 | 资源因素缩小参数，结果更差 |
| 6 | m0_64843793 | Numpy+sklearn | 主动转置 `(N, 23, 4096)` | ~0.72 | 正确实现 | 代码中显式 `np.transpose(x, (0, 2, 1))`，但归因于"参数优化" |
| 7 | Lctong1021 | sktime | 内部自动处理 | ~0.72 | 正确实现 | sktime 内部容错，无需用户手动转置 |
| 8 | **AMBC** | sktime/tsai 双路线 | tsai版修复 | **~0.72-0.73** | 正确实现 | 通过双路线对比发现官方 bug |

> **注**：m0_64843793 并非使用 sktime 的现成实现，而是基于 NumPy 手搓 MiniRocket 特征提取 + sklearn LogisticRegression。其代码中显式包含 `np.transpose(x, (0, 2, 1))`，说明作者主动理解并修复了维度问题。然而，由于未考据官方实现的细节，作者将 71.85% 的高准确率归因于"参数优化"，未能识别出官方 59.8% 本身即为缺陷结果。

#### 为什么tsai路线集体中招，而sktime路线逃过一劫？
sktime 的 MiniRocketMultivariate 采用自动推断设计：用户传入 (N, T, C) 或 (N, C, T) 均可，库内部自动检测并转换维度，无需手动干预。而 tsai 的 MiniRocketFeatures(c_in, seq_len) 是显式声明设计：用户必须明确告诉它"通道数是多少、时间步是多少"，库本身不做形状校验，给什么参数就按什么参数运行。
NGAFID 预处理后的数据格式是 (N, 4096, 23)（时间步 × 传感器），但 tsai 要求 (N, 23, 4096)（传感器 × 时间步）。官方Colab代码直接取了 train_X.shape[1] 和 train_X.shape[2]，未进行转置，导致 c_in=4096, seq_len=23——MiniRocket 的 1D 卷积核在 4096 个"伪通道"上、每个通道仅 23 个时间步运行，dilation 逻辑完全失效。
这一设计差异决定了传播路径：使用 sktime 的复现者（Lctong1021）因库的自动容错而无意识幸存；使用 tsai 的复现者（官方、标答 A/B、SongX-1、ccDT2022）因为没有容错和照搬官方代码（未转置）而集体中招。另外，复现者m0_64843793的手搓版实现中显式加入了np.transpose，同样避开了陷阱，这说明该错误并非依赖问题，而是tsai的显式参数设计与官方代码的疏忽共同导致的系统性陷阱。若官方代码正确转置，tsai与sktime的结果应基本一致（tsai的NN分类头可能略高 1–2%），笔者的消融实验也证明了这一点。

### 7. “别人：MiniRocket，我：Mamba，就很...”
WSL2 环境终于把 mamba-ssm 编译通过了（最费劲的部分），README 里写得挺唬人，CNN+BiMamba 81.0%，CNN-only 79.0%，把双向Mamba从NLP领域迁移过来，听起来很牛逼，但标准协议跑下来：
- **CNN-only**: Acc 0.7331±0.0214, F1 0.7236±0.0157, Latency 2.84 ms
- **BiMamba**: Acc 0.7063±0.0397, F1 0.6820±0.1117, Latency 6.09 ms
笑点解析：**BiMamba 不仅没提升，反而掉了3个百分点，F1的标准差直接爆炸到0.11**（说明某些fold上崩得亲妈都不认），更重量级的是BiMamba在这个任务上表现还不如纯CNN。
所以那个能跑出 81% 纯属因为**片段级（fragment-level）评估**。把一条航班切成多个滑动窗口片段，训练数据量变相膨胀，评估时以片段为单位算准确率，数字自然好看。但是统一基准之后就原形毕露了。
说白了，这位复现者把 NLP 里常用的"文本切分+滑动窗口"思路搬过来，在航空维护场景里水土不服。NLP 里双向建模能捕捉上下文依赖，但航空传感器数据是**多变量时间序列**，23 个传感器之间的空间关联和 4096 个时间步上的时序依赖是两回事。Mamba 的 SSM 机制在这个数据尺度上（4096 步，不算特别长）并没有展现出比 CNN 更强的预测能力，反而因为参数增加和双向计算的复杂性，拖慢了推理速度（6ms vs 2.8ms），稳定性还更差，唯一的好处在于给AMBC提供了一个新架构的样本（这倒是真的），虽然是作为反面教材。

**启示**：绝大多数时候，跨领域迁移不是换个模块名字那么简单（笔者也算是从NLP过来的，最开始完全就是自在极意豪，想着Transformer会不会在这里继续赢家通吃，但是被ai直接泼冷水）。NLP 里的双向架构在时序里不一定 work，所以当评估协议严格限制"不能切片段"的时候，真实性能会原形毕露。另外，mamba-ssm 在 WSL2 里装了一晚上，CUDA 13.2 配环境配到怀疑人生，结果跑出来不如CNN，我直接轻松绷住.jpg

### 8. 勘误：关于官方仓库来源的混淆
之前AI 初次全网搜索时，将 `tinyurl.com/b35mxv98` 标记为"官方基准 (Yang et al., AAAI 2022)"，并作为 MiniRocket / InceptionTime / ConvMHSA 的引用来源。
**更正**：经过直接下载源码审计，该链接实际指向的是 IAAI-22（与 AAAI-22 合办）论文 *Predictive Maintenance for General Aviation Using Convolutional Transformers* 的实验复现 Colab，而非 NGAFID 数据集论文的官方实现。
**关键区分**：
- **IAAI-22 论文**：使用 C28/C37 子集（7,505 航班），30 epochs，batch 32，含大量数据增强（Cutout/Mixup/Cutmix）。这是 ConvMHSA 架构的首次发表。
- **数据集论文（arXiv 2022）**：使用 `2days` 子集（11,446 航班），200 epochs，batch 128，无数据增强。ConvMHSA 仅作为 baseline 被复现，架构相同但训练协议完全不同。

**AMBC 处理**：由于两个工作使用的数据集子集不同（C28/C37 vs 2days），数字不可直接比较。AMBC 仅审计数据集论文的 `2days` 子集及配套 Colab（`NGAFID_DATASET_MINIROCKET_EXAMPLE.ipynb` 与 `NGAFID_DATASET_TF_EXAMPLE.ipynb`），IAAI-22 的复现仓库与 AMBC 标准协议无关，不再追踪。

**教训**：AI搜索的结果需要人工二次验证

### 9. 增加架构收录原则
因为看到很多社区复现者会自己做调参和消融，所以增加了这一条（避免benchmark无限膨胀）：
**不收录的**：
- **调参**：改学习率、epochs、batch size、特征数，这是炼丹，不是新架构
- **消融**：比如 CNN+BiMamba 架构里只留下CNN，这是原技术报告自己该做的诊断，不作为主要结果
- **数据增强**：用了评估协议3.1禁止的数据增强的， 改成标准协议，改不了的收录进Extended Track并标注

**可以收录的**：
- **同一算法的不同实现路径**：MiniRocket-sktime 和 MiniRocket-tsai 可以并列（并且标注）
- **结构性修改**：把 Mamba 换成 LSTM，或者加了完全不同的特征提取阶段——这算新架构，但得走审核

**判定原则**：如果两条目唯一的区别是"训练了多久"或者"提了多少特征"，非默认的那个直接踢掉。AMBC 比的是**架构**，不是**预算**。

### 10. InceptionTime 和 ConvMHSA 的硬件兼容性问题
原版基准（Yang et al., 2022）是 TensorFlow + Colab TPU v2-8。本地 RTX 5060 Ti（Blackwell / sm_120）完全跑不了TF（参见/archive里的归档版代码注释，这还不像mamba可以开wsl2），所以 InceptionTime 和 ConvMHSA 都重写了 PyTorch 版。
- **InceptionTime**：PyTorch 移植顺利，官方配置（batch size=128, 200 epochs）直接能跑
- **ConvMHSA**：batch=128 会把 16GB 显存吃到 97.8%，触发 allocator thrashing，单 epoch 400 秒（23 小时/fold）。做了 gradient accumulation（micro_batch=64, accumulation_steps=2，等效 batch=128）。代码和结果 JSON 里都打了 `[HARDWARE APPROXIMATION]` 标签。

### 11. 另一些有意思的结果
- **MiniRocket+特征工程**：来自复现者Lctong1021，复现者最后得出的结论是特征工程没啥用，我改成标准协议之后也验证了，确实。
- **InceptionTime**：复现者m0_64843793的结果明显偏低，是因为训练轮数比官方少很多（原因不明，或许是资源/设备限制），出现了欠拟合，按照官方给出的训练轮数训出来结果是一致的。
- **ConvMHSA**：[待补充]

### 12.暂时收尾与后续工作
工作量比预想的大得多，决定到此为止，当前测评视为**第一阶段**，把预实验结果作为依据，不追求完美覆盖。
**未来工作**：
- 继续审计剩下的架构：论文（较复杂/需要改任务）和 Kaggle 实现（代码不完整）
- 放一些自己感兴趣的架构（比如随机森林，ResNet-1D）
- 建立统一虚拟环境和完整测试流水线（现在训练和推理还混在一块，环境也是各跑各的）
- 扩充指标：Precision、AUC-ROC、训练时间等，尽量让评估更贴近实际部署
- 修订协议（如果需要）
**备注**：第一阶段的核心目标是"建立一个完整的测试框架"，而不是"做一个完美的benchmark"。能跑通就行，剩下的慢慢填。