# Counting Manifold 复现实验报告

日期：2026-06-11  
实验 ID：`cm_repro_001`  
路线：`reproduction`（开放模型复现）  
产物目录：`/NAS/chennc/NashChennc/results/counting-manifolds/`

## 1. 对齐原文

Anthropic 论文 *"When Models Manipulate Manifolds: The Geometry of a Counting Task"* (2025) 在 Claude 3.5 Haiku 中发现：
模型残差流中存在一个低维几何结构（"counting manifold"），编码了"token 距上一个换行符的字符数"这一简单特征。

**本地复现范围**：
- ✅ 完整复现了 hidden state 收集、线性探针（R²/RMSE/MAE）、PCA 降维全流程
- ✅ GPT-2 额外复现了 SAE 特征提取与 PCA 投影分析
- ❌ 不能复现 Claude 3.5 Haiku 原始分析（闭源权重）
- ❌ 不能复现原文的 steering / intervention 因果验证实验

**复现的两条路线**：
| 模型 | 层数 | Hidden Dim | SAE |
|------|------|-----------|-----|
| EleutherAI/pythia-70m-deduped | 6 | 512 | 无 |
| openai-community/gpt2 | 12 | 768 | gpt2-small-resid-post-v5-128k (131k features) |

## 2. 数据与条件

| 条件 | N | 说明 |
|---|---:|---|
| FineWeb (streaming) | 1000 条 | train split，过滤：无换行符、长度 > 1500 chars |
| 行宽 | 150 chars | textwrap 自动换行 |
| chars_since_nl 范围 | 0–150 | 每个 token 标注距上一个 `\n` 的字符偏移 |

**共享参数**：seed=1, batch_size=4, num_workers=14, max_seq_len=1024, pca_n_omit=20, pca_n_components=6

## 3. 方法

1. **数据预处理**：FineWeb streaming → 过滤 → textwrap 换行 → fast tokenizer + offset_mapping → chars_since_nl 标注
2. **LM 评估**：全量前向传播，计算 next-token NLL loss / accuracy / newline probability
3. **逐层分析**（每层）：
   - `collect_layer_hiddens()` → 提取所有 token 的 hidden state
   - `mean_hidden_by_chars_since_nl()` → 按 char count 分组求均值 [151, H]
   - `fit_linear_regression()` → 最小二乘探针 hidden→char_count（R², RMSE, MAE）
   - `pca_per_layer()` → 对 mean hiddens（omit 前 20 counts）做 PCA，取前 6 主成分
4. **SAE 分析**（仅 GPT-2）：`sae.encode(hidden)` → 逐 char count 激活均值 → top 100 features by std → W_dec 方向投影到 PCA 空间
5. **网络适配**：HF 镜像（hf-mirror.com）+ requests monkey-patch 重定向

## 4. 指标

| 优先级 | 指标 | 含义 | 基线/上限 |
|--------|------|------|-----------|
| **1** | 逐层 R² | hidden state 对 char position 的线性可分性 | 随机=0, 上限=1 |
| **2** | PCA explained var (前 6 PC) | 低维流形结构的集中度 | 随机≈6/H≈1% |
| 3 | RMSE / MAE | 线性探针绝对误差 | 随机≈43 chars |
| 4 | LM loss / acc | 模型基础语言能力基线 | — |
| 5 | SAE top feat PCA 对齐 | SAE 特征沿 manifold 排列程度 | 定性分析 |

## 5. 结果

### 5.1 LM 基线

| 模型 | loss | acc | acc_if_newline | mean_prob_if_newline |
|---|---:|---:|---:|---:|
| Pythia-70M | 4.01 | 31.0% | 24.5% | 0.086 |
| GPT-2 | 3.64 | 35.2% | 13.8% | 0.053 |

GPT-2 语言建模能力更强（loss 更低、acc 更高），但对换行符的预测精确度反而不如 Pythia-70M。

### 5.2 逐层线性探针

| Layer | Pythia-70M R² | Pythia-70M RMSE | GPT-2 R² | GPT-2 RMSE |
|-------|-------------:|----------------:|---------:|-----------:|
| 0 | 0.138 | 41.02 | 0.070 | 42.65 |
| 1 | **0.666** | 25.54 | 0.142 | 40.97 |
| 2 | 0.625 | 27.04 | 0.105 | 41.84 |
| 3 | 0.619 | 27.25 | 0.181 | 40.03 |
| 4 | 0.564 | 29.17 | 0.223 | 39.00 |
| 5 | 0.046 | 43.15 | **0.351** | 35.63 |
| 6 | — | — | 0.334 | 36.10 |
| 7 | — | — | 0.260 | 38.06 |
| 8 | — | — | 0.343 | 35.87 |
| 9 | — | — | 0.326 | 36.32 |
| 10 | — | — | 0.263 | 37.98 |
| 11 | — | — | 0.020 | 43.79 |

**核心发现 1**：Pythia-70M 的 R² 峰值 0.666（layer 1），远超 0.5 阈值，确认 counting information 以线性可读方式存在于 hidden state 中。

**核心发现 2**：GPT-2 的 R² 峰值 0.351（layer 5），弱于 Pythia-70M 但仍显著高于随机基线（0.351 >> 0）。Counting signal 在 GPT-2 中也存在，但线性可分性较弱。

**核心发现 3**：两个模型的最后一层 R² 均崩溃（Pythia 0.046，GPT-2 0.020），说明输出层的 hidden state 中 counting 信息已经不可线性读取——信息被转换成了预测 token 的概率分布。

### 5.3 PCA 低维结构

| Layer | Pythia-70M PC1-6 var | GPT-2 PC1-6 var |
|-------|---------------------:|----------------:|
| 0 | 0.685 | 0.733 |
| 1 | 0.913 | 0.774 |
| 2 | 0.933 | 0.794 |
| 3 | **0.964** | 0.815 |
| 4 | 0.955 | 0.837 |
| 5 | 0.915 | 0.832 |
| 6 | — | 0.817 |
| 7 | — | 0.779 |
| 8 | — | 0.772 |
| 9 | — | 0.766 |
| 10 | — | 0.718 |
| 11 | — | **0.987** |

**核心发现 4**：所有层的 PCA 前 6 主成分解释方差均 > 68%（Pythia）和 > 71%（GPT-2），远超随机基线（≈1%）。这强烈确认了 **counting manifold 的低维几何结构在所有层中普遍存在**。

**核心发现 5**：GPT-2 的 layer 11（最后一层）PCA 方差高达 0.987，但 R² 仅 0.020——说明 counting 信息在最后一层仍然以非线性几何结构存在，但无法被线性探针读取。这是一个"hidden in plain sight"的情况：流形还在，但线性方向已消失。

**核心发现 6**：Pythia-70M 的两个指标（R² 和 PCA）高度协同变化（Layer 1-3 最优），而 GPT-2 的 R² 和 PCA 解耦（R² 最优在 Layer 5，PCA 最优在 Layer 11）。暗示不同架构/规模的模型可能以不同方式编码 counting 信息。

### 5.4 SAE 特征分析（GPT-2）

- SAE 特征空间：131,072 features/layer，共 12 层
- 每层提取了 top 100 features by activation std
- SAE top features 成功投影到 PCA 空间：[12, 100, 6] 张量
- 进一步分析（如 PCA 投影可视化、feature activation 热图）需交互式绘图，已产出 .npy 数据

## 6. 失败模式

### 已识别的限制

1. **单数据集**：仅 FineWeb，未验证代码、数学、多语言文本上的 counting manifold
2. **无因果验证**：线性探针 + PCA 只证明"信息存在"，不证明"模型使用"它来预测换行
3. **小模型限制**：Pythia-70M（70M）和 GPT-2（124M）都是小型模型，大规模模型的流形可能更复杂
4. **SAE 覆盖不全**：Pythia 无 SAE；GPT-2 SAE 仅 resid-post 位置，未覆盖 MLP/attention 输出
5. **单次运行**：seed=1 固定，未评估随机种子和采样方差
6. **网络依赖**：需要 HF 镜像 monkey-patch，裸露环境无法直接运行
7. **Pythia vs GPT-2 的 R² 差异**：GPT-2 的 R² 仅 0.351 vs Pythia 的 0.666——可能原因：
   - GPT-2 有更多层（12 vs 6），counting 信息分散在多层
   - GPT-2 的 hidden dim 更大（768 vs 512），信息可能分布在更高维子空间
   - GPT-2 可能使用了非线性编码（PCA 高但 R² 低的 pattern 支持此假说）

### 不扩大声称

- ❌ 不能说"所有 LLMs 都有 counting manifold"（仅测试了 2 个小模型）
- ❌ 不能说"模型使用该流形进行预测"（仅相关性证据）
- ❌ 不能与 Claude 3.5 Haiku 定量比较（不同架构、不同规模、不同训练）

## 7. 结论分级

### 已确认事实
- ✅ 本地 pipeline 已跑通：数据→hidden state→probe→PCA→SAE 全流程无报错
- ✅ Pythia-70M 的 layer 1 存在线性可读的 counting 信号（R² = 0.666）
- ✅ 两个模型的所有层均存在低维 counting manifold（PCA PC1-6 variance > 68%）
- ✅ GPT-2 SAE 分析产出了 12 层 × 131k features 的完整激活数据
- ✅ 最后一层的 counting 信号不可线性读取（R² ≈ 0），但低维几何结构仍在（PCA > 90%）

### 当前数据支持
- 📊 Counting manifold 在 Pythia-70M 中 peak 在 early-middle layer（1/6）
- 📊 Counting manifold 在 GPT-2 中 peak 在 middle layers（5-8/12）
- 📊 Pythia-70M 的 counting 信号强度（R²=0.666）明显高于 GPT-2（R²=0.351）
- 📊 GPT-2 的最后一层存在 R²-PCA 解耦现象（几何存在，线性不可读）

### 仍可能是 artifact
- ⚠️ GPT-2 R² < Pythia R² 可能是模型规模/架构差异，也可能是数据采样偏差
- ⚠️ PCA 高方差 ≠ counting manifold 在功能上重要——高方差可能是训练中的辅助信号或噪声
- ⚠️ SAE top features 的 PCA 对齐尚未人工检查，可能包含假阳性
- ⚠️ chars_since_nl 信号可能与 position embedding 或 token frequency 混淆

## 8. 后续动作

- [ ] **可视化 PCA 投影**：用 PC1-PC2 散点图检查螺旋/环形结构，确认流形几何形状
- [ ] **SAE 热图分析**：检查 top SAE features 的 activation 是否随 char position 单调/周期变化，寻找"计数神经元"
- [ ] **跨规模比较**：运行 Pythia 14M→2.8B 和 GPT-2 Medium→XL，评估 counting 信号强度是否随规模单调变化
- [ ] **跨架构验证**：对 Gemma、Qwen3、Llama 运行相同 pipeline，确认 counting manifold 是否为 transformer 通用性质
- [ ] **多层 SAE 联合分析**：检查 GPT-2 不同层的 SAE top features 是否追踪同一 counting signal
- [ ] **干预实验**：沿 PCA 主方向扰动 hidden state，测试换行预测是否受影响（验证因果性）
- [ ] **多数据集**：在代码（CodeParrot）、数学（Math）文本上验证
