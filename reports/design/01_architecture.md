# 方法理解 — Counting Manifolds

## 核心主张

Anthropic 论文 *"When Models Manipulate Manifolds: The Geometry of a Counting Task"* 的核心主张：

1. **计数流形存在**：语言模型的 hidden state 中存在一个低维几何结构（"counting manifold"），编码了"当前 token 在行内的字符位置"（chars since newline）这一信息。
2. **模型利用该流形**：模型在预测换行符时会利用这个流形结构，而非仅仅依赖表面模式。
3. **SAE 特征可解释该流形**：稀疏自编码器（SAE）的特征方向与 counting manifold 的方向对齐，可用于解释模型的内部计数机制。

本仓库 `corl-team/counting_manifolds` 将这些实验迁移到开放权重模型（Pythia、GPT-2、Gemma、Qwen3、Llama），验证 counting manifold 在 open LLMs 中是否同样存在。

## 输入 → 处理 → 输出

```
FineWeb 原始文本
    │
    ▼
[数据预处理]
  - 过滤：文本长度 > line_length × min_lines（150×10=1500），不含换行符
  - 换行包装：按 150 字符宽度自动换行（textwrap）
  - Tokenization：fast tokenizer + offset_mapping
  - 标注 chars_since_nl：每个 token 距离上一个换行符的字符数
    │
    ▼
[LM 评估] eval_next_token_metrics()
  - 计算 next-token NLL loss、accuracy
  - 换行符预测准确率、概率质量
  - 输出：lm_metrics.csv, newline_probs.npy
    │
    ▼
[逐层分析] for each transformer layer:
  ├── collect_layer_hiddens()     → 收集该层所有 token 的 hidden state
  ├── mean_hidden_by_chars_since_nl() → 按 char count 分组求均值 [151, H]
  ├── fit_linear_regression()     → hidden → char count 线性探针 (R², RMSE, MAE)
  ├── pca_per_layer()             → 对 mean hiddens 做 PCA（omit 前20个count）
  └── (if SAE) get_sae_acts()     → SAE 特征激活 per count → top K features
    │
    ▼
[输出产物]
  metrics.csv          — 逐层 R², RMSE, MAE, PCA variance explained
  mean_hiddens.npy     — [num_layers, 151, hidden_dim]
  explained_variance_ratio.npy
  mean_hiddens_pca_slice_*.npy  — PCA 投影 [num_layers, 131, 6]
  (SAE) sae_mean_acts.npy, top_sae_features_projected.npy 等
```

## 实验设置

- **数据集**：HuggingFaceFW/fineweb（train split, streaming）
- **样本量**：1000 条（过滤后），行宽 150 字符，最少 10 行
- **模型**（本次复现）：
  - EleutherAI/pythia-70m-deduped（6 层, hidden_dim=512, float32）
  - openai-community/gpt2（12 层, hidden_dim=768, float32 + SAE）
- **PCA 参数**：省略前 20 个 char count，取前 6 个主成分
- **评估指标**：
  - LM: loss, accuracy, newline accuracy, newline prob mass
  - Probe: R², RMSE, MAE（线性回归 hidden→char_count）
  - PCA: explained variance ratio（前 6 个 PC）
  - SAE: top 100 features by activation std, PCA 投影

## 开放程度

- [x] 代码开源（Apache 2.0）
- [x] 模型可下载（Pythia-70M, GPT-2 均公开在 HuggingFace）
- [x] 数据公开（FineWeb 公开可用）
- [ ] 仅论文——原文 Claude 3.5 Haiku 内部分析不可复现

## 原文 Limitation

- Anthropic 原文使用 Claude 3.5 Haiku（闭源权重），其内部机制分析不可精确复现
- 原文发现的 counting manifold 详细几何性质可能因模型架构和训练方式不同而有差异
- SAE 覆盖范围受限：仅 GPT-2、Llama-3.1-8B、Gemma-2-9B 有可用 SAE
- 单数据集（FineWeb），未测试其他文本分布

## 本地复现范围

**能复现的**：
- Pythia-70M：完整的 hidden state 收集、线性探针、PCA 分析
- GPT-2：额外包括 SAE 特征分析与 PCA 投影

**不能复现的**：
- Claude 3.5 Haiku 的原始分析（无权重）
- 原文中的 intervention / steering 实验（本仓库未实现）
- 多数据集交叉验证（仅 FineWeb）
