# 下一步建议 — Counting Manifold 复现

## 立即可以做的（数据已产出）

1. **PCA 可视化**：用已有 `mean_hiddens_pca_slice_*.npy` 绘制 PC1-PC2 散点图，按 char position 着色，检查螺旋/环形结构
2. **SAE 热图**：用 `sae_mean_top_acts.npy` 绘制 top K features 的 activation × char position 热图
3. **SAE 对齐检查**：用 `top_sae_features_projected.npy` 在 PCA 空间中标出 top SAE features 方向，检查是否沿流形排列
4. **逐层对比**：叠加两个模型的 R²、PCA var、RMSE 曲线，分析层间差异

## 短期扩展（已有 config，改参数即可）

5. **跨规模 Pythia**：运行 pythia-14m, 160m, 410m, 1b, 1.4b, 2.8b — 检验 R² 是否随模型规模单调增长
6. **跨规模 GPT-2**：运行 gpt2-medium, large, xl — 检验 SAE 特征的 counting 特异性是否随规模增强
7. **跨架构**：运行 gemma-2-2b, qwen3-0.6b — 检验不同架构是否共享相同 counting manifold 几何

## 中期实验（需新代码）

8. **因果验证**：沿 PCA PC1 方向 ±ε 扰动 hidden state，测量换行预测概率变化
9. **非线性探针**：训练 MLP / RBF 探针替代线性探针，验证 GPT-2 的非线性编码假说
10. **多数据集验证**：在代码、数学、多语言文本上运行，检验 manifold 是否 domain-specific

## 成本优化

- Pythia-70M 运行时间 ~2 分钟，GPT-2 ~24 分钟（含 SAE）
- 跨模型扩展预计每人时 1-2 个 GPU-小时（小模型快，大模型慢）
- 可批量运行不同 config，利用 `family_metrics.py` 自动汇总
- 网络：使用 HF 镜像 + monkey-patch，无需代理

## 优先级排序

| 优先级 | 动作 | 预期信息价值 | 成本 |
|--------|------|-------------|------|
| P0 | PCA 可视化 + SAE 热图 | 最高——直接验证核心假说 | 低（数据已有） |
| P1 | 跨规模比较（Pythia/GPT-2 家族） | 高——回答"是否随规模增强" | 中（需跑更多模型） |
| P2 | 因果干预实验 | 最高——验证因果性 | 高（需新代码） |
| P3 | 跨架构验证 | 中——回答"是否通用性质" | 中 |
| P4 | 非线性探针 | 中——解释 GPT-2 的低 R² | 低 |
| P5 | 多数据集 | 低——domain 泛化性 | 中 |
