# 汇报方案 — Counting Manifolds 复现

## 判定标准

### 成功（确认复现 counting manifold）
- Pythia-70M 和 GPT-2 均正常完成全流程，无 NaN/OOM/崩溃
- 线性探针 R² > 0.5（至少某一层），证明 hidden state 包含 char position 信息
- PCA 前 6 个主成分解释方差 > 50%（至少某一层），确认低维流形结构
- GPT-2 SAE 分析产出 top features 并成功投影到 PCA 空间

### 存疑（需要进一步检查）
- R² 在所有层均 < 0.3：counting signal 弱，可能需要更多样本或调整参数
- PCA explained variance < 30%：低维结构不明显
- SAE features 投影与 PCA 方向无对齐：SAE 未捕获 counting 相关特征
- 仅一个模型成功：可能为模型特异的 artifact

### 失败（复现未成功）
- 实验崩溃（OOM、CUDA error、数据 pipeline 错误）
- 产物缺失或大小为 0
- 指标与随机基线无显著差异（R² ≈ 0）

## 可视化清单

| 图/表 | 用途 | 数据来源 |
|-------|------|---------|
| 逐层 R² 曲线 | 展示 counting 信息在哪一层最强 | `metrics.csv` layer vs r2 |
| 逐层 RMSE 曲线 | 补充 R²，展示绝对误差 | `metrics.csv` layer vs rmse |
| PCA explained variance 曲线 | 展示低维结构集中度 | `explained_variance_ratio.npy` 逐层累积 |
| PCA 投影散点图（前2 PC） | 可视化 counting manifold 的螺旋/环形结构 | `mean_hiddens_pca_slice_*.npy` |
| LM 指标汇总表 | 模型基础语言能力基线 | `lm_metrics.csv` |
| SAE top features 在 PCA 空间的投影（仅 GPT-2） | 展示 SAE 特征与 counting manifold 的对齐 | `top_sae_features_projected.npy` |
| SAE feature activations 热图（仅 GPT-2） | 展示 top features 如何随 char position 变化 | `sae_mean_top_acts.npy` |

## 指标优先级

1. **逐层 R²**（主指标）— 直接衡量 hidden state 中 char position 信息的线性可分性
2. **PCA explained variance（前 6 PC）**（主指标）— 衡量低维流形结构的集中度
3. **RMSE / MAE**（辅助指标）— 线性探针的绝对误差
4. **LM loss / accuracy**（基线指标）— 确认模型正常工作
5. **SAE top features 的 PCA 对齐**（定性分析，仅 GPT-2）— 解释性

## Baseline

- **随机基线**：R² ≈ 0（线性回归无预测能力），RMSE ≈ std(chars_since_nl) ≈ 43
- **上限**：R² → 1.0（完美线性编码）
- **文献参考**：Anthropic 原文在 Claude 3.5 Haiku 上报告存在清晰的低维流形结构
- **对比**：如果有多个模型的复现结果，比较不同模型家族（Pythia vs GPT-2）的流形质量差异
