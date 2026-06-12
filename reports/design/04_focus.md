# 重点关注 — Counting Manifolds 复现

## 核心问题（3 个）

1. **Counting manifold 在 Pythia-70M 和 GPT-2 中是否存在？**
   判定：至少某一层的线性探针 R² > 0.5，且 PCA 前 2 个主成分呈现清晰的螺旋/环形结构（char position 的循环特性）。

2. **Counting 信息主要集中在哪一层？**
   判定：观察 R² 和 PCA explained variance 的逐层曲线。原文报告中间层（而非最后层）通常包含最强的 counting 信号。

3. **GPT-2 的 SAE 特征是否与 counting manifold 对齐？**
   判定：top SAE features 在 PCA 空间的投影是否沿 counting manifold 的主要方向排列，是否存在"计数特征"（激活随 char position 单调/周期变化）。

## 止损线

- [ ] 任一模特的线性探针在所有层的 R² < 0.1：counting signal 缺失，检查数据 pipeline（chars_since_nl 标注是否正确）
- [ ] 实验崩溃（OOM/CUDA error）：降低 batch_size 或 num_workers 重试
- [ ] 产物文件缺失或大小为 0：检查磁盘空间、output_path 权限
- [ ] PCA explained variance 在所有层 < 20%：低维结构可能不存在，或 PCA 参数需调整（如 n_omit 太小）

## 惊喜处理

以下发现需要额外验证，不直接采信：

- **某一层 R² 异常高（> 0.9）但相邻层急剧下降**：可能是该层恰好在特定位置过拟合，需检查该层的 PCA 结构是否平滑
- **SAE top feature 在 PCA 投影中恰好通过原点**：可能是 artifact（SAE 训练中的 bias），需检查该 feature 的激活是否对 char position 真正敏感
- **Pythia-70M 和 GPT-2 的 counting manifold 几何形状显著不同**：可能是架构差异（Pythia vs GPT-2）或规模效应，不能直接推广到"所有 open LLMs"
- **某个 char count 位置出现异常偏离**：可能是该位置恰好对应某种语言模式（如行首大写字母），而非纯粹的 counting 信号
