#!/usr/bin/env python3
"""Counting Manifold 可视化脚本。
从 results/ 目录读取 .npy/.csv 产物，生成：
  1. PCA 流形图（PC1-PC2 散点，按 char position 着色）
  2. 逐层指标对比（R², RMSE, PCA variance）
  3. SAE 特征热图 + PCA 空间对齐（仅 GPT-2）
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colormaps

# ── Config ──────────────────────────────────────────────
RESULT_ROOT = Path("/NAS/chennc/NashChennc/results/counting-manifolds")
OUT_DIR = Path("/NAS/chennc/NashChennc/counting_manifolds/reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "pythia-70m": RESULT_ROOT / "pythia-70m-deduped/fineweb",
    "gpt2": RESULT_ROOT / "gpt2/fineweb",
}

# Shared style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def load_pca_projs(model_key: str) -> np.ndarray:
    """Load PCA projections: [n_layers, 151, 6]"""
    p = MODELS[model_key]
    f = sorted(p.glob("mean_hiddens_pca_slice_*.npy"))
    if not f:
        raise FileNotFoundError(f"No PCA slice found in {p}")
    return np.load(f[0])


def load_metrics(model_key: str) -> pd.DataFrame:
    return pd.read_csv(MODELS[model_key] / "metrics.csv")


def load_lm_metrics(model_key: str) -> pd.Series:
    return pd.read_csv(MODELS[model_key] / "lm_metrics.csv").iloc[0]


def load_sae_data(model_key: str) -> dict:
    """Load SAE arrays (GPT-2 only)."""
    p = MODELS[model_key]
    result = {}
    for name in ["sae_mean_top_acts", "sae_mean_indices",
                 "top_sae_features_projected", "top_sae_features_projected_scaled"]:
        f = p / f"{name}.npy"
        if f.exists():
            result[name] = np.load(f)
    return result


# ╔══════════════════════════════════════════════════════╗
# ║  1. PCA 流形图：PC1-PC2 按 char position 着色       ║
# ╚══════════════════════════════════════════════════════╝

def plot_pca_manifold(model_key: str):
    pca = load_pca_projs(model_key)  # [L, 151, 6]
    n_layers = pca.shape[0]
    n_omit = 20
    char_positions = np.arange(151)

    # 用 viridis 按 char position 着色
    cmap = colormaps["viridis"]
    norm = Normalize(vmin=0, vmax=150)

    # 确定子图网格
    cols = min(4, n_layers)
    rows = int(np.ceil(n_layers / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows),
                             squeeze=False)
    fig.suptitle(f"Counting Manifold — PCA Projection ({model_key.upper()})",
                 fontsize=16, fontweight="bold", y=1.01)

    for layer in range(n_layers):
        ax = axes[layer // cols][layer % cols]
        data = pca[layer]  # [151, 6]

        # Omit first n_omit (shown as faint gray)
        if n_omit > 0:
            ax.scatter(data[:n_omit, 0], data[:n_omit, 1],
                       c="gray", alpha=0.25, s=8, zorder=1)

        sc = ax.scatter(data[n_omit:, 0], data[n_omit:, 1],
                        c=char_positions[n_omit:], cmap=cmap, norm=norm,
                        s=18, edgecolors="k", linewidth=0.3, zorder=2)

        # Connect consecutive points
        ax.plot(data[n_omit:, 0], data[n_omit:, 1],
                color="gray", alpha=0.3, linewidth=0.5, zorder=0)

        ax.set_title(f"Layer {layer}", fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_aspect("equal", adjustable="datalim")

    # Hide unused subplots
    for idx in range(n_layers, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    cbar = fig.colorbar(sc, ax=[ax for row in axes for ax in row if ax.get_visible()],
                        label="chars since newline", shrink=0.6, pad=0.02)
    fig.tight_layout()
    out = OUT_DIR / f"pca_manifold_{model_key}.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[1] PCA manifold saved → {out}")


# ╔══════════════════════════════════════════════════════╗
# ║  2. PCA 流形精选（最佳层 + 最差层）                 ║
# ╚══════════════════════════════════════════════════════╝

def plot_pca_best_worst(model_key: str):
    pca = load_pca_projs(model_key)  # [L, 151, 6]
    metrics = load_metrics(model_key)
    n_layers = pca.shape[0]
    n_omit = 20
    char_positions = np.arange(151)

    # 找 best（R² 最高）和 last（最后一层）
    best_layer = metrics["r2"].idxmax()
    last_layer = n_layers - 1

    cmap = colormaps["inferno"]
    norm = Normalize(vmin=0, vmax=150)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Counting Manifold — Best vs Last Layer ({model_key.upper()})",
                 fontsize=14, fontweight="bold")

    for ax, layer, label in [
        (axes[0], best_layer, f"Layer {best_layer} (best R²)"),
        (axes[1], last_layer, f"Layer {last_layer} (last)"),
    ]:
        data = pca[layer]
        if n_omit > 0:
            ax.scatter(data[:n_omit, 0], data[:n_omit, 1],
                       c="gray", alpha=0.2, s=10, zorder=1)
        sc = ax.scatter(data[n_omit:, 0], data[n_omit:, 1],
                        c=char_positions[n_omit:], cmap=cmap, norm=norm,
                        s=25, edgecolors="white", linewidth=0.4, zorder=2)
        ax.plot(data[n_omit:, 0], data[n_omit:, 1],
                color="gray", alpha=0.35, linewidth=0.6, zorder=0)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_aspect("equal", adjustable="datalim")

    cbar = fig.colorbar(sc, ax=axes.tolist(), label="chars since newline",
                        shrink=0.8)
    fig.tight_layout()
    out = OUT_DIR / f"pca_best_worst_{model_key}.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[2] PCA best/worst saved → {out}")


# ╔══════════════════════════════════════════════════════╗
# ║  3. 逐层指标对比图                                   ║
# ╚══════════════════════════════════════════════════════╝

def plot_layer_metrics(pythia_metrics: pd.DataFrame,
                       gpt2_metrics: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Per-Layer Metrics Comparison: Pythia-70M vs GPT-2",
                 fontsize=14, fontweight="bold")

    # R²
    ax = axes[0]
    ax.plot(pythia_metrics["layer"], pythia_metrics["r2"],
            "o-", color="#1A5276", linewidth=2, markersize=8, label="Pythia-70M")
    ax.plot(gpt2_metrics["layer"], gpt2_metrics["r2"],
            "s-", color="#B9770E", linewidth=2, markersize=8, label="GPT-2")
    ax.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="R²=0.5 threshold")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title("Linear Probe R²")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # RMSE
    ax = axes[1]
    ax.plot(pythia_metrics["layer"], pythia_metrics["rmse"],
            "o-", color="#1A5276", linewidth=2, markersize=8, label="Pythia-70M")
    ax.plot(gpt2_metrics["layer"], gpt2_metrics["rmse"],
            "s-", color="#B9770E", linewidth=2, markersize=8, label="GPT-2")
    ax.axhline(y=43.3, color="gray", linestyle="--", alpha=0.5, label="Random baseline (43.3)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("RMSE (chars)")
    ax.set_title("Linear Probe RMSE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # PCA Variance
    ax = axes[2]
    pca_col_p = [c for c in pythia_metrics.columns if "pca_varexp" in c][0]
    pca_col_g = [c for c in gpt2_metrics.columns if "pca_varexp" in c][0]
    ax.plot(pythia_metrics["layer"], pythia_metrics[pca_col_p],
            "o-", color="#1A5276", linewidth=2, markersize=8, label="Pythia-70M")
    ax.plot(gpt2_metrics["layer"], gpt2_metrics[pca_col_g],
            "s-", color="#B9770E", linewidth=2, markersize=8, label="GPT-2")
    ax.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="50% threshold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cumulative Variance")
    ax.set_title("PCA PC1-6 Explained Variance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "layer_metrics_comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[3] Layer metrics saved → {out}")


# ╔══════════════════════════════════════════════════════╗
# ║  4. SAE 特征热图（GPT-2 only）                      ║
# ╚══════════════════════════════════════════════════════╝

def plot_sae_heatmap():
    sae = load_sae_data("gpt2")
    if not sae:
        print("[4] No SAE data found — skipping SAE plots")
        return

    top_acts = sae["sae_mean_top_acts"]       # [L, 150, K]
    top_idx = sae["sae_mean_indices"]          # [L, K]
    n_layers, n_positions, top_k = top_acts.shape

    # 选择关键层：first, middle (best R²), last
    metrics = load_metrics("gpt2")
    best_layer = metrics["r2"].idxmax()
    layers_to_show = [0, best_layer, n_layers - 1]
    layer_labels = [f"Layer {l}" + (" (best R²)" if l == best_layer else
                                     " (first)" if l == 0 else " (last)")
                    for l in layers_to_show]

    cmap = colormaps["inferno"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("GPT-2 SAE Top-20 Feature Activations × Char Position",
                 fontsize=14, fontweight="bold")

    for ax, layer, label in zip(axes, layers_to_show, layer_labels):
        # Show top 20 features
        data = top_acts[layer, 1:, :20].T  # [20, 149] (skip char=0 = BOS)
        im = ax.imshow(data, aspect="auto", cmap=cmap,
                       extent=[1, 150, 20, 1], interpolation="bilinear")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("chars since newline")
        ax.set_ylabel("Top-20 SAE Feature (rank)")

    cbar = fig.colorbar(im, ax=axes.tolist(), label="Mean Activation (× W_dec norm)",
                        shrink=0.8)
    fig.tight_layout()
    out = OUT_DIR / "sae_heatmap_gpt2.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[4] SAE heatmap saved → {out}")


# ╔══════════════════════════════════════════════════════╗
# ║  5. SAE Features 在 PCA 空间的投影（GPT-2）         ║
# ╚══════════════════════════════════════════════════════╝

def plot_sae_in_pca():
    sae = load_sae_data("gpt2")
    pca = load_pca_projs("gpt2")
    if "top_sae_features_projected" not in sae:
        print("[5] No SAE-PCA projection data — skipping")
        return

    proj = sae["top_sae_features_projected"]      # [L, 100, 6]
    proj_scaled = sae.get("top_sae_features_projected_scaled",
                           None)                   # [L, 100, 6]
    n_layers = pca.shape[0]
    n_omit = 20

    metrics = load_metrics("gpt2")
    best_layer = metrics["r2"].idxmax()

    cmap_features = colormaps["plasma"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("GPT-2 SAE Top-100 Features in PCA Space",
                 fontsize=14, fontweight="bold")

    for idx, (ax, layer, label) in enumerate([
        (axes[0], 0, "Layer 0 (first)"),
        (axes[1], best_layer, f"Layer {best_layer} (best R²)"),
        (axes[2], n_layers - 1, f"Layer {n_layers - 1} (last)"),
    ]):
        # Background: PCA manifold
        data = pca[layer]
        ax.plot(data[n_omit:, 0], data[n_omit:, 1],
                color="gray", alpha=0.3, linewidth=0.8, zorder=0)
        ax.scatter(data[n_omit:, 0], data[n_omit:, 1],
                   color="lightgray", s=12, zorder=1, alpha=0.6)

        # SAE features projected (scaled version if available)
        feat_proj = proj_scaled[layer] if proj_scaled is not None else proj[layer]
        # [100, 6]
        norms = np.linalg.norm(feat_proj[:, :2], axis=1)  # PC1-PC2 magnitude

        sc = ax.scatter(feat_proj[:, 0], feat_proj[:, 1],
                        c=np.arange(100), cmap=cmap_features,
                        s=30, edgecolors="k", linewidth=0.3,
                        zorder=3, alpha=0.85)
        # Arrow from origin
        for i in range(min(10, len(feat_proj))):  # top 10
            ax.arrow(0, 0, feat_proj[i, 0] * 0.85, feat_proj[i, 1] * 0.85,
                     head_width=0.015, head_length=0.02,
                     fc=cmap_features(i / 100), ec=cmap_features(i / 100),
                     alpha=0.7, linewidth=0.5, zorder=4)

        ax.set_title(label, fontsize=12)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_aspect("equal", adjustable="datalim")
        ax.axhline(y=0, color="gray", alpha=0.2, linewidth=0.5)
        ax.axvline(x=0, color="gray", alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    out = OUT_DIR / "sae_in_pca_gpt2.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[5] SAE-in-PCA saved → {out}")


# ╔══════════════════════════════════════════════════════╗
# ║  6. LM Metrics 对比柱状图                            ║
# ╚══════════════════════════════════════════════════════╝

def plot_lm_comparison():
    pythia_lm = load_lm_metrics("pythia-70m")
    gpt2_lm = load_lm_metrics("gpt2")

    metrics_names = ["lm_loss", "lm_acc", "lm_acc_if_newline",
                     "lm_acc_if_newline_any", "mean_prob_if_newline"]
    labels = ["Loss", "Accuracy", "Acc (if \\n)",
              "Acc (any \\n)", "Mean Prob (\\n)"]

    pythia_vals = [pythia_lm[m] for m in metrics_names]
    gpt2_vals = [gpt2_lm[m] for m in metrics_names]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("Language Modeling Baselines: Pythia-70M vs GPT-2",
                 fontsize=14, fontweight="bold")

    x = np.arange(2)
    width = 0.6
    for ax, label, p_val, g_val in zip(axes, labels, pythia_vals, gpt2_vals):
        bars = ax.bar(["Pythia", "GPT-2"], [p_val, g_val],
                      color=["#1A5276", "#B9770E"], width=width, edgecolor="white")
        ax.set_title(label, fontsize=11)
        # Add value on top
        for bar, val in zip(bars, [p_val, g_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}" if isinstance(val, float) else str(val),
                    ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "lm_metrics_comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[6] LM comparison saved → {out}")


# ╔══════════════════════════════════════════════════════╗
# ║  Main                                                ║
# ╚══════════════════════════════════════════════════════╝

def main():
    print("=" * 60)
    print("Counting Manifold Visualization")
    print("=" * 60)

    # Load data
    pythia_metrics = load_metrics("pythia-70m")
    gpt2_metrics = load_metrics("gpt2")

    # 1. PCA manifold (all layers) — both models
    print("\n[1] PCA manifold plots...")
    try:
        plot_pca_manifold("pythia-70m")
    except Exception as e:
        print(f"  Pythia PCA failed: {e}")
    try:
        plot_pca_manifold("gpt2")
    except Exception as e:
        print(f"  GPT-2 PCA failed: {e}")

    # 2. PCA best vs worst
    print("\n[2] PCA best/worst layer plots...")
    try:
        plot_pca_best_worst("pythia-70m")
    except Exception as e:
        print(f"  Pythia best/worst failed: {e}")
    try:
        plot_pca_best_worst("gpt2")
    except Exception as e:
        print(f"  GPT-2 best/worst failed: {e}")

    # 3. Layer metrics comparison
    print("\n[3] Layer metrics comparison...")
    plot_layer_metrics(pythia_metrics, gpt2_metrics)

    # 4. SAE heatmap
    print("\n[4] SAE heatmap...")
    try:
        plot_sae_heatmap()
    except Exception as e:
        print(f"  SAE heatmap failed: {e}")

    # 5. SAE in PCA
    print("\n[5] SAE features in PCA space...")
    try:
        plot_sae_in_pca()
    except Exception as e:
        print(f"  SAE-in-PCA failed: {e}")

    # 6. LM comparison
    print("\n[6] LM metrics comparison...")
    plot_lm_comparison()

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
