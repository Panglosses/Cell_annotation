#!/usr/bin/env python
"""Create Nature-style report figures from existing benchmark outputs.

No model outputs are recalculated here. The script reads existing metrics,
classification-report and bootstrap CSV files and redraws the report figures
with a unified publication-style visual language.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
OUT = FIGURES / "nature_style"
OUT.mkdir(parents=True, exist_ok=True)


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.0
plt.rcParams["axes.linewidth"] = 0.75
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False


PALETTE = {
    "pca": "#6E7481",
    "scgpt": "#BD5E73",
    "geneformer": "#C68A48",
    "uce": "#5E9D67",
    "neutral": "#767676",
    "grid": "#D8D8D8",
    "text": "#222222",
    "soft_bg": "#F4F1EC",
}


MODEL_ORDER = [
    "PCA + Logistic Regression",
    "PCA + kNN",
    "scGPT + Logistic Regression",
    "scGPT + MLP",
    "Geneformer V1-10M + Logistic Regression",
    "Geneformer V1-10M + MLP",
    "UCE + Logistic Regression",
    "UCE + MLP",
]


DISPLAY_NAMES = {
    "PCA + Logistic Regression": "PCA LR",
    "PCA + kNN": "PCA kNN",
    "scGPT + Logistic Regression": "scGPT LR",
    "scGPT + MLP": "scGPT MLP",
    "Geneformer V1-10M + Logistic Regression": "Geneformer LR",
    "Geneformer V1-10M + MLP": "Geneformer MLP",
    "UCE + Logistic Regression": "UCE LR",
    "UCE + MLP": "UCE MLP",
}


def model_color(name: str) -> str:
    if name.startswith("PCA"):
        return PALETTE["pca"]
    if name.startswith("scGPT"):
        return PALETTE["scgpt"]
    if name.startswith("Geneformer"):
        return PALETTE["geneformer"]
    if name.startswith("UCE"):
        return PALETTE["uce"]
    return PALETTE["neutral"]


def save_all(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(stem.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)


def make_model_label(row: pd.Series) -> str:
    if row["feature_type"].startswith("PCA"):
        return f"PCA + {row['classifier']}"
    return f"{row['model']} + {row['classifier']}"


def draw_main_figure() -> None:
    metrics = pd.read_csv(METRICS / "consolidated_main_metrics.csv")
    metrics = metrics[
        (metrics["label_level"] == "celltype_l4")
        & ~metrics["model"].str.contains("cross-tissue", case=False, na=False)
    ].copy()
    metrics["method"] = metrics.apply(make_model_label, axis=1)
    metrics = metrics[metrics["method"].isin(MODEL_ORDER)].copy()
    metrics["display"] = metrics["method"].map(DISPLAY_NAMES)
    metrics["order"] = metrics["method"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    metrics = metrics.sort_values("weighted_f1", ascending=True)

    paired = pd.read_csv(TABLES / "bootstrap_paired_weighted_f1_differences.csv")
    keep = {
        "scGPT + MLP minus UCE + MLP": "scGPT MLP - UCE MLP",
        "scGPT + MLP minus scGPT + Logistic Regression": "scGPT MLP - scGPT LR",
        "UCE + MLP minus UCE + Logistic Regression": "UCE MLP - UCE LR",
        "Geneformer V1-10M + MLP minus Geneformer V1-10M + Logistic Regression": (
            "Geneformer MLP - LR"
        ),
    }
    paired = paired[paired["comparison"].isin(keep)].copy()
    paired["display"] = paired["comparison"].map(keep)
    paired["color"] = [
        PALETTE["neutral"],
        PALETTE["scgpt"],
        PALETTE["uce"],
        PALETTE["geneformer"],
    ]
    paired["y"] = np.arange(len(paired))[::-1]

    fig = plt.figure(figsize=(7.2, 3.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.42)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    y = np.arange(len(metrics))
    colors = [model_color(m) for m in metrics["method"]]
    ax1.barh(
        y,
        metrics["weighted_f1"],
        color=colors,
        edgecolor="#222222",
        linewidth=0.55,
        height=0.68,
    )
    for yi, (_, row) in zip(y, metrics.iterrows()):
        ax1.text(
            row["weighted_f1"] + 0.003,
            yi,
            f"{row['weighted_f1']:.3f}",
            va="center",
            ha="left",
            fontsize=6.6,
            color=PALETTE["text"],
        )
    ax1.set_yticks(y)
    ax1.set_yticklabels(metrics["display"], fontsize=7)
    for tick in ax1.get_yticklabels():
        if tick.get_text() == "PCA LR":
            tick.set_fontweight("bold")
    ax1.set_xlim(0.66, 0.925)
    ax1.set_xlabel("Weighted F1")
    ax1.set_title("Fine-grained annotation", fontsize=8.5, fontweight="bold", pad=5)
    ax1.grid(axis="x", color=PALETTE["grid"], lw=0.45, ls="--", alpha=0.65)
    ax1.tick_params(axis="both", width=0.75, length=2.5)
    ax1.text(-0.14, 1.045, "a", transform=ax1.transAxes, fontsize=10, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", label="PCA", markerfacecolor=PALETTE["pca"], markeredgecolor="#222222", markersize=4.5),
        Line2D([0], [0], marker="o", color="none", label="scGPT", markerfacecolor=PALETTE["scgpt"], markeredgecolor="#222222", markersize=4.5),
        Line2D([0], [0], marker="o", color="none", label="Geneformer", markerfacecolor=PALETTE["geneformer"], markeredgecolor="#222222", markersize=4.5),
        Line2D([0], [0], marker="o", color="none", label="UCE", markerfacecolor=PALETTE["uce"], markeredgecolor="#222222", markersize=4.5),
    ]
    ax1.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=5.8,
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.7,
        borderaxespad=0.2,
    )

    for _, row in paired.iterrows():
        ax2.plot(
            [row["ci95_low"], row["ci95_high"]],
            [row["y"], row["y"]],
            color="#222222",
            lw=0.8,
            solid_capstyle="round",
        )
        ax2.scatter(
            row["point_difference"],
            row["y"],
            s=28,
            color=row["color"],
            edgecolor="#222222",
            linewidth=0.55,
            zorder=3,
        )
    ax2.axvline(0, color=PALETTE["neutral"], lw=0.75, ls="--")
    ax2.set_yticks(paired["y"])
    ax2.set_yticklabels(paired["display"], fontsize=6.5)
    ax2.set_xlabel("Δ weighted F1")
    ax2.set_title("Paired bootstrap contrasts", fontsize=8.5, fontweight="bold", pad=5)
    ax2.set_xlim(-0.03, 0.105)
    ax2.set_ylim(-0.65, len(paired) - 0.35)
    ax2.grid(axis="x", color=PALETTE["grid"], lw=0.45, ls="--", alpha=0.65)
    ax2.tick_params(axis="both", width=0.75, length=2.5)
    ax2.text(-0.26, 1.045, "b", transform=ax2.transAxes, fontsize=10, fontweight="bold")
    ax2.text(
        0.98,
        0.04,
        "5000 resamples",
        transform=ax2.transAxes,
        fontsize=5.9,
        color=PALETTE["neutral"],
        ha="right",
        va="bottom",
    )

    save_all(fig, OUT / "figure1_nature_weighted_f1_bootstrap")


def load_f1(report_path: Path) -> pd.Series:
    df = pd.read_csv(report_path)
    label_col = "label" if "label" in df.columns else df.columns[0]
    f1_col = "f1-score" if "f1-score" in df.columns else "f1_score"
    df = df[~df[label_col].astype(str).isin(["accuracy", "macro avg", "weighted avg"])]
    return df.set_index(label_col)[f1_col].astype(float)


def draw_supplementary_heatmap() -> None:
    sources = {
        "PCA LR": METRICS / "logreg_classification_report.csv",
        "scGPT MLP": TABLES / "scgpt_mlp_probe_classification_report.csv",
        "Geneformer MLP": TABLES / "geneformer_mlp_probe_classification_report.csv",
        "UCE MLP": TABLES / "uce_mlp_probe_classification_report.csv",
    }
    f1 = pd.DataFrame({name: load_f1(path) for name, path in sources.items()}).T
    labels = [
        "C1",
        "C2",
        "C3",
        "CD56bright NK",
        "CD56dim NK",
        "Cycling NK",
        "ILC2",
        "ILC3",
        "Naïve-like ILC",
    ]
    f1 = f1[labels]

    cmap = LinearSegmentedColormap.from_list(
        "soft_benchmark",
        ["#F7F4EF", "#D9E5C8", "#8EBC8A", "#3F8046"],
    )
    fig, ax = plt.subplots(figsize=(7.0, 2.45))
    im = ax.imshow(f1.values, aspect="auto", vmin=0.45, vmax=1.0, cmap=cmap)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.7)
    ax.set_yticks(np.arange(len(f1.index)))
    ax.set_yticklabels(f1.index, fontsize=7.2)
    ax.tick_params(length=0)
    ax.set_title("Per-class F1 across selected celltype_l4 models", fontsize=8.5, fontweight="bold", pad=7)

    for i in range(f1.shape[0]):
        for j in range(f1.shape[1]):
            val = f1.iat[i, j]
            txt_color = "white" if val >= 0.86 else "#222222"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.9, color=txt_color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, f1.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, f1.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.018)
    cbar.set_label("F1", fontsize=7)
    cbar.ax.tick_params(labelsize=6.2, width=0.5, length=2)
    ax.text(-0.08, 1.08, "a", transform=ax.transAxes, fontsize=12, fontweight="bold")
    save_all(fig, OUT / "supplementary_figure1_nature_per_class_f1")


def parse_percent(value: str) -> float:
    match = re.search(r"\(([-0-9.]+)%\)", str(value))
    if not match:
        return np.nan
    return float(match.group(1))


def draw_supplementary_confusion_summary() -> None:
    df = pd.read_csv(TABLES / "supplementary_table2_confusion_f1_summary.csv")
    rate_cols = [
        "C1_to_CD56bright_NK",
        "C1_C2_bidirectional",
        "C2_C3_bidirectional",
        "Naive_like_ILC_to_C1",
    ]
    rate = df[rate_cols].map(parse_percent)
    rate.index = df["model"]
    rate.columns = [
        "C1→CD56bright NK",
        "C1↔C2",
        "C2↔C3",
        "Naive-like ILC→C1",
    ]
    f1 = df.set_index("model")[["CD56dim_NK_F1", "ILC2_F1"]]
    f1.columns = ["CD56dim NK F1", "ILC2 F1"]

    fig = plt.figure(figsize=(7.0, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.75], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    cmap_err = LinearSegmentedColormap.from_list("err", ["#F7F4EF", "#E7C07A", "#B95D4C"])
    im1 = ax1.imshow(rate.values, aspect="auto", vmin=0, vmax=35, cmap=cmap_err)
    ax1.set_xticks(np.arange(rate.shape[1]))
    ax1.set_xticklabels(rate.columns, rotation=28, ha="right", fontsize=6.5)
    ax1.set_yticks(np.arange(rate.shape[0]))
    ax1.set_yticklabels(rate.index, fontsize=7)
    ax1.tick_params(length=0)
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            val = rate.iat[i, j]
            ax1.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=6, color="#222222")
    ax1.set_title("Selected confusion rates", fontsize=8.2, fontweight="bold", pad=6)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_xticks(np.arange(-0.5, rate.shape[1], 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, rate.shape[0], 1), minor=True)
    ax1.grid(which="minor", color="white", linewidth=1.0)
    ax1.tick_params(which="minor", bottom=False, left=False)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.035, pad=0.02)
    cbar1.set_label("%", fontsize=7)
    cbar1.ax.tick_params(labelsize=6.2, width=0.5, length=2)
    ax1.text(-0.10, 1.08, "a", transform=ax1.transAxes, fontsize=12, fontweight="bold")

    cmap_f1 = LinearSegmentedColormap.from_list("f1", ["#F7F4EF", "#D9E5C8", "#5E9D67"])
    im2 = ax2.imshow(f1.values, aspect="auto", vmin=0.80, vmax=1.00, cmap=cmap_f1)
    ax2.set_xticks(np.arange(f1.shape[1]))
    ax2.set_xticklabels(f1.columns, rotation=28, ha="right", fontsize=6.5)
    ax2.set_yticks(np.arange(f1.shape[0]))
    ax2.set_yticklabels(f1.index, fontsize=7)
    ax2.tick_params(length=0)
    for i in range(f1.shape[0]):
        for j in range(f1.shape[1]):
            ax2.text(j, i, f"{f1.iat[i, j]:.3f}", ha="center", va="center", fontsize=6, color="#222222")
    ax2.set_title("Selected strong classes", fontsize=8.2, fontweight="bold", pad=6)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_xticks(np.arange(-0.5, f1.shape[1], 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, f1.shape[0], 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=1.0)
    ax2.tick_params(which="minor", bottom=False, left=False)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.055, pad=0.03)
    cbar2.set_label("F1", fontsize=7)
    cbar2.ax.tick_params(labelsize=6.2, width=0.5, length=2)
    ax2.text(-0.19, 1.08, "b", transform=ax2.transAxes, fontsize=12, fontweight="bold")

    save_all(fig, OUT / "supplementary_figure2_nature_confusion_f1")


if __name__ == "__main__":
    draw_main_figure()
    draw_supplementary_heatmap()
    draw_supplementary_confusion_summary()
