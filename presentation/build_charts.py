"""Generate static chart PNGs for the slide deck.

Plotly figures don't embed cleanly in PowerPoint, so we render the same
analysis with matplotlib at 300 DPI and write PNGs into
``presentation/charts/``. Style is shared with ``src/dashboard/components/charts.py``
so the slides feel like a continuation of the dashboard.

Run:
    python -m presentation.build_charts
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches

PRIMARY = "#4f46e5"
PRIMARY_DARK = "#3730a3"
ACCENT = "#ec4899"
GOOD = "#10b981"
BAD = "#ef4444"
WARN = "#f59e0b"
MUTED = "#94a3b8"
TEXT = "#0f172a"
LIGHT = "#f8fafc"
GRID = "#e2e8f0"

DPI = 300
FIG_W, FIG_H = 9.0, 5.0  # 16:9-friendly aspect

CHART_DIR = Path("presentation/charts")
FEATURES_CSV = Path("data/processed/features.csv")
METRICS_DIR = Path("models/metrics")


def _styled_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=11)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(TEXT)
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("700")
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def _save(fig: plt.Figure, name: str) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    out = CHART_DIR / f"{name}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out} ({out.stat().st_size // 1024} KB)")


def chart_class_balance(df: pd.DataFrame) -> None:
    counts = df["is_top_10"].value_counts().sort_index()
    labels = ["Not top-10", "Top-10"]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bars = ax.bar(labels, counts.values, color=[MUTED, PRIMARY], width=0.55,
                  edgecolor="white", linewidth=2)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + counts.max() * 0.02,
                f"{v:,}\n({v / counts.sum() * 100:.1f}%)",
                ha="center", va="bottom", fontsize=12, color=TEXT, fontweight="600")
    ax.set_title("Class balance after oversampling", pad=20)
    ax.set_ylabel("Pages")
    ax.set_ylim(0, counts.max() * 1.18)
    _styled_axes(ax)
    _save(fig, "01_class_balance")


def chart_domain_breakdown(df: pd.DataFrame) -> None:
    if "domain" not in df.columns:
        return
    grouped = df.groupby(["domain", "is_top_10"]).size().unstack(fill_value=0)
    grouped["total"] = grouped.sum(axis=1)
    grouped = grouped.sort_values("total", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    y = np.arange(len(grouped))
    ax.barh(y, grouped[0], color=MUTED, label="Not top-10", height=0.6,
            edgecolor="white", linewidth=1)
    ax.barh(y, grouped[1] if 1 in grouped.columns else 0, left=grouped[0],
            color=PRIMARY, label="Top-10", height=0.6,
            edgecolor="white", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(grouped.index)
    ax.set_title("Pages by domain (top 10)", pad=20)
    ax.set_xlabel("Pages")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    _styled_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    _save(fig, "02_domain_breakdown")


def chart_top_correlations(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include=[np.number])
    if "is_top_10" not in numeric:
        return
    corr = (numeric.corr()["is_top_10"]
            .drop("is_top_10")
            .sort_values(key=abs, ascending=True)
            .tail(12))
    colors = [PRIMARY if v >= 0 else BAD for v in corr.values]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bars = ax.barh(corr.index, corr.values, color=colors, height=0.65,
                   edgecolor="white", linewidth=1)
    for b, v in zip(bars, corr.values):
        offset = corr.abs().max() * 0.015
        ax.text(b.get_width() + (offset if v >= 0 else -offset),
                b.get_y() + b.get_height() / 2,
                f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right",
                fontsize=10, color=TEXT, fontweight="600")
    ax.axvline(0, color=GRID, linewidth=1)
    ax.set_title("Top features by correlation with `is_top_10`", pad=20)
    ax.set_xlabel("Pearson correlation")
    _styled_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    _save(fig, "03_top_correlations")


def chart_distribution_by_class(df: pd.DataFrame, feature: str, name: str) -> None:
    if feature not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bins = np.linspace(df[feature].min(), df[feature].quantile(0.98), 30)
    for cls, color, label in [(0, MUTED, "Not top-10"), (1, PRIMARY, "Top-10")]:
        sub = df.loc[df["is_top_10"] == cls, feature].dropna()
        ax.hist(sub, bins=bins, color=color, alpha=0.65, label=label,
                edgecolor="white", linewidth=0.6)
    ax.set_title(f"Distribution of {feature} by class", pad=20)
    ax.set_xlabel(feature); ax.set_ylabel("Pages")
    ax.legend(frameon=False, fontsize=10)
    _styled_axes(ax)
    _save(fig, name)


def chart_pagerank(df: pd.DataFrame) -> None:
    if "pagerank" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    df_unique = df.drop_duplicates(subset=["url"]) if "url" in df.columns else df
    ax.hist(df_unique["pagerank"], bins=40, color=PRIMARY, alpha=0.85,
            edgecolor="white", linewidth=0.6)
    ax.set_title(f"PageRank distribution ({len(df_unique):,} unique pages)", pad=20)
    ax.set_xlabel("PageRank score (α = 0.85)"); ax.set_ylabel("Pages")
    _styled_axes(ax)
    _save(fig, "06_pagerank_hist")


def chart_hits(df: pd.DataFrame) -> None:
    if not {"hits_hub", "hits_authority", "is_top_10", "pagerank"}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    df_unique = df.drop_duplicates(subset=["url"]) if "url" in df.columns else df
    for cls, color, label in [(0, MUTED, "Not top-10"), (1, PRIMARY, "Top-10")]:
        sub = df_unique[df_unique["is_top_10"] == cls]
        sizes = 20 + sub["pagerank"] * 8000
        ax.scatter(sub["hits_hub"], sub["hits_authority"],
                   s=sizes, c=color, alpha=0.6, edgecolors="white",
                   linewidths=0.7, label=label)
    ax.set_title("HITS hub vs authority (size ∝ PageRank)", pad=20)
    ax.set_xlabel("HITS hub"); ax.set_ylabel("HITS authority")
    ax.legend(frameon=False, fontsize=10)
    _styled_axes(ax)
    _save(fig, "07_hits_scatter")


def chart_metrics_comparison() -> None:
    metrics = {}
    for path in sorted(METRICS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            metrics[data.get("model", path.stem)] = data
        except Exception:
            continue
    if not metrics:
        return
    keys = ["f1", "roc_auc", "pr_auc"]; pretty = ["F1", "ROC-AUC", "PR-AUC"]
    models = list(metrics.keys())
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(pretty)); w = 0.8 / len(models)
    palette = [PRIMARY, ACCENT, GOOD, WARN]
    for i, m in enumerate(models):
        vals = [metrics[m].get(k, 0.0) for k in keys]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w,
                      color=palette[i % len(palette)], label=m,
                      edgecolor="white", linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color=TEXT, fontweight="600")
    ax.set_xticks(x); ax.set_xticklabels(pretty)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model comparison · held-out test set", pad=20)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    _styled_axes(ax)
    _save(fig, "08_model_comparison")


def chart_confusion(name: str, cm: list[list[int]]) -> None:
    z = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(z, cmap="Blues", vmin=0, vmax=z.max() * 1.05)
    labels = ["Not top-10", "Top-10"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion · {name}", pad=20, fontsize=15, fontweight="700",
                 color=TEXT)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            color = "white" if z[i, j] > z.max() / 2 else TEXT
            ax.text(j, i, str(z[i, j]), ha="center", va="center",
                    fontsize=22, fontweight="700", color=color)
    ax.tick_params(colors=TEXT, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    _save(fig, f"09_confusion_{name}")


def main() -> None:
    if not FEATURES_CSV.exists():
        print(f"missing {FEATURES_CSV}")
        return
    df = pd.read_csv(FEATURES_CSV)
    print(f"source: {FEATURES_CSV} ({len(df):,} rows)")

    chart_class_balance(df)
    chart_domain_breakdown(df)
    chart_top_correlations(df)
    chart_distribution_by_class(df, "title_length", "04_title_length")
    chart_distribution_by_class(df, "h2_count", "05_h2_count")
    chart_pagerank(df)
    chart_hits(df)
    chart_metrics_comparison()

    # Confusion matrix per saved metrics JSON.
    for path in sorted(METRICS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            cm = data.get("confusion_matrix")
            if cm:
                chart_confusion(data.get("model", path.stem), cm)
        except Exception:
            continue

    print("done.")


if __name__ == "__main__":
    main()
