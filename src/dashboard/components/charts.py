"""Plotly chart factories for the Streamlit dashboard.

All charts share a common style and color palette so the EDA, Graph, and
Models tabs feel like one coherent product. Every helper takes data in,
returns a ``plotly.graph_objects.Figure`` out — nothing is rendered here,
which keeps the orchestration in ``app.py`` clean.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Shared palette — keep aligned with styles.py CSS variables so the chart
# colors look intentional next to the metric cards / banners.
PRIMARY = "#4f46e5"
ACCENT = "#ec4899"
GOOD = "#10b981"
BAD = "#ef4444"
WARN = "#f59e0b"
MUTED = "#94a3b8"
BG_PANEL = "rgba(255,255,255,0)"

CLASS_COLORS = {0: "#cbd5e1", 1: PRIMARY}
CLASS_LABELS = {0: "Not top-10", 1: "Top-10"}


def _apply_layout(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    """Project-wide chart shell: transparent background, tight margins, minimal grid."""
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=15, color="#0f172a")) if title else None,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
                  color="#0f172a", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", zerolinecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", zerolinecolor="#e2e8f0"),
    )
    return fig


# ─────────────────────────────────── EDA charts ───────────────────────────────────


def class_balance_bar(df: pd.DataFrame, target_col: str = "is_top_10") -> go.Figure:
    """Vertical bar — count of each class with the imbalance ratio called out."""
    counts = df[target_col].value_counts().sort_index()
    labels = [CLASS_LABELS.get(int(i), str(i)) for i in counts.index]
    colors = [CLASS_COLORS.get(int(i), MUTED) for i in counts.index]

    fig = go.Figure(go.Bar(
        x=labels, y=counts.values,
        marker_color=colors,
        text=[f"{v} ({v/counts.sum()*100:.1f}%)" for v in counts.values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Pages: %{y}<extra></extra>",
    ))
    return _apply_layout(fig, height=320, title="Class balance (target imbalance)")


def domain_breakdown_bar(df: pd.DataFrame, target_col: str = "is_top_10") -> go.Figure:
    """Horizontal stacked bar — pages per domain split by class."""
    grouped = (df.groupby(["domain", target_col]).size()
                 .unstack(fill_value=0)
                 .sort_values(by=list(df[target_col].unique()), ascending=True))
    fig = go.Figure()
    for cls in sorted(df[target_col].unique()):
        if cls in grouped.columns:
            fig.add_trace(go.Bar(
                y=grouped.index, x=grouped[cls],
                orientation="h",
                name=CLASS_LABELS.get(int(cls), str(cls)),
                marker_color=CLASS_COLORS.get(int(cls), MUTED),
                hovertemplate="<b>%{y}</b><br>" + CLASS_LABELS.get(int(cls), str(cls))
                              + ": %{x}<extra></extra>",
            ))
    fig.update_layout(barmode="stack")
    return _apply_layout(fig, height=320, title="Pages by domain")


def feature_histogram(df: pd.DataFrame, feature: str, target_col: str = "is_top_10") -> go.Figure:
    """Overlaid histogram of one feature, split by class — surfaces separability."""
    fig = go.Figure()
    for cls in sorted(df[target_col].unique()):
        sub = df.loc[df[target_col] == cls, feature].dropna()
        fig.add_trace(go.Histogram(
            x=sub, name=CLASS_LABELS.get(int(cls), str(cls)),
            marker_color=CLASS_COLORS.get(int(cls), MUTED),
            opacity=0.65, nbinsx=24,
        ))
    fig.update_layout(barmode="overlay", bargap=0.05)
    return _apply_layout(fig, height=300, title=f"Distribution: {feature}")


def feature_box(df: pd.DataFrame, feature: str, target_col: str = "is_top_10") -> go.Figure:
    """Box-plot of one feature by class — outliers + median separation."""
    fig = go.Figure()
    for cls in sorted(df[target_col].unique()):
        sub = df.loc[df[target_col] == cls, feature].dropna()
        fig.add_trace(go.Box(
            y=sub, name=CLASS_LABELS.get(int(cls), str(cls)),
            marker_color=CLASS_COLORS.get(int(cls), MUTED),
            boxmean=True,
        ))
    return _apply_layout(fig, height=300, title=f"Box plot: {feature}")


def correlation_heatmap(df: pd.DataFrame, top_k: int = 15,
                        target_col: str = "is_top_10") -> go.Figure:
    """Heatmap of the top-K features most correlated with the target."""
    numeric = df.select_dtypes(include=[np.number]).copy()
    if target_col not in numeric:
        return _apply_layout(go.Figure(), title="Correlation heatmap (target missing)")
    corr_with_target = numeric.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
    keep = list(corr_with_target.head(top_k).index) + [target_col]
    sub = numeric[keep].corr()

    fig = go.Figure(go.Heatmap(
        z=sub.values, x=sub.columns, y=sub.columns,
        colorscale=[[0.0, "#ef4444"], [0.5, "#ffffff"], [1.0, PRIMARY]],
        zmin=-1, zmax=1, colorbar=dict(title="ρ", thickness=12, len=0.6),
        hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>ρ = %{z:.2f}<extra></extra>",
    ))
    return _apply_layout(fig, height=480,
                         title=f"Correlation heatmap (top-{top_k} vs {target_col})")


def feature_target_scatter(df: pd.DataFrame, x: str, y: str,
                           target_col: str = "is_top_10") -> go.Figure:
    """Scatter of two features colored by class — class separability sanity check."""
    fig = go.Figure()
    for cls in sorted(df[target_col].unique()):
        sub = df[df[target_col] == cls]
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub[y], mode="markers",
            name=CLASS_LABELS.get(int(cls), str(cls)),
            marker=dict(color=CLASS_COLORS.get(int(cls), MUTED), size=9, opacity=0.75,
                        line=dict(width=0.5, color="#0f172a")),
            text=sub.get("url", pd.Series([""] * len(sub))),
            hovertemplate="<b>%{text}</b><br>" + f"{x}: " + "%{x}<br>"
                          + f"{y}: " + "%{y}<extra></extra>",
        ))
    fig.update_xaxes(title=x); fig.update_yaxes(title=y)
    return _apply_layout(fig, height=380, title=f"{y} vs {x} (by class)")


def top_features_correlation_bar(df: pd.DataFrame, top_k: int = 12,
                                 target_col: str = "is_top_10") -> go.Figure:
    """Bar — top-K features ranked by absolute Pearson correlation with target."""
    numeric = df.select_dtypes(include=[np.number]).copy()
    if target_col not in numeric:
        return _apply_layout(go.Figure(), title="Top features (target missing)")
    corr = numeric.corr()[target_col].drop(target_col).sort_values(key=abs, ascending=True).tail(top_k)
    colors = [PRIMARY if v >= 0 else BAD for v in corr.values]
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in corr.values], textposition="outside",
        hovertemplate="<b>%{y}</b><br>ρ = %{x:.3f}<extra></extra>",
    ))
    return _apply_layout(fig, height=420,
                         title=f"Top {top_k} features by |correlation| with {target_col}")


# ─────────────────────────────────── Graph charts ───────────────────────────────────


def pagerank_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of PageRank scores — heavy-tailed shape is the SEO story."""
    fig = go.Figure(go.Histogram(
        x=df["pagerank"], nbinsx=30, marker_color=PRIMARY, opacity=0.85,
    ))
    fig.update_xaxes(title="PageRank score"); fig.update_yaxes(title="Pages")
    return _apply_layout(fig, height=320, title="PageRank distribution")


def hits_hub_authority_scatter(df: pd.DataFrame, target_col: str = "is_top_10") -> go.Figure:
    """Hub vs authority scatter — Kleinberg's two-sided centrality, colored by class."""
    fig = go.Figure()
    for cls in sorted(df[target_col].unique()):
        sub = df[df[target_col] == cls]
        fig.add_trace(go.Scatter(
            x=sub["hits_hub"], y=sub["hits_authority"], mode="markers",
            name=CLASS_LABELS.get(int(cls), str(cls)),
            marker=dict(color=CLASS_COLORS.get(int(cls), MUTED),
                        size=9 + sub["pagerank"].fillna(0) * 800,
                        opacity=0.75, line=dict(width=0.5, color="#0f172a")),
            text=sub.get("url", pd.Series([""] * len(sub))),
            hovertemplate="<b>%{text}</b><br>hub: %{x:.4f}<br>authority: %{y:.4f}<extra></extra>",
        ))
    fig.update_xaxes(title="HITS hub"); fig.update_yaxes(title="HITS authority")
    return _apply_layout(fig, height=420, title="HITS hub vs authority (size ∝ PageRank)")


def degree_scatter(df: pd.DataFrame, target_col: str = "is_top_10") -> go.Figure:
    """In-degree vs out-degree, color by class — link-economy quadrant view."""
    fig = go.Figure()
    for cls in sorted(df[target_col].unique()):
        sub = df[df[target_col] == cls]
        fig.add_trace(go.Scatter(
            x=sub["out_degree"], y=sub["in_degree"], mode="markers",
            name=CLASS_LABELS.get(int(cls), str(cls)),
            marker=dict(color=CLASS_COLORS.get(int(cls), MUTED), size=10, opacity=0.75,
                        line=dict(width=0.5, color="#0f172a")),
            text=sub.get("url", pd.Series([""] * len(sub))),
            hovertemplate="<b>%{text}</b><br>out: %{x}<br>in: %{y}<extra></extra>",
        ))
    fig.update_xaxes(title="Out-degree"); fig.update_yaxes(title="In-degree")
    return _apply_layout(fig, height=380, title="In-degree vs out-degree")


def build_url_hierarchy_graph(df: pd.DataFrame, max_nodes: int = 80) -> nx.DiGraph:
    """Construct a directed URL-hierarchy graph from the feature table.

    Each URL becomes a node; edges go from a path to its parent path. This is
    a deterministic structural view of the corpus — it does not require the
    raw HTML link graph to be present, which makes the dashboard work in any
    clone of the repo.
    """
    from urllib.parse import urlparse

    g: nx.DiGraph = nx.DiGraph()
    df_top = df.copy()
    if "pagerank" in df_top.columns:
        df_top = df_top.sort_values("pagerank", ascending=False).head(max_nodes)
    else:
        df_top = df_top.head(max_nodes)

    nodes_added: dict[str, dict[str, Any]] = {}
    for _, row in df_top.iterrows():
        url = str(row["url"])
        p = urlparse(url)
        parts = [s for s in p.path.strip("/").split("/") if s]
        # Walk path: domain → domain/a → domain/a/b → …
        prev = p.netloc.lower()
        nodes_added.setdefault(prev, {"is_root": True, "is_top_10": False, "pagerank": 0.0,
                                      "label": p.netloc.lower()})
        for i, seg in enumerate(parts):
            curr = f"{prev}/{seg}"
            is_leaf = (i == len(parts) - 1)
            existing = nodes_added.get(curr, {"is_root": False, "is_top_10": False, "pagerank": 0.0,
                                              "label": seg})
            if is_leaf:
                existing["is_top_10"] = bool(row.get("is_top_10", 0))
                existing["pagerank"] = float(row.get("pagerank", 0.0))
                existing["url"] = url
            nodes_added[curr] = existing
            g.add_edge(prev, curr)
            prev = curr

    for n, attrs in nodes_added.items():
        if n in g:
            g.nodes[n].update(attrs)
    return g


def url_hierarchy_network(df: pd.DataFrame, max_nodes: int = 80) -> go.Figure:
    """Plotly network visualization of the URL hierarchy graph.

    Edges: thin gray; nodes: colored by `is_top_10` and sized by PageRank.
    Layout: spring layout (force-directed). For corpora of 50-100 pages this
    fits a single figure cleanly.
    """
    g = build_url_hierarchy_graph(df, max_nodes=max_nodes)
    if g.number_of_nodes() == 0:
        return _apply_layout(go.Figure(), title="URL hierarchy graph (empty)")

    pos = nx.spring_layout(g, seed=42, k=1.6 / max(1, np.sqrt(g.number_of_nodes())))

    edge_x: list[float] = []
    edge_y: list[float] = []
    for u, v in g.edges():
        edge_x.extend([pos[u][0], pos[v][0], None])
        edge_y.extend([pos[u][1], pos[v][1], None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines", line=dict(width=0.6, color="#cbd5e1"),
        hoverinfo="none", showlegend=False,
    )

    node_x, node_y, colors, sizes, texts, lines = [], [], [], [], [], []
    for n, data in g.nodes(data=True):
        node_x.append(pos[n][0]); node_y.append(pos[n][1])
        is_root = data.get("is_root", False)
        is_top = data.get("is_top_10", False)
        pr = data.get("pagerank", 0.0)
        if is_root:
            colors.append(ACCENT); sizes.append(18); lines.append(2.0)
        elif is_top:
            colors.append(PRIMARY); sizes.append(8 + pr * 1200); lines.append(1.0)
        else:
            colors.append("#cbd5e1"); sizes.append(7 + pr * 800); lines.append(0.5)
        url = data.get("url", n)
        texts.append(f"<b>{data.get('label', n)}</b><br>{url}<br>"
                     f"PageRank: {pr:.4f}<br>"
                     f"Top-10: {'Yes' if is_top else 'No'}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers", hoverinfo="text",
        text=texts,
        marker=dict(color=colors, size=sizes,
                    line=dict(width=lines, color="#0f172a")),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return _apply_layout(fig, height=540, title=f"URL hierarchy graph ({g.number_of_nodes()} nodes)")


# ─────────────────────────────────── Modeling charts ───────────────────────────────────


def metrics_comparison_bar(metrics_by_model: dict[str, dict[str, float]]) -> go.Figure:
    """Grouped bar: F1 / ROC-AUC / PR-AUC per model — the headline comparison."""
    keys = ["f1", "roc_auc", "pr_auc"]
    pretty = {"f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
    fig = go.Figure()
    palette = [PRIMARY, ACCENT, GOOD, WARN]
    for i, (model, m) in enumerate(metrics_by_model.items()):
        fig.add_trace(go.Bar(
            name=model, x=[pretty[k] for k in keys],
            y=[m.get(k, 0.0) for k in keys],
            marker_color=palette[i % len(palette)],
            text=[f"{m.get(k, 0.0):.3f}" for k in keys], textposition="outside",
            hovertemplate=f"<b>{model}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.0], title="Score")
    return _apply_layout(fig, height=380, title="Model comparison (held-out test set)")


def confusion_matrix_heatmap(cm: list[list[int]], model_name: str) -> go.Figure:
    """Heatmap of one model's confusion matrix with cell counts annotated."""
    z = np.asarray(cm)
    labels = ["Not top-10", "Top-10"]
    annot = [[str(z[i][j]) for j in range(z.shape[1])] for i in range(z.shape[0])]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=[[0.0, "#ffffff"], [1.0, PRIMARY]],
        showscale=False,
        text=annot, texttemplate="%{text}", textfont=dict(size=18, color="#0f172a"),
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return _apply_layout(fig, height=300, title=f"{model_name} confusion matrix")


def feature_importance_bar(importances: dict[str, float], top_k: int = 15,
                           model_name: str = "model") -> go.Figure:
    """Horizontal bar of top-K feature importances (signed)."""
    items = sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    items.reverse()  # so largest is at top of horizontal bar
    names = [k for k, _ in items]; vals = [v for _, v in items]
    colors = [PRIMARY if v >= 0 else BAD for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    return _apply_layout(fig, height=460, title=f"{model_name} — top {top_k} features")


def roc_curve_chart(curves: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> go.Figure:
    """ROC curves for several models on the same axes (FPR / TPR / AUC in legend)."""
    fig = go.Figure()
    palette = [PRIMARY, ACCENT, GOOD, WARN]
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            line=dict(color=palette[i % len(palette)], width=2.5),
            name=f"{name} (AUC = {auc:.3f})",
            hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=MUTED, width=1, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1])
    return _apply_layout(fig, height=360, title="ROC curves")


def pr_curve_chart(curves: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> go.Figure:
    """Precision-recall curves — better signal than ROC under class imbalance."""
    fig = go.Figure()
    palette = [PRIMARY, ACCENT, GOOD, WARN]
    for i, (name, (precision, recall, ap)) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=recall, y=precision, mode="lines",
            line=dict(color=palette[i % len(palette)], width=2.5),
            name=f"{name} (AP = {ap:.3f})",
            hovertemplate=f"<b>{name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1])
    return _apply_layout(fig, height=360, title="Precision-recall curves")
