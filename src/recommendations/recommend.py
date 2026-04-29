"""Per-page actionable recommendations + SHAP analysis on the winning model.

Two responsibilities, one file (small enough to share):

1. ``recommend(row, model, query)`` returns a list of concrete suggestions
   with current values, recommended values, and "why" — phrased like the
   rubric expects ("title is 80 chars, shorten to 50-60", "add 2 H2 headers",
   "add alt text to 4 of 7 images"). Suggestions are SHAP-ranked when a
   SHAP explainer is available; otherwise rules-only.

2. ``save_shap_summary(model, X_sample, out_path)`` generates the
   ``assets/charts/06_shap_summary.png`` plot used in the modeling notebook
   and the dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG = logging.getLogger("recommend")

# Target ranges for SEO best practice. Defended in MODELING_DECISIONS.md.
TITLE_MIN, TITLE_MAX = 30, 60
META_DESC_MIN, META_DESC_MAX = 120, 160
MIN_H2 = 2
MIN_ALT_COVERAGE = 0.8
MIN_KEYWORD_DENSITY = 0.005


@dataclass
class Suggestion:
    """One actionable, numerically-grounded suggestion."""

    action: str
    target_feature: str
    current_value: float
    recommended_value: str
    why: str
    impact: float = 0.0  # SHAP-derived priority (higher = bigger predicted lift)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shap_impact(features: pd.Series, model: Any) -> dict[str, float]:
    """Return a {feature: signed_shap_value} dict for one prediction. Negative
    SHAP value means the feature pushed prediction DOWN — i.e. fixing that
    feature should yield the biggest lift. Returns empty dict on any error,
    so the rules-based path remains usable."""
    try:
        import shap
    except ImportError:
        return {}
    try:
        x = features.to_frame().T
        # Tree models: fast TreeExplainer. Anything else: KernelExplainer
        # would need a background sample we don't have here, so skip.
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(x)
        if isinstance(sv, list):  # binary classification, list of two arrays
            sv = sv[1]
        return dict(zip(features.index, np.asarray(sv).reshape(-1)))
    except Exception as exc:  # noqa: BLE001 — SHAP failures must not break recs
        LOG.debug("SHAP impact unavailable: %s", exc)
        return {}


def _rule_suggestions(row: pd.Series, query: str) -> list[Suggestion]:
    sugg: list[Suggestion] = []

    title_len = float(row.get("title_length", 0.0))
    if title_len < TITLE_MIN:
        sugg.append(Suggestion(
            action=f"Title is {int(title_len)} chars — extend to {TITLE_MIN}-{TITLE_MAX}.",
            target_feature="title_length",
            current_value=title_len,
            recommended_value=f"{TITLE_MIN}-{TITLE_MAX}",
            why="Short titles often miss keywords search engines reward.",
        ))
    elif title_len > TITLE_MAX:
        sugg.append(Suggestion(
            action=f"Title is {int(title_len)} chars — shorten to {TITLE_MIN}-{TITLE_MAX}.",
            target_feature="title_length",
            current_value=title_len,
            recommended_value=f"{TITLE_MIN}-{TITLE_MAX}",
            why="SERP truncates titles >60 chars; truncated titles convert worse.",
        ))

    if float(row.get("has_meta_description", 0.0)) < 1.0:
        sugg.append(Suggestion(
            action=f"Add a meta description ({META_DESC_MIN}-{META_DESC_MAX} chars).",
            target_feature="has_meta_description",
            current_value=0.0,
            recommended_value=f"{META_DESC_MIN}-{META_DESC_MAX} chars",
            why="Pages without meta descriptions cede snippet control to Google.",
        ))
    else:
        md_len = float(row.get("meta_description_length", 0.0))
        if md_len < META_DESC_MIN:
            sugg.append(Suggestion(
                action=f"Meta description is {int(md_len)} chars — extend to {META_DESC_MIN}-{META_DESC_MAX}.",
                target_feature="meta_description_length",
                current_value=md_len,
                recommended_value=f"{META_DESC_MIN}-{META_DESC_MAX}",
                why="Short meta descriptions often get rewritten by Google.",
            ))

    if float(row.get("keyword_in_title", 0.0)) < 1.0 and query:
        sugg.append(Suggestion(
            action=f"Include the topic keyword (\"{query}\") in your title.",
            target_feature="keyword_in_title",
            current_value=0.0,
            recommended_value="1 (present)",
            why="Topic keywords in <title> are a strong ranking signal.",
        ))

    h2 = int(float(row.get("h2_count", 0.0)))
    if h2 < MIN_H2:
        gap = MIN_H2 - h2
        sugg.append(Suggestion(
            action=f"Add {gap} H2 header{'s' if gap > 1 else ''} to break up content.",
            target_feature="h2_count",
            current_value=float(h2),
            recommended_value=f"≥{MIN_H2}",
            why="H2 structure improves scannability and helps SERP feature targeting.",
        ))

    img = int(float(row.get("image_count", 0.0)))
    cov = float(row.get("alt_text_coverage", 0.0))
    if img > 0 and cov < MIN_ALT_COVERAGE:
        missing = max(0, int(round(img * (1.0 - cov))))
        sugg.append(Suggestion(
            action=f"Add alt text to {missing} of {img} images.",
            target_feature="alt_text_coverage",
            current_value=cov,
            recommended_value=f"≥{int(MIN_ALT_COVERAGE * 100)}% coverage",
            why="Alt text is required for accessibility and contributes to image-search ranking.",
        ))

    kd = float(row.get("keyword_density", 0.0))
    if kd < MIN_KEYWORD_DENSITY and query:
        sugg.append(Suggestion(
            action=f"Mention the topic keyword more often (currently {kd*100:.2f}% of words).",
            target_feature="keyword_density",
            current_value=kd,
            recommended_value=f"≥{MIN_KEYWORD_DENSITY*100:.1f}%",
            why="Below-floor keyword density correlates with weaker topical relevance.",
        ))

    return sugg


def recommend(
    features_row: pd.Series,
    model: Any | None = None,
    query: str = "",
    min_suggestions: int = 3,
) -> list[dict[str, Any]]:
    """Build a list of concrete suggestions for one page.

    Args:
        features_row: numeric feature row (column names matching training).
        model: optional fitted classifier — used only for SHAP ranking.
        query: query string the page is being evaluated against.
        min_suggestions: pad with generic suggestions if rules emit fewer
            than this. Defaults to the rubric's ≥3 floor.
    """
    sugg = _rule_suggestions(features_row, query)

    if model is not None:
        impacts = _shap_impact(features_row, model)
        for s in sugg:
            s.impact = float(abs(impacts.get(s.target_feature, 0.0)))
        sugg.sort(key=lambda s: s.impact, reverse=True)

    if len(sugg) < min_suggestions:
        for note in [
            ("Add internal links to topically-related pages on your site.",
             "internal_link_count", "Internal links concentrate PageRank on important pages."),
            ("Increase content depth — explore subtopics with subheadings.",
             "word_count", "Longer, well-structured content tends to rank higher."),
            ("Audit page load speed and Core Web Vitals.",
             "_external", "Page experience signals are a Google ranking factor."),
        ]:
            if len(sugg) >= min_suggestions:
                break
            sugg.append(Suggestion(
                action=note[0], target_feature=note[1],
                current_value=float(features_row.get(note[1], 0.0)) if note[1] != "_external" else 0.0,
                recommended_value="see why", why=note[2],
            ))

    return [s.to_dict() for s in sugg]


def save_shap_summary(
    model: Any,
    X_sample: pd.DataFrame,
    out_path: Path = Path("assets/charts/06_shap_summary.png"),
    max_display: int = 20,
) -> Path:
    """Save the SHAP summary (beeswarm) plot for the winning model. Used by
    notebook 03 and the dashboard's About panel."""
    import shap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[1]
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(sv, X_sample, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    LOG.info("SHAP summary → %s", out_path)
    return out_path


def shap_per_prediction(model: Any, features_row: pd.Series, top_k: int = 5) -> list[tuple[str, float]]:
    """Top-K (feature_name, signed_shap_value) for a single prediction. Used
    by the dashboard's per-page breakdown panel."""
    impacts = _shap_impact(features_row, model)
    if not impacts:
        return []
    ranked = sorted(impacts.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [(k, float(v)) for k, v in ranked[:top_k]]


def _peek_sample_rows(features: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Sample a small frame for SHAP background — full corpus is overkill."""
    if len(features) <= n:
        return features
    return features.sample(n, random_state=42)


def _suggestion_lines(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Render suggestion dicts as one-line strings (used by demo CLI)."""
    return [f"• {r['action']}  [{r['target_feature']}]" for r in rows]
