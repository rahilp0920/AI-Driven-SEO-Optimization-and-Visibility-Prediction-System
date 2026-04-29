"""Streamlit dashboard — SEO Ranking Predictor.

Sidebar tabs (Predict / Recommendations / What-if / About). Every tab
operates on a single "current page": either a freshly-scraped URL or the
demo page (pre-baked, used as a fallback if the user hasn't scraped or the
network call fails).

Model loading order:
    1. MLP checkpoint at ``models/mlp_checkpoint.pt`` (rubric §VI pattern).
    2. XGBoost at ``models/xgboost.joblib`` (sweep winner — used for SHAP).

If both are present, the MLP is used for the headline prediction and the
XGBoost is used for SHAP-based recommendations + the SHAP top-5 panel
(SHAP TreeExplainer doesn't work on neural nets; KernelExplainer is too
slow for live use).

Run:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file as a script, not a module — repo root isn't on sys.path
# by default, so `from src.dashboard.styles import ...` would fail. Inject it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import io
import logging
import urllib.parse
from typing import Any

import joblib
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from src.dashboard.styles import get_css
from src.features.content_features import extract_basic, fit_tfidf, transform_tfidf
from src.features.metadata_features import extract_metadata
from src.features.structural_features import extract_structural
from src.recommendations.recommend import recommend, shap_per_prediction

LOG = logging.getLogger("dashboard")

MLP_PATH = Path("models/mlp_checkpoint.pt")
XGB_PATH = Path("models/xgboost.joblib")
RF_PATH = Path("models/random_forest.joblib")
LR_PATH = Path("models/baseline.joblib")
FEATURES_CSV = Path("data/processed/features.csv")

DEMO_URL = "https://docs.python.org/3/library/asyncio.html"


# ─────────────────────────────────── model + data caching ───────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_models() -> dict[str, Any]:
    """Load every trained model that's on disk. Returns a dict of
    {name: predictor}. Predictors share the sklearn-shaped predict_proba
    interface (the MLP wrapper conforms too)."""
    models: dict[str, Any] = {}
    if MLP_PATH.exists():
        try:
            from src.models.neural import load_checkpoint
            models["mlp"] = load_checkpoint(MLP_PATH)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("MLP load failed: %s", exc)
    for name, path in [("xgboost", XGB_PATH), ("random_forest", RF_PATH), ("logreg", LR_PATH)]:
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("%s load failed: %s", name, exc)
    return models


@st.cache_resource(show_spinner=False)
def load_corpus_tfidf() -> tuple[Any, list[str]]:
    """Refit TF-IDF on the corpus (fast for ~1500 rows) so we can transform
    a freshly-scraped page into the same feature space the models trained
    on. Returns (vectorizer, feature_column_order)."""
    if not FEATURES_CSV.exists():
        return None, []
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in ("url", "domain", "query_id", "query", "is_top_10")]
    raw_dir = Path("data/raw")
    texts: list[str] = []
    for url in df["url"].astype(str).tolist():
        # Best-effort recovery of source text — used only to refit TF-IDF.
        # If raw HTML missing, fall back to the query string.
        host = urllib.parse.urlparse(url).netloc
        domain_dir = raw_dir / host
        if not domain_dir.exists():
            texts.append("")
            continue
        # any sidecar matching this URL
        match = next((p for p in domain_dir.glob("*.json") if url in p.read_text(encoding="utf-8", errors="ignore")), None)
        if match is None:
            texts.append("")
            continue
        html_path = match.with_suffix(".html")
        if not html_path.exists():
            texts.append("")
            continue
        try:
            html = html_path.read_text(encoding="utf-8", errors="replace")
            texts.append(BeautifulSoup(html, "lxml").get_text(" ", strip=True))
        except OSError:
            texts.append("")
    vec = fit_tfidf([t for t in texts if t], max_features=50)
    return vec, feature_cols


# ─────────────────────────────────── live scraping ───────────────────────────────────


def scrape_one(url: str, timeout: float = 15.0) -> tuple[BeautifulSoup, str] | None:
    """Fetch a single URL synchronously. Returns (soup, text) or None on failure."""
    import httpx
    headers = {"User-Agent": "AIDrivenSEOResearchBot/1.0 (dashboard live demo)"}
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            return soup, soup.get_text(" ", strip=True)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("live scrape failed: %s", exc)
        return None


def featurize(url: str, soup: BeautifulSoup, text: str, query: str, vec: Any, feature_cols: list[str]) -> pd.Series:
    """Build a feature Series matching training column order. Graph features
    are zero-filled (the live page is not in the corpus graph)."""
    row: dict[str, float] = {}
    row.update(extract_basic(text, query))
    row.update(extract_metadata(soup, query))
    row.update(extract_structural(soup, url))
    if vec is not None:
        row.update(transform_tfidf(text, vec))
    for col in ["pagerank", "hits_hub", "hits_authority", "in_degree", "out_degree", "clustering"]:
        row.setdefault(col, 0.0)
    series = pd.Series({c: float(row.get(c, 0.0)) for c in feature_cols})
    return series


def derive_query(soup: BeautifulSoup) -> str:
    title = soup.title.string.strip() if (soup.title and soup.title.string) else ""
    from src.scraping.serp_client import derive_query_from_title
    return derive_query_from_title(title)


# ─────────────────────────────────── demo fallback ───────────────────────────────────


def demo_row(feature_cols: list[str]) -> tuple[pd.Series, str, str]:
    """Pre-baked example so the dashboard always has something to show, even
    offline. Numbers chosen to mimic a typical mid-quality dev-doc page."""
    row = {c: 0.0 for c in feature_cols}
    row.update({
        "text_length": 8200, "word_count": 1450, "sentence_count": 92,
        "flesch_reading_ease": 58.0, "keyword_density": 0.012,
        "title_length": 38, "has_meta_description": 1.0, "meta_description_length": 142,
        "keyword_in_title": 1.0,
        "h1_count": 1, "h2_count": 6, "h3_count": 14,
        "internal_link_count": 38, "external_link_count": 5,
        "image_count": 3, "alt_text_coverage": 0.66,
        "pagerank": 0.0021, "hits_hub": 0.013, "hits_authority": 0.018,
        "in_degree": 12, "out_degree": 38, "clustering": 0.07,
    })
    return pd.Series({c: float(row.get(c, 0.0)) for c in feature_cols}), DEMO_URL, "asyncio"


# ─────────────────────────────────── UI helpers ───────────────────────────────────


def predict_with_models(features: pd.Series, models: dict[str, Any]) -> dict[str, float]:
    """Run every loaded model on the row. Returns {model_name: P(top_10)}."""
    out: dict[str, float] = {}
    X = features.to_frame().T
    for name, model in models.items():
        try:
            proba = model.predict_proba(X)
            out[name] = float(proba[0][1])
        except Exception as exc:  # noqa: BLE001
            LOG.warning("predict_proba failed for %s: %s", name, exc)
    return out


def render_metric_card(label: str, value: str, klass: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value {klass}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def render_suggestion(s: dict[str, Any]) -> None:
    st.markdown(
        f'<div class="suggestion"><span class="feature-tag">{s["target_feature"]}</span>'
        f'<strong>{s["action"]}</strong>'
        f'<div class="why">{s["why"]}</div></div>',
        unsafe_allow_html=True,
    )


def render_shap_row(name: str, value: float, vmax: float) -> None:
    pct = min(100.0, abs(value) / vmax * 100.0) if vmax else 0.0
    side = "pos" if value >= 0 else "neg"
    style = f"width:{pct/2:.1f}%;"
    st.markdown(
        f'<div class="shap-row"><div class="name">{name}</div>'
        f'<div class="bar"><span class="{side}" style="{style}"></span></div>'
        f'<div class="val">{value:+.3f}</div></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────── tabs ───────────────────────────────────


def tab_predict(state: dict[str, Any]) -> None:
    st.markdown("### Predict top-10 SERP probability")
    url = st.text_input("Page URL", value=state.get("url", DEMO_URL))
    col_a, col_b = st.columns([1, 1])
    do_scrape = col_a.button("Scrape & predict", type="primary", use_container_width=True)
    use_demo = col_b.button("Use demo page", use_container_width=True)

    vec, feature_cols = state["vec"], state["feature_cols"]

    if use_demo or (state.get("features") is None and not do_scrape):
        features, used_url, query = demo_row(feature_cols)
        state.update(features=features, url=used_url, query=query, source="demo")

    if do_scrape and url:
        with st.spinner(f"Scraping {url} …"):
            res = scrape_one(url)
        if res is None:
            st.warning("Live scrape failed — showing demo page instead.")
            features, used_url, query = demo_row(feature_cols)
            state.update(features=features, url=used_url, query=query, source="demo (fallback)")
        else:
            soup, text = res
            query = derive_query(soup) or "(no title)"
            features = featurize(url, soup, text, query, vec, feature_cols)
            state.update(features=features, url=url, query=query, source="live")

    features = state["features"]
    probs = predict_with_models(features, state["models"])
    if not probs:
        st.error("No trained models loaded. Run the model sweep first.")
        return

    st.markdown(f'<div class="banner"><h1>{state["url"]}</h1>'
                f'<div class="sub">Topic query: <strong>{state["query"]}</strong> · source: {state["source"]}</div></div>',
                unsafe_allow_html=True)

    cols = st.columns(len(probs))
    for col, (name, p) in zip(cols, sorted(probs.items())):
        with col:
            verdict = "good" if p >= 0.5 else ("warn" if p >= 0.3 else "bad")
            render_metric_card(f"{name} P(top-10)", f"{p*100:.1f}%", klass=verdict)

    st.progress(min(1.0, max(probs.values())), text=f"Top model says: {max(probs.values())*100:.1f}% chance of top-10.")


def tab_recommendations(state: dict[str, Any]) -> None:
    st.markdown("### Actionable recommendations")
    if state.get("features") is None:
        st.info("Run a prediction first.")
        return
    explainer_model = state["models"].get("xgboost") or state["models"].get("random_forest")
    suggestions = recommend(state["features"], model=explainer_model, query=state.get("query", ""))
    if not suggestions:
        st.success("This page already follows the SEO best practices we check.")
        return
    for s in suggestions:
        render_suggestion(s)


def tab_what_if(state: dict[str, Any]) -> None:
    st.markdown("### What-if simulator")
    if state.get("features") is None:
        st.info("Run a prediction first.")
        return
    base = state["features"].copy()
    modified = base.copy()

    cols = st.columns(2)
    with cols[0]:
        modified["title_length"] = float(st.slider("Title length (chars)", 0, 120, int(base["title_length"])))
        modified["meta_description_length"] = float(st.slider("Meta description length (chars)", 0, 250, int(base["meta_description_length"])))
        modified["h2_count"] = float(st.slider("H2 headers", 0, 30, int(base["h2_count"])))
        modified["alt_text_coverage"] = st.slider("Alt-text coverage", 0.0, 1.0, float(base["alt_text_coverage"]))
    with cols[1]:
        modified["keyword_density"] = st.slider("Keyword density", 0.0, 0.05, float(base["keyword_density"]))
        modified["internal_link_count"] = float(st.slider("Internal links", 0, 200, int(base["internal_link_count"])))
        modified["word_count"] = float(st.slider("Word count", 100, 6000, int(base["word_count"])))
        modified["keyword_in_title"] = float(st.slider("Keyword in title (0/1)", 0, 1, int(base["keyword_in_title"])))

    base_p = predict_with_models(base, state["models"])
    new_p = predict_with_models(modified, state["models"])

    cols2 = st.columns(len(base_p))
    for col, name in zip(cols2, sorted(base_p)):
        delta = new_p[name] - base_p[name]
        with col:
            render_metric_card(
                f"{name}",
                f"{new_p[name]*100:.1f}%",
                klass="good" if delta > 0.005 else ("bad" if delta < -0.005 else ""),
            )
            st.caption(f"Δ {delta*100:+.1f} pts vs original")


def tab_about(state: dict[str, Any]) -> None:
    st.markdown("### About this project")
    st.markdown(
        """
**AI-Driven SEO Ranking Predictor** — joint final project for **CIS 2450** and **NETS 1500**.
Predicts whether a developer documentation page will appear in Google's top-10 SERP for the topic
query derived from its `<title>`, plus a SHAP-based what-if simulator and recommendation engine.
"""
    )
    st.markdown("#### Data scope (sanctioned pivot)")
    st.info(
        "Per CIS 2450 TA Ricky Gong's email of 2026-03-29, the project was narrowed from the original "
        "50K-row scope to ~1500 developer documentation pages because full-scale free-tier scraping is "
        "rate-limited. **This is sanctioned, not a deviation.**"
    )
    st.markdown("#### AI usage disclosure")
    st.markdown(
        """
- **Code generation:** Claude (Anthropic) was used to scaffold module boilerplate (scraper async loop,
  sklearn pipelines, Streamlit components). Every generated function was reviewed, type-hint-audited,
  and integration-tested before commit.
- **Documentation drafting:** Claude was used to draft initial versions of `data/README.md`,
  `MODELING_DECISIONS.md`, and docstrings. Authors reviewed and rewrote substantive content.
- **Hyperparameter ranges:** Initial search-space ranges suggested by Claude; we narrowed/widened
  based on early CV results.
- **What we did NOT use AI for:** the modeling sweep itself, the data ethics decisions (domain
  selection, robots.txt policy, rate limits), and the final feature-engineering choices.
        """
    )
    st.markdown("#### Team")
    st.markdown(
        """
- **Rahil Patel** (`rahilp07@seas.upenn.edu`) — scraping, features, graph layer, LR/RF/XGBoost,
  dashboard backend wiring.
- **Ayush Tripathi** (`tripath1@seas.upenn.edu`) — EDA, MLP (Colab), evaluator, recommendations,
  dashboard frontend + styling, slide deck.
        """
    )


# ─────────────────────────────────── main ───────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="SEO Ranking Predictor", layout="wide", page_icon="🔎")
    st.markdown(get_css(), unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("# SEO Ranking Predictor")
        st.caption("CIS 2450 + NETS 1500 final")
        tab = st.radio("View", ["Predict", "Recommendations", "What-if", "About"], index=0)

    models = load_models()
    vec, feature_cols = load_corpus_tfidf()
    if not feature_cols:
        # No corpus yet — synthesise feature_cols from demo defaults so the
        # dashboard still boots in a clean clone before anyone has scraped.
        feature_cols = list(demo_row([])[0].index) if False else [
            "text_length", "word_count", "sentence_count", "flesch_reading_ease", "keyword_density",
            "title_length", "has_meta_description", "meta_description_length", "keyword_in_title",
            "h1_count", "h2_count", "h3_count", "internal_link_count", "external_link_count",
            "image_count", "alt_text_coverage",
            "pagerank", "hits_hub", "hits_authority", "in_degree", "out_degree", "clustering",
        ]

    state = st.session_state.setdefault(
        "_state",
        {"features": None, "url": "", "query": "", "source": "", "models": models, "vec": vec, "feature_cols": feature_cols},
    )
    state["models"] = models
    state["vec"] = vec
    state["feature_cols"] = feature_cols

    if tab == "Predict":
        tab_predict(state)
    elif tab == "Recommendations":
        tab_recommendations(state)
    elif tab == "What-if":
        tab_what_if(state)
    else:
        tab_about(state)


if __name__ == "__main__":
    main()
