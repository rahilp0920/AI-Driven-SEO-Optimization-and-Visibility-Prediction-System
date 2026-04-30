"""Streamlit dashboard — SEO Ranking Predictor.

Sidebar tabs (Predict / Recommendations / What-if / About). The current page is built from a live HTTP scrape of the URL the user enters
(same feature pipeline as training, with graph features zero-filled for out-of-
graph URLs). The topic query defaults to the same ``<title>``-derived rule as
training; users may optionally override it for keyword features without
re-scraping.

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

import html
import logging
import urllib.parse
from typing import Any

import joblib
import numpy as np
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
            html_text = html_path.read_text(encoding="utf-8", errors="replace")
            texts.append(BeautifulSoup(html_text, "lxml").get_text(" ", strip=True))
        except OSError:
            texts.append("")
    corpus_texts = [t for t in texts if t]
    vec = fit_tfidf(corpus_texts, max_features=50) if corpus_texts else None
    return vec, feature_cols


@st.cache_resource(show_spinner="Computing reference distributions…")
def load_reference_probs(_models: dict[str, Any], feature_cols: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Run each loaded model over the full training feature matrix once so the
    dashboard can express live predictions as percentiles within the corpus
    distribution. Returns {model_name: sorted_proba_array}.

    The leading-underscore on ``_models`` tells Streamlit to skip hashing it
    (model objects aren't reliably hashable). ``feature_cols`` is a tuple so
    the cache key is stable across reruns."""
    if not FEATURES_CSV.exists() or not _models:
        return {}
    df = pd.read_csv(FEATURES_CSV)
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].select_dtypes(include=[np.number]).fillna(0.0)
    out: dict[str, np.ndarray] = {}
    for name, model in _models.items():
        try:
            p = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
            out[name] = np.sort(p)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("reference predict_proba failed for %s: %s", name, exc)
    return out


# ─────────────────────────────────── live scraping ───────────────────────────────────


def scrape_one(url: str, timeout: float = 15.0) -> tuple[BeautifulSoup, str] | None:
    """Fetch a single URL synchronously. Returns (soup, text) or None on failure."""
    import httpx
    headers = {"User-Agent": "AIDrivenSEOResearchBot/1.0 (dashboard scrape)"}
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


def _percentile_of(value: float, sorted_ref: np.ndarray) -> float:
    """0-100 percentile rank of ``value`` within a sorted reference vector."""
    if sorted_ref.size == 0:
        return float("nan")
    rank = float(np.searchsorted(sorted_ref, value, side="right"))
    return 100.0 * rank / float(sorted_ref.size)


def compute_seo_score(
    probs: dict[str, float],
    refs: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Combine per-model probabilities into a single 0-100 SEO score.

    For each model we compute the percentile rank of the live ``P(top_10)``
    inside that model's training-set probability distribution. The mean of
    those percentiles is reported as the SEO score (relative ranking, not a
    calibrated probability). ``agreement`` summarises spread across models.
    """
    pct: dict[str, float] = {}
    for name, p in probs.items():
        ref = refs.get(name)
        if ref is None or ref.size == 0:
            continue
        pct[name] = _percentile_of(float(p), ref)

    if not pct:
        return {
            "score": float("nan"),
            "verdict": "n/a",
            "agreement": "n/a",
            "spread": float("nan"),
            "per_model": {},
        }

    values = list(pct.values())
    score = float(np.mean(values))
    spread = float(max(values) - min(values))

    if score >= 70.0:
        verdict = "Strong"
    elif score >= 40.0:
        verdict = "Moderate"
    else:
        verdict = "Weak"

    if spread <= 15.0:
        agreement = "high"
    elif spread <= 35.0:
        agreement = "mixed"
    else:
        agreement = "low"

    return {
        "score": score,
        "verdict": verdict,
        "agreement": agreement,
        "spread": spread,
        "per_model": pct,
    }


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


def _render_seo_score(
    score_info: dict[str, Any],
    probs: dict[str, float],
    refs: dict[str, np.ndarray],
) -> None:
    """Headline SEO score (percentile-rank ensemble), agreement, and per-model
    breakdown in a collapsible expander."""
    score = score_info.get("score", float("nan"))
    verdict = str(score_info.get("verdict", "n/a"))
    agreement = str(score_info.get("agreement", "n/a"))
    pct: dict[str, float] = score_info.get("per_model") or {}

    if not refs or not pct or not np.isfinite(score):
        st.warning(
            "SEO score unavailable — reference distribution missing. "
            "Showing per-model probabilities only."
        )
        cols = st.columns(len(probs)) if probs else []
        for col, (name, p) in zip(cols, sorted(probs.items())):
            with col:
                klass = "good" if p >= 0.5 else ("warn" if p >= 0.3 else "bad")
                render_metric_card(f"{name} P(top-10)", f"{p*100:.1f}%", klass=klass)
        return

    if verdict == "Strong":
        klass = "good"
    elif verdict == "Moderate":
        klass = "warn"
    else:
        klass = "bad"

    headline_cols = st.columns([2, 1, 1])
    with headline_cols[0]:
        render_metric_card("SEO score", f"{score:.0f} / 100", klass=klass)
    with headline_cols[1]:
        render_metric_card("Verdict", verdict, klass=klass)
    with headline_cols[2]:
        render_metric_card("Model agreement", agreement, klass="" if agreement == "high" else "warn")

    st.progress(
        min(1.0, max(0.0, score / 100.0)),
        text=(
            f"{verdict} SEO signal — page ranks at the {score:.0f}th percentile "
            f"of the trained corpus (ensemble)."
        ),
    )

    st.caption(
        "Score = mean of per-model **percentile ranks** within the training-set probability "
        "distribution. Use as a relative SEO quality signal across pages, not as a literal "
        "top-10 probability."
    )

    with st.expander("Per-model breakdown", expanded=False):
        cols = st.columns(len(probs))
        for col, name in zip(cols, sorted(probs)):
            p = probs[name]
            pc = pct.get(name, float("nan"))
            with col:
                pclass = "good" if pc >= 70 else ("warn" if pc >= 40 else "bad")
                render_metric_card(name, f"P={p*100:.1f}%", klass=pclass)
                st.caption(f"percentile: {pc:.0f}")


# ─────────────────────────────────── tabs ───────────────────────────────────


def _manual_query_override() -> str:
    """Reads the optional topic-query widget (Streamlit session key)."""
    return (st.session_state.get("topic_query_override") or "").strip()


def _resolve_topic_query(soup: BeautifulSoup) -> tuple[str, str, str]:
    """Return ``(query_used_for_features, title_derived_query, 'manual'|'title')``."""
    derived = derive_query(soup) or "(no title)"
    manual = _manual_query_override()
    if manual:
        return manual, derived, "manual"
    return derived, derived, "title"


def tab_predict(state: dict[str, Any]) -> None:
    st.markdown("### Predict top-10 SERP probability")
    url = st.text_input("Page URL", value=state.get("url", ""), placeholder="https://docs.example.com/…")

    st.caption(
        "Training labels use one query per page, derived from **<title>** (stripped site suffixes). "
        "If that string is a poor search topic (e.g. a version number only), set an override below."
    )
    st.text_input(
        "Topic query override (optional)",
        key="topic_query_override",
        placeholder='e.g. Python 3 documentation — leave empty for title-derived (matches training)',
        help="Used for keyword-in-title, keyword density, and related text features. "
        "Does not re-fetch SERP; the model still predicts top-10 probability from features only.",
    )

    c1, c2 = st.columns(2)
    with c1:
        do_scrape = st.button("Scrape & predict", type="primary", use_container_width=True)
    with c2:
        do_repredict = st.button(
            "Re-predict with query only",
            use_container_width=True,
            disabled=state.get("_scrape_soup") is None,
            help="Uses the last successful scrape and the override box (or title-derived if empty).",
        )

    vec, feature_cols = state["vec"], state["feature_cols"]

    if do_scrape:
        if not (url or "").strip():
            st.warning("Enter a page URL first.")
        else:
            with st.spinner(f"Scraping {url} …"):
                res = scrape_one(url.strip())
            if res is None:
                st.error("Could not fetch that URL (network error, timeout, or HTTP error). Try another URL.")
            else:
                soup, text = res
                q_use, q_derived, q_src = _resolve_topic_query(soup)
                features = featurize(url.strip(), soup, text, q_use, vec, feature_cols)
                state.update(
                    features=features,
                    url=url.strip(),
                    query=q_use,
                    derived_query=q_derived,
                    query_source=q_src,
                    _scrape_soup=soup,
                    _scrape_text=text,
                )

    if do_repredict and state.get("_scrape_soup") is not None:
        soup = state["_scrape_soup"]
        text = state["_scrape_text"]
        u = state.get("url") or ""
        q_use, q_derived, q_src = _resolve_topic_query(soup)
        features = featurize(u, soup, text, q_use, vec, feature_cols)
        state.update(
            features=features,
            query=q_use,
            derived_query=q_derived,
            query_source=q_src,
        )

    if state.get("features") is None:
        st.info("Enter a documentation page URL and click **Scrape & predict** to run the models.")
        return

    features = state["features"]
    probs = predict_with_models(features, state["models"])
    if not probs:
        st.error("No trained models loaded. Run the model sweep first.")
        return

    q_esc = html.escape(str(state.get("query", "")))
    u_esc = html.escape(str(state.get("url", "")))
    sub = f"Topic query: <strong>{q_esc}</strong>"
    if state.get("query_source") == "manual" and state.get("derived_query"):
        d_esc = html.escape(str(state["derived_query"]))
        if str(state.get("query")) != str(state.get("derived_query")):
            sub += f' <span style="opacity:0.85;font-size:0.92em">(title-derived: {d_esc})</span>'
        sub += ' <span style="opacity:0.85;font-size:0.92em">· manual override</span>'
    else:
        sub += ' <span style="opacity:0.85;font-size:0.92em">· from page title (training default)</span>'

    st.markdown(
        f'<div class="banner"><h1>{u_esc}</h1><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

    refs: dict[str, np.ndarray] = state.get("refs") or {}
    score_info = compute_seo_score(probs, refs)
    _render_seo_score(score_info, probs, refs)


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

    refs: dict[str, np.ndarray] = state.get("refs") or {}
    base_score = compute_seo_score(base_p, refs)
    new_score = compute_seo_score(new_p, refs)

    headline_cols = st.columns([2, 1, 1])
    delta_score = float("nan")
    if np.isfinite(new_score["score"]) and np.isfinite(base_score["score"]):
        delta_score = float(new_score["score"] - base_score["score"])

    if new_score["verdict"] == "Strong":
        klass = "good"
    elif new_score["verdict"] == "Moderate":
        klass = "warn"
    else:
        klass = "bad"

    with headline_cols[0]:
        render_metric_card("New SEO score", f"{new_score['score']:.0f} / 100", klass=klass)
    with headline_cols[1]:
        render_metric_card("Verdict", str(new_score["verdict"]), klass=klass)
    with headline_cols[2]:
        if np.isfinite(delta_score):
            d_klass = "good" if delta_score > 0.5 else ("bad" if delta_score < -0.5 else "")
            render_metric_card("Δ vs original", f"{delta_score:+.1f}", klass=d_klass)
        else:
            render_metric_card("Δ vs original", "n/a")

    if np.isfinite(new_score["score"]):
        st.progress(
            min(1.0, max(0.0, new_score["score"] / 100.0)),
            text=f"Adjusted page ranks at the {new_score['score']:.0f}th percentile of the trained corpus.",
        )

    with st.expander("Per-model breakdown", expanded=False):
        cols2 = st.columns(len(base_p))
        for col, name in zip(cols2, sorted(base_p)):
            delta_p = new_p[name] - base_p[name]
            with col:
                render_metric_card(
                    f"{name}",
                    f"P={new_p[name]*100:.1f}%",
                    klass="good" if delta_p > 0.005 else ("bad" if delta_p < -0.005 else ""),
                )
                st.caption(f"Δ {delta_p*100:+.1f} pts vs original")


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
    if not FEATURES_CSV.exists():
        st.error(
            f"Missing `{FEATURES_CSV}` — run `make features` (after scrape + SERP) so the "
            "dashboard can align TF-IDF and column order with training."
        )
        st.stop()

    vec, feature_cols = load_corpus_tfidf()
    if not feature_cols:
        st.error(f"`{FEATURES_CSV}` has no usable feature columns. Re-run `make features`.")
        st.stop()

    refs = load_reference_probs(models, tuple(feature_cols))

    state = st.session_state.setdefault(
        "_state",
        {
            "features": None,
            "url": "",
            "query": "",
            "derived_query": "",
            "query_source": "title",
            "_scrape_soup": None,
            "_scrape_text": None,
            "models": models,
            "vec": vec,
            "feature_cols": feature_cols,
            "refs": refs,
        },
    )
    state["models"] = models
    state["vec"] = vec
    state["feature_cols"] = feature_cols
    state["refs"] = refs
    state.setdefault("derived_query", "")
    state.setdefault("query_source", "title")
    state.setdefault("_scrape_soup", None)
    state.setdefault("_scrape_text", None)

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
