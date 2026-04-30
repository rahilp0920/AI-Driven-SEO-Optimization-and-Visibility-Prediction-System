"""Streamlit dashboard — SEO Ranking Predictor.

Sidebar navigation across seven tabs:

* **Predict**          live URL scrape → top-10 probability + per-model breakdown.
* **EDA**              corpus exploration: class balance, domain mix, feature
                       distributions, correlation heatmap, top-feature ranking.
* **Graph**            link-graph view: PageRank distribution, HITS hub vs
                       authority, in/out-degree, URL hierarchy network.
* **Models**           model comparison: metrics table, ROC + PR curves,
                       confusion matrices, feature importance.
* **Recommendations**  rules + SHAP-ranked actionable suggestions.
* **What-if**          slider-driven counterfactual probability simulator.
* **About**            objective, data sources, work split, methodology notes.

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

import json
import logging
import urllib.parse
from typing import Any

import joblib
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from src.dashboard.components import charts as ch
from src.dashboard.components.model_helpers import (
    confusion_for_model,
    feature_columns_from_model,
    model_curves,
    model_feature_importance,
)
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
METRICS_DIR = Path("models/metrics")
FEATURES_CSV = Path("data/processed/features.csv")

DEMO_URL = "https://docs.python.org/3/library/asyncio.html"

NAV_TABS = ["Predict", "EDA", "Graph", "Models", "Recommendations", "What-if", "About"]


# ─────────────────────────────────── caching ───────────────────────────────────


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


@st.cache_data(show_spinner=False)
def load_features_df() -> pd.DataFrame:
    """Load the processed feature matrix once per session."""
    if not FEATURES_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURES_CSV)


@st.cache_data(show_spinner=False)
def load_saved_metrics() -> dict[str, dict[str, Any]]:
    """Read every ``models/metrics/*.json`` produced by the training CLI."""
    out: dict[str, dict[str, Any]] = {}
    if not METRICS_DIR.exists():
        return out
    for path in sorted(METRICS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out[data.get("model", path.stem)] = data
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("metrics load failed for %s: %s", path, exc)
    return out


@st.cache_resource(show_spinner=False)
def load_corpus_tfidf() -> tuple[Any, list[str]]:
    """Refit TF-IDF on the corpus (fast for ~100 rows) so we can transform
    a freshly-scraped page into the same feature space the models trained
    on. Returns (vectorizer, feature_column_order). When raw HTML is missing
    we fall back to an empty corpus and the live-scrape path skips TF-IDF."""
    df = load_features_df()
    if df.empty:
        return None, []
    feature_cols = [c for c in df.columns if c not in ("url", "domain", "query_id", "query", "is_top_10")]
    raw_dir = Path("data/raw")
    texts: list[str] = []
    for url in df["url"].astype(str).tolist():
        host = urllib.parse.urlparse(url).netloc
        domain_dir = raw_dir / host
        if not domain_dir.exists():
            texts.append("")
            continue
        match = next((p for p in domain_dir.glob("*.json")
                      if url in p.read_text(encoding="utf-8", errors="ignore")), None)
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
    non_empty = [t for t in texts if t]
    vec = fit_tfidf(non_empty, max_features=50) if non_empty else None
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


def featurize(url: str, soup: BeautifulSoup, text: str, query: str,
              vec: Any, feature_cols: list[str]) -> pd.Series:
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
    return pd.Series({c: float(row.get(c, 0.0)) for c in feature_cols})


def derive_query(soup: BeautifulSoup) -> str:
    title = soup.title.string.strip() if (soup.title and soup.title.string) else ""
    from src.scraping.serp_client import derive_query_from_title
    return derive_query_from_title(title)


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


# ─────────────────────────────────── render helpers ───────────────────────────────────


def render_metric_card(label: str, value: str, klass: str = "", sub: str | None = None) -> None:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value {klass}">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def render_stat(label: str, value: str) -> None:
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">{label}</div>'
        f'<div class="stat-num">{value}</div></div>',
        unsafe_allow_html=True,
    )


def render_callout(html: str) -> None:
    st.markdown(f'<div class="callout">{html}</div>', unsafe_allow_html=True)


def render_section_header(title: str, hint: str = "") -> None:
    hint_html = f'<span class="hint">{hint}</span>' if hint else ""
    st.markdown(
        f'<div class="section-header"><h3>{title}</h3>{hint_html}</div>',
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
    st.markdown("### Live SERP top-10 predictor")
    render_callout(
        "Paste any developer-doc URL. We scrape it, extract the same content / "
        "metadata / structural / TF-IDF features the models trained on, and run "
        "every loaded model. Override the auto-derived topic query if it doesn't "
        "match the search intent you want to evaluate against."
    )

    url = st.text_input("Page URL", value=state.get("url", DEMO_URL))
    query_input = st.text_input(
        "Topic query (override)",
        value=state.get("query", ""),
        help="Auto-filled from the page <title> after scraping. Edit to evaluate "
             "against a different search intent.",
    )

    col_a, col_b, col_c = st.columns(3)
    do_scrape = col_a.button("Scrape & predict", type="primary", width="stretch")
    do_requery = col_b.button("Apply query", width="stretch",
                              help="Recompute features with the query above (no rescrape).")
    use_demo = col_c.button("Use demo page", width="stretch")

    vec, feature_cols = state["vec"], state["feature_cols"]

    if use_demo or (state.get("features") is None and not do_scrape and not do_requery):
        features, used_url, query = demo_row(feature_cols)
        state.update(features=features, url=used_url, query=query, source="demo",
                     soup=None, text=None)

    if do_scrape and url:
        with st.spinner(f"Scraping {url} …"):
            res = scrape_one(url)
        if res is None:
            st.warning("Live scrape failed — showing demo page instead.")
            features, used_url, query = demo_row(feature_cols)
            state.update(features=features, url=used_url, query=query, source="demo (fallback)",
                         soup=None, text=None)
        else:
            soup, text = res
            query = query_input.strip() or derive_query(soup) or "(no title)"
            features = featurize(url, soup, text, query, vec, feature_cols)
            state.update(features=features, url=url, query=query, source="live",
                         soup=soup, text=text)

    if do_requery and query_input.strip():
        new_query = query_input.strip()
        soup, text = state.get("soup"), state.get("text")
        if soup is not None and text is not None:
            features = featurize(state["url"], soup, text, new_query, vec, feature_cols)
            state.update(features=features, query=new_query, source="live (custom query)")
        else:
            state.update(query=new_query,
                         source=f"{state.get('source', 'demo')} + custom query")

    features = state["features"]
    probs = predict_with_models(features, state["models"])
    if not probs:
        st.error("No trained models loaded. Run the model sweep first.")
        return

    st.markdown(f'<div class="banner"><h1>{state["url"]}</h1>'
                f'<div class="sub">Topic query: <strong>{state["query"]}</strong> · '
                f'source: {state["source"]}</div></div>',
                unsafe_allow_html=True)

    cols = st.columns(len(probs))
    for col, (name, p) in zip(cols, sorted(probs.items())):
        with col:
            verdict = "good" if p >= 0.5 else ("warn" if p >= 0.3 else "bad")
            sub = "Likely top-10" if p >= 0.5 else ("Borderline" if p >= 0.3 else "Below cutoff")
            render_metric_card(f"{name} · P(top-10)", f"{p*100:.1f}%", klass=verdict, sub=sub)

    st.progress(min(1.0, max(probs.values())),
                text=f"Top model says: {max(probs.values())*100:.1f}% chance of top-10.")

    explainer_model = state["models"].get("xgboost") or state["models"].get("random_forest")
    if explainer_model is not None:
        impacts = shap_per_prediction(explainer_model, features, top_k=8)
        if impacts:
            render_section_header("Per-prediction SHAP attribution",
                                  "What pushed this score up vs. down")
            vmax = max(abs(v) for _, v in impacts) or 1.0
            for name, val in impacts:
                render_shap_row(name, val, vmax)


def tab_eda(state: dict[str, Any]) -> None:
    df = state["features_df"]
    if df.empty:
        st.warning("`data/processed/features.csv` is missing — run "
                   "`python -m src.features.build_features` to generate it.")
        return

    st.markdown("### Exploratory data analysis")
    render_callout(
        "Every modeling decision starts here — class balance dictates our metric "
        "choice (PR-AUC over accuracy), feature distributions justify scaling and "
        "outlier policy, and the correlation map drives the feature-selection "
        "discussion in <code>MODELING_DECISIONS.md</code>."
    )

    pos = int(df["is_top_10"].sum())
    neg = int(len(df) - pos)
    pos_rate = pos / max(1, len(df))
    n_domains = df["domain"].nunique() if "domain" in df.columns else 0
    n_features = len([c for c in df.columns if c not in ("url", "domain", "query_id", "query", "is_top_10")])

    cols = st.columns(4)
    with cols[0]: render_stat("Pages", f"{len(df):,}")
    with cols[1]: render_stat("Top-10 rate", f"{pos_rate*100:.1f}%")
    with cols[2]: render_stat("Domains", f"{n_domains}")
    with cols[3]: render_stat("Features", f"{n_features}")

    render_section_header("Class balance & domain mix",
                          f"{pos} positive · {neg} negative · imbalance ratio {neg/max(1,pos):.1f}:1")
    c1, c2 = st.columns([1, 1.3])
    with c1: st.plotly_chart(ch.class_balance_bar(df), width="stretch")
    with c2: st.plotly_chart(ch.domain_breakdown_bar(df), width="stretch")

    render_section_header("Top features by |correlation| with target",
                          "Bar length = strength of linear association with `is_top_10`")
    st.plotly_chart(ch.top_features_correlation_bar(df, top_k=12), width="stretch")

    render_section_header("Feature distributions by class",
                          "Pick any numeric feature — overlaid histogram + box plot")
    candidate_features = [c for c in df.columns
                          if c not in ("url", "domain", "query_id", "query", "is_top_10")
                          and not c.startswith("tfidf_")]
    default_idx = candidate_features.index("word_count") if "word_count" in candidate_features else 0
    selected = st.selectbox("Feature", candidate_features, index=default_idx)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(ch.feature_histogram(df, selected), width="stretch")
    with c2: st.plotly_chart(ch.feature_box(df, selected), width="stretch")

    render_section_header("Correlation heatmap", "Top-15 features ordered by |ρ| with target")
    st.plotly_chart(ch.correlation_heatmap(df, top_k=15), width="stretch")

    render_section_header("Two-feature scatter", "Pick two features — class separability sanity check")
    sc1, sc2 = st.columns(2)
    with sc1:
        x_feat = st.selectbox("X axis", candidate_features,
                              index=candidate_features.index("word_count") if "word_count" in candidate_features else 0,
                              key="eda_x")
    with sc2:
        y_feat = st.selectbox("Y axis", candidate_features,
                              index=candidate_features.index("h2_count") if "h2_count" in candidate_features else 1,
                              key="eda_y")
    st.plotly_chart(ch.feature_target_scatter(df, x_feat, y_feat), width="stretch")


def tab_graph(state: dict[str, Any]) -> None:
    df = state["features_df"]
    if df.empty or "pagerank" not in df.columns:
        st.warning("Graph features missing — run `python -m src.graph.build_graph` "
                   "and `python -m src.graph.graph_features`.")
        return

    st.markdown("### Link-graph analysis")
    render_callout(
        "Pages are nodes; outbound links between scraped pages are edges. "
        "<strong>PageRank</strong> (α=0.85) measures stationary-distribution importance. "
        "<strong>HITS</strong> separates hubs (link out to authoritative pages) from "
        "authorities (linked to by hubs). All three become per-page features that the "
        "models can use directly."
    )

    pr_mean = df["pagerank"].mean()
    pr_max = df["pagerank"].max()
    in_deg_mean = df["in_degree"].mean() if "in_degree" in df.columns else 0
    out_deg_mean = df["out_degree"].mean() if "out_degree" in df.columns else 0

    cols = st.columns(4)
    with cols[0]: render_stat("Mean PageRank", f"{pr_mean:.4f}")
    with cols[1]: render_stat("Max PageRank", f"{pr_max:.4f}")
    with cols[2]: render_stat("Mean in-degree", f"{in_deg_mean:.1f}")
    with cols[3]: render_stat("Mean out-degree", f"{out_deg_mean:.1f}")

    render_section_header("Centrality distributions",
                          "Heavy-tailed shape is the signature of hub-and-spoke link economies")
    c1, c2 = st.columns([1, 1.3])
    with c1: st.plotly_chart(ch.pagerank_distribution(df), width="stretch")
    with c2: st.plotly_chart(ch.degree_scatter(df), width="stretch")

    render_section_header("HITS hub vs authority",
                          "Top-right quadrant = pages that both link to and are linked from authorities")
    st.plotly_chart(ch.hits_hub_authority_scatter(df), width="stretch")

    render_section_header("URL hierarchy network",
                          "Force-directed layout · accent = root domain · indigo = top-10 page")
    max_nodes = st.slider("Max nodes shown", min_value=20, max_value=min(120, len(df)),
                          value=min(80, len(df)), step=10)
    st.plotly_chart(ch.url_hierarchy_network(df, max_nodes=max_nodes),
                    width="stretch")

    render_section_header("Top pages by PageRank",
                          "Highest-authority pages in the corpus — the ones SEO would call \"link magnets\"")
    top = (df.sort_values("pagerank", ascending=False)
              .head(15)[["url", "domain", "pagerank", "in_degree", "out_degree", "is_top_10"]]
              .copy())
    top["pagerank"] = top["pagerank"].map(lambda v: f"{v:.4f}")
    top["is_top_10"] = top["is_top_10"].map({1: "Yes", 0: "No"})
    st.dataframe(top, width="stretch", hide_index=True)


def tab_models(state: dict[str, Any]) -> None:
    models = state["models"]
    metrics = state["metrics"]
    if not models:
        st.warning("No trained models on disk. Run the model sweep first "
                   "(`python -m src.models.{baseline,tree_models,boosting}`).")
        return

    st.markdown("### Model comparison")
    render_callout(
        "Four-model sweep: Logistic Regression (linear baseline) → Random Forest "
        "(bagging) → XGBoost (boosting) → MLP (neural). Every model was tuned with "
        "<strong>RandomizedSearchCV</strong> over a hand-narrowed search space, "
        "evaluated on a single stratified 80/20 hold-out (seed 42). PR-AUC is the "
        "tiebreaker because the data is class-imbalanced — accuracy would be "
        "misleading."
    )

    # ── Headline metrics table ──────────────────────────────────────────────
    rows = []
    for name in sorted(metrics.keys()):
        m = metrics[name]
        rows.append({
            "Model": name,
            "F1": f"{m.get('f1', 0.0):.3f}",
            "ROC-AUC": f"{m.get('roc_auc', 0.0):.3f}",
            "PR-AUC": f"{m.get('pr_auc', 0.0):.3f}",
            "Precision": f"{m.get('precision', 0.0):.3f}",
            "Recall": f"{m.get('recall', 0.0):.3f}",
            "n_test": m.get("n_test", 0),
            "n_pos_test": m.get("n_pos_test", 0),
        })
    if rows:
        render_section_header("Held-out test metrics", "Single 80/20 stratified split, seed 42")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.plotly_chart(ch.metrics_comparison_bar(metrics), width="stretch")

    # ── ROC + PR curves regenerated on the fly ──────────────────────────────
    render_section_header("ROC and precision-recall curves",
                          "Regenerated from saved estimators on the same held-out split")
    roc, pr = model_curves(models, FEATURES_CSV)
    if roc and pr:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(ch.roc_curve_chart(roc), width="stretch")
        with c2: st.plotly_chart(ch.pr_curve_chart(pr), width="stretch")
    else:
        st.info("Curve regeneration needs `data/processed/features.csv`.")

    # ── Confusion matrices side by side ─────────────────────────────────────
    render_section_header("Confusion matrices",
                          "Recomputed on the held-out split — rows are actual, columns are predicted")
    cm_cols = st.columns(len(models))
    for col, (name, model) in zip(cm_cols, sorted(models.items())):
        with col:
            cm = confusion_for_model(model, FEATURES_CSV)
            if cm is not None:
                st.plotly_chart(ch.confusion_matrix_heatmap(cm, name), width="stretch")

    # ── Feature importance for one chosen model ─────────────────────────────
    render_section_header("Feature importance",
                          "Tree models: split-gain importance · linear models: coefficient magnitude")
    importance_eligible = [n for n in models if n in ("xgboost", "random_forest", "logreg")]
    if importance_eligible:
        chosen = st.selectbox("Model", importance_eligible,
                              index=importance_eligible.index("xgboost") if "xgboost" in importance_eligible else 0)
        model = models[chosen]
        feat_cols = feature_columns_from_model(model, fallback=state["feature_cols"])
        impacts = model_feature_importance(model, feat_cols)
        if impacts:
            st.plotly_chart(ch.feature_importance_bar(impacts, top_k=15, model_name=chosen),
                            width="stretch")
        else:
            st.info(f"{chosen} doesn't expose a usable importance attribute.")


def tab_recommendations(state: dict[str, Any]) -> None:
    st.markdown("### Actionable recommendations")
    render_callout(
        "Suggestions are produced by a hybrid of (a) hard-coded SEO rules — title "
        "length 30-60 chars, ≥2 H2s, alt-text coverage ≥80%, etc. — and (b) "
        "<strong>SHAP</strong> attributions from the winning tree model, which "
        "rank rules by how much each feature would lift the predicted probability "
        "for the page in question."
    )

    if state.get("features") is None:
        st.info("Run a prediction first.")
        return
    explainer_model = state["models"].get("xgboost") or state["models"].get("random_forest")
    suggestions = recommend(state["features"], model=explainer_model,
                            query=state.get("query", ""))
    if not suggestions:
        st.success("This page already follows the SEO best practices we check.")
        return
    for s in suggestions:
        render_suggestion(s)


def tab_what_if(state: dict[str, Any]) -> None:
    st.markdown("### What-if simulator")
    render_callout(
        "Drag the sliders to nudge a feature in either direction; every model "
        "reruns on the modified row and the delta vs. the original prediction is "
        "shown below each card. This is the dashboard's interactive answer to "
        "\"what would actually move my ranking?\""
    )

    if state.get("features") is None:
        st.info("Run a prediction first.")
        return
    base = state["features"].copy()
    modified = base.copy()

    cols = st.columns(2)
    with cols[0]:
        modified["title_length"] = float(
            st.slider("Title length (chars)", 0, 120, int(base["title_length"])))
        modified["meta_description_length"] = float(
            st.slider("Meta description length (chars)", 0, 250, int(base["meta_description_length"])))
        modified["h2_count"] = float(st.slider("H2 headers", 0, 30, int(base["h2_count"])))
        modified["alt_text_coverage"] = st.slider("Alt-text coverage", 0.0, 1.0,
                                                  float(base["alt_text_coverage"]))
    with cols[1]:
        modified["keyword_density"] = st.slider("Keyword density", 0.0, 0.05,
                                                float(base["keyword_density"]))
        modified["internal_link_count"] = float(
            st.slider("Internal links", 0, 200, int(base["internal_link_count"])))
        modified["word_count"] = float(st.slider("Word count", 100, 6000, int(base["word_count"])))
        modified["keyword_in_title"] = float(
            st.slider("Keyword in title (0/1)", 0, 1, int(base["keyword_in_title"])))

    base_p = predict_with_models(base, state["models"])
    new_p = predict_with_models(modified, state["models"])

    cols2 = st.columns(len(base_p))
    for col, name in zip(cols2, sorted(base_p)):
        delta = new_p[name] - base_p[name]
        with col:
            klass = "good" if delta > 0.005 else ("bad" if delta < -0.005 else "")
            sub = f"Δ {delta*100:+.1f} pts vs original"
            render_metric_card(name, f"{new_p[name]*100:.1f}%", klass=klass, sub=sub)


def tab_about(state: dict[str, Any]) -> None:
    st.markdown("### About this project")
    st.markdown(
        """
**SEO Ranking Predictor** — final project for **CIS 2450: Big Data Analytics**.

The system predicts whether a developer-documentation page will appear in Google's top-10
SERP results for the topic query derived from its `<title>`, then explains every prediction
with a SHAP-driven what-if simulator and a concrete, rule-grounded recommendation engine.
        """
    )

    st.markdown("#### Data scope")
    st.info(
        "Per CIS 2450 TA Ricky Gong's email of 2026-03-29, the project was narrowed from the "
        "originally-scoped 50K rows to ~1500 developer-documentation pages because full-scale "
        "free-tier SERP scraping is rate-limited. The narrowing is sanctioned and documented "
        "in `data/README.md`."
    )

    st.markdown("#### Methodology")
    st.markdown(
        """
- **Two distinct sources** — scraped HTML (async, robots.txt-aware crawler) joined to
  Google SERP rankings (Brave Search API, SerpApi fallback).
- **Five feature families** — content (TF-IDF, Flesch, keyword density), metadata
  (title / meta-description), structural (heading + link counts, alt-text coverage),
  graph (PageRank α=0.85, HITS hub/authority, in/out-degree, clustering coefficient).
- **Four-model sweep** — Logistic Regression → Random Forest → XGBoost → MLP, each
  hyperparameter-tuned with RandomizedSearchCV against the same StratifiedKFold splitter.
- **Imbalance-aware metrics** — F1, ROC-AUC, PR-AUC, full confusion matrix; PR-AUC is
  the tiebreaker because positive class is a minority.
- **Explainability** — SHAP TreeExplainer on the boosted-tree winner, rendered per
  prediction in the Predict tab and aggregated for recommendations.
        """
    )

    st.markdown("#### Course-topic coverage")
    st.markdown(
        """
- **Supervised learning** — Logistic Regression, Random Forest, XGBoost, MLP.
- **Graphs** — link graph between scraped pages with PageRank, HITS hub/authority, clustering coefficient.
- **Text representations** — corpus-level TF-IDF, keyword density, title-keyword match.
- **Hypothesis testing** — chi-square / t-tests on top features by class (notebook 01).
- **Hyperparameter tuning** — RandomizedSearchCV on shared `StratifiedKFold` splits.
- **Joins** — page features joined to SERP labels on a `host + path` key.
        """
    )

    st.markdown("#### Team")
    st.markdown(
        """
- **Rahil Patel** (`rahilp07@seas.upenn.edu`) — scraping, feature pipeline, link-graph
  construction, LR / RF / XGBoost trainers, dashboard backend wiring.
- **Ayush Tripathi** (`tripath1@seas.upenn.edu`) — exploratory analysis, MLP (PyTorch),
  evaluator + metrics harness, recommendation engine, dashboard frontend + styling, slides.
        """
    )


# ─────────────────────────────────── main ───────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="SEO Ranking Predictor", layout="wide", page_icon="📈")
    st.markdown(get_css(), unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("# SEO Ranking Predictor")
        st.caption("CIS 2450 final project")
        tab = st.radio("View", NAV_TABS, index=0, label_visibility="collapsed")
        st.markdown("---")
        st.caption("**Authors**\n\nRahil Patel · Ayush Tripathi")
        st.caption("**Stack**\n\nPython · scikit-learn · XGBoost · "
                   "PyTorch · NetworkX · SHAP · Plotly · Streamlit")

    models = load_models()
    vec, feature_cols = load_corpus_tfidf()
    if not feature_cols:
        feature_cols = [
            "text_length", "word_count", "sentence_count", "flesch_reading_ease", "keyword_density",
            "title_length", "has_meta_description", "meta_description_length", "keyword_in_title",
            "h1_count", "h2_count", "h3_count", "internal_link_count", "external_link_count",
            "image_count", "alt_text_coverage",
            "pagerank", "hits_hub", "hits_authority", "in_degree", "out_degree", "clustering",
        ]

    state = st.session_state.setdefault(
        "_state",
        {"features": None, "url": "", "query": "", "source": "",
         "soup": None, "text": None,
         "models": models, "vec": vec, "feature_cols": feature_cols,
         "features_df": load_features_df(), "metrics": load_saved_metrics()},
    )
    state["models"] = models
    state["vec"] = vec
    state["feature_cols"] = feature_cols
    state["features_df"] = load_features_df()
    state["metrics"] = load_saved_metrics()

    dispatch = {
        "Predict": tab_predict,
        "EDA": tab_eda,
        "Graph": tab_graph,
        "Models": tab_models,
        "Recommendations": tab_recommendations,
        "What-if": tab_what_if,
        "About": tab_about,
    }
    dispatch.get(tab, tab_about)(state)


if __name__ == "__main__":
    main()
