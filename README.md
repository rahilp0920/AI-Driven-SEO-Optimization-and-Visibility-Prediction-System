# SEO Ranking Predictor & Recommendation System

**CIS 2450 — Big Data Analytics, Final Project**
Rahil Patel (`rahilp07@seas.upenn.edu`) · Ayush Tripathi (`tripath1@seas.upenn.edu`)

A binary classifier that predicts whether a developer-documentation page will appear
in Google's top-10 SERP results for the topic query derived from its `<title>`, paired
with an interactive Streamlit dashboard, a SHAP-driven what-if simulator, and a
rule-grounded recommendation engine.

```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

---

## Problem statement

Search-engine ranking is a black box, but the inputs are not. Page-level signals —
content length, heading structure, keyword placement, internal linking, link-graph
authority — are all observable. We frame "will this page rank in the top 10?" as a
binary classification problem, then turn the trained model into an explainability
surface that tells a documentation author **what to change** to improve their page.

## Pivot from the proposal

Per CIS 2450 TA Ricky Gong's email of 2026-03-29, the project was narrowed from the
originally-scoped 50K rows to **~1500 developer-documentation pages** because
full-scale free-tier SERP scraping is rate-limited inside the deadline. This is
sanctioned, not a deviation, and is recorded verbatim in
[`data/README.md`](data/README.md).

## Dataset

Two distinct sources, joined on a per-page key:

1. **Developer documentation HTML** — async, robots.txt-aware crawler over
   `docs.python.org`, `developer.mozilla.org`, `react.dev`, `nodejs.org`,
   `kubernetes.io`, and `fastapi.tiangolo.com` (full domain table in
   [`data/README.md`](data/README.md)). Each page is persisted to
   `data/raw/<domain>/<sha1>.html` with a `<sha1>.json` sidecar holding URL,
   fetch timestamp, parsed title, and outbound links.
2. **Google SERP rankings** — top-10 organic results per page-derived query,
   fetched via the Brave Search API (free tier; SerpApi as fallback). One query
   is generated per scraped page from its `<title>`. The page-level `is_top_10`
   label is true iff the page's own URL appears among the top-10 results for its
   own derived query.

The committed `data/processed/features.csv` is the joined feature matrix
(content + metadata + structural + TF-IDF + graph features, plus the binary
target). For the imbalance-handling rubric concept (CIS 2450 §3) we additionally
ship two derivative datasets — `features_balanced.csv` (random oversample to
class parity) and `features_augmented.csv` (bootstrap × 40 with σ = 2 % Gaussian
jitter on numeric columns, ~52K rows). Generation logic is in
[`src/features/balance.py`](src/features/balance.py); distributional fidelity
checks (means within 0.5 %, stds within 0.1 %) are in
[`MODELING_DECISIONS.md`](MODELING_DECISIONS.md).

## Feature pipeline (`src/features/`, `src/graph/`)

| Family | Source module | Examples |
|--------|---------------|----------|
| Content | `content_features.py` | `text_length`, `word_count`, `flesch_reading_ease`, `keyword_density`, 50-dim TF-IDF |
| Metadata | `metadata_features.py` | `title_length`, `has_meta_description`, `meta_description_length`, `keyword_in_title` |
| Structural | `structural_features.py` | `h1_count`, `h2_count`, `h3_count`, `internal_link_count`, `external_link_count`, `image_count`, `alt_text_coverage` |
| Graph | `graph/graph_features.py` | `pagerank` (α=0.85), `hits_hub`, `hits_authority`, `in_degree`, `out_degree`, `clustering` |

The link graph is constructed by `src/graph/build_graph.py`: nodes are scraped
pages, edges are page-to-page outbound links between scraped pages. Out-of-corpus
links are dropped so PageRank and HITS converge meaningfully on a closed graph.

## Modeling sweep (`src/models/`)

| Family | Model | Tuner | Saved artifact |
|--------|-------|-------|----------------|
| Linear | Logistic Regression (L1 / L2, balanced class weight) | RandomizedSearchCV (30 iter, 5-fold) | `models/baseline.joblib` |
| Bagging | Random Forest (depth, leaf, max-features) | RandomizedSearchCV (30 iter, 5-fold) | `models/random_forest.joblib` |
| Boosting | XGBoost (η, max-depth, subsample, colsample) | RandomizedSearchCV (30 iter, 5-fold) | `models/xgboost.joblib` |
| Neural | PyTorch MLP (2-layer, dropout, batch-norm) | Manual sweep on Colab | `models/mlp_checkpoint.pt` |

Every model uses the same `StratifiedKFold(n_splits=5, random_state=42)` splitter
and is evaluated on the same single 80/20 stratified hold-out (seed 42), so
cross-model comparisons are apples-to-apples. Per-model decisions are logged in
[`MODELING_DECISIONS.md`](MODELING_DECISIONS.md).

### Why these metrics

The data is class-imbalanced (≈ 26 % positive in the committed cut). Accuracy
would be misleading, so the metric panel is:

- **F1** — primary, balances precision and recall.
- **PR-AUC** — tiebreaker when F1 is close, more sensitive to the minority class.
- **ROC-AUC** — cross-checks the threshold-independent ranking quality.
- **Confusion matrix** — discloses which class each model is biased toward.

## Dashboard (`src/dashboard/app.py`)

Seven-tab Streamlit app with custom CSS, Plotly charts, and live URL scraping.

| Tab | What it shows |
|-----|---------------|
| **Predict** | Live URL scrape → top-10 probability per loaded model, banner with the topic query (overridable), SHAP per-prediction attribution. |
| **EDA** | Class balance, domain mix, top-feature correlation ranking, distribution explorer (histogram + box plot), correlation heatmap, two-feature scatter. |
| **Graph** | PageRank distribution, in/out-degree scatter, HITS hub vs authority (size = PageRank), URL hierarchy network rendered with a NetworkX spring layout, top-pages-by-PageRank table. |
| **Models** | Held-out metrics table, F1 / ROC-AUC / PR-AUC comparison bar, ROC + PR curves regenerated from saved estimators on the same split, side-by-side confusion matrices, feature importance for tree and linear models. |
| **Recommendations** | Hybrid rule-grounded + SHAP-ranked actionable suggestions per page. |
| **What-if** | Slider-driven counterfactual: drag a feature, watch every model's predicted probability move. |
| **About** | Objective, data scope pivot, methodology summary, team. |

## Repository layout

```
.
├── README.md                  ← this file
├── USER_MANUAL.md             ← screenshots + step-by-step walkthrough
├── MODELING_DECISIONS.md      ← per-model decision log + cut-model justifications
├── requirements.txt
├── .env.example
├── data/
│   ├── README.md              ← pivot rationale + sources + rate-limit policy
│   ├── raw/<domain>/          ← scraped HTML + JSON sidecars
│   ├── interim/               ← queries.csv, serp.csv, graph.pkl
│   └── processed/features.csv ← final feature matrix
├── src/
│   ├── scraping/              ← doc_scraper.py, serp_client.py
│   ├── features/              ← content / metadata / structural extractors + builder
│   ├── graph/                 ← build_graph.py, graph_features.py
│   ├── models/                ← baseline, tree_models, boosting, neural, evaluate
│   ├── recommendations/       ← recommend.py
│   └── dashboard/             ← app.py, styles.py, components/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_graph_analysis.ipynb
│   └── 03_model_comparison.ipynb
├── tests/                     ← pytest smoke tests
├── assets/                    ← EDA + SHAP plots, dashboard screenshots
└── presentation/              ← slides.pdf + recording.mp4
```

## CIS 2450 rubric → file map

Anchor points for graders.

### Hard requirements

| Requirement | Where |
|-------------|-------|
| Free, public, legal data | [`src/scraping/doc_scraper.py`](src/scraping/doc_scraper.py) — `urllib.robotparser` per domain, polite User-Agent, configurable rate limit. Full policy in [`data/README.md`](data/README.md). |
| ≥ 2 distinct data sources | Scraped HTML + Google SERP via Brave / SerpApi. |
| Documented narrower-domain pivot | Quoted verbatim in [`data/README.md`](data/README.md) and above. |
| 7-10+ feature columns | [`data/processed/features.csv`](data/processed/features.csv) — 5 content + 4 metadata + 7 structural + 50 TF-IDF + 6 graph = 72 numeric columns. |

### Codebase (Sections 4-7)

| Rubric item | Where |
|-------------|-------|
| Modularity (Sec. 7) | `src/{scraping,features,graph,models,recommendations,dashboard}/` — clean per-concern packages. |
| Documentation (Sec. 7) | Every module has a top-of-file docstring; every function has docstring + type hints. |
| EDA (Sec. 4a) | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) — distribution / outlier / correlation analysis tied to downstream modeling decisions. **Dashboard EDA tab** mirrors the analysis interactively. |
| Pre-processing & feature engineering (Sec. 4b) | Five feature families (content / metadata / structural / TF-IDF / graph) joined in `src/features/build_features.py`. Null fills, scaling (`StandardScaler`), correlation audit. |
| Imbalance-aware metrics (Sec. 5b) | F1 + ROC-AUC + PR-AUC + confusion matrix in [`src/models/evaluate.py`](src/models/evaluate.py). PR-AUC is the tiebreaker. |
| Hyperparameter tuning (Sec. 5b) | RandomizedSearchCV in every model file (`PARAM_DIST` constants), search results saved to `models/metrics/*.json`. |
| Code quality (Sec. 7) | `from __future__ import annotations` throughout, explicit type hints, no dead code, consistent naming. |

### Difficulty (Sec. 3)

Three concepts implemented in depth:

| Concept | Where it lives | What it produces |
|---------|----------------|------------------|
| **Feature importance** | `src/models/{boosting,tree_models,baseline}.py` train tree + linear models that expose `feature_importances_` / `coef_`; the dashboard's Models tab renders the top-15 with sign and magnitude. | Bar chart per model + SHAP per-prediction attribution. |
| **Hyperparameter tuning** | RandomizedSearchCV (loguniform priors for `C` and `learning_rate`, integer ranges for tree depths and leaf sizes) in every model trainer; CV mean ± std reported in `models/metrics/*.json`. | Tuned best estimators for LR / RF / XGBoost. |
| **Imbalance / oversampling** | `src/features/balance.py` exposes `random_oversample` (minority duplication to class parity) and `bootstrap_augment` (per-class bootstrap with σ=0.02 Gaussian jitter on numeric features). CLI: `python -m scripts.balance_dataset {oversample,bootstrap}`. Distributional fidelity (means within 0.5 %, stds within 0.1 %) verified in [`MODELING_DECISIONS.md`](MODELING_DECISIONS.md). | `data/processed/features_balanced.csv` (1.3K rows, parity) and `features_augmented.csv` (~52K rows, factor=40). |

### Application of course topics (Sec. 8)

| Topic | Where |
|-------|-------|
| Supervised learning | LR / RF / XGBoost / MLP in `src/models/`. |
| Graphs | Page-level link graph + PageRank / HITS / clustering features in `src/graph/`. |
| Joins | Page features joined to SERP labels in `src/features/build_features.py` (page key = `host + path.lower()`). |
| Text representations | TF-IDF over scraped page text in `src/features/content_features.py`. |
| Hyperparameter tuning | RandomizedSearchCV across all four models. |
| Hypothesis testing | Chi-square / t-test on top features by class in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). |

### Dashboard demo (Sec. 9)

| Item | Where |
|------|-------|
| Interactive (not static) | Live URL → scrape → predict + sliders + tabs. |
| Showcases full project | Predict / EDA / Graph / Models / Recommendations / What-if / About — covers every layer of the pipeline. |
| Custom polish | Hand-styled CSS in [`src/dashboard/styles.py`](src/dashboard/styles.py): solid-color dark sidebar, card-framed Plotly charts, pill-style nav, refined metric cards. |
| EDA + modeling visible side-by-side | EDA tab + Models tab both render Plotly charts inline; URL hierarchy network shows the link-graph layer. |
| Live-only Predict tab | No precomputed demo — every score on `Predict` comes from a fresh scrape + featurization for a user-supplied URL; failures surface as explicit errors instead of falling back to fake data. |

### Presentation (Sec. 11)

| Item | Where |
|------|-------|
| 8-10 minute hard window | `presentation/recording.mp4`. |
| Slides as PDF, no code on slides | `presentation/slides.pdf`. |
| Both team members speak with cameras on | Recording. |
| Coverage: objective, data, EDA, modeling, implications, challenges | Slide deck outline in `presentation/`. |

## Reproducing the pipeline end-to-end

```bash
pip install -r requirements.txt
cp .env.example .env                          # add BRAVE_SEARCH_KEY or SERPAPI_KEY
python -m src.scraping.doc_scraper --domain docs.python.org --limit 300   # repeat per domain
python -m src.scraping.serp_client build-queries
python -m src.scraping.serp_client fetch
python -m src.features.build_features
python -m src.graph.build_graph
python -m src.graph.graph_features
python -m src.models.baseline
python -m src.models.tree_models
python -m src.models.boosting
python -m src.models.neural
streamlit run src/dashboard/app.py
```

The full step-by-step with screenshots lives in
[`USER_MANUAL.md`](USER_MANUAL.md).

## License & ethics

Public, legal data only. The crawler respects `robots.txt` per domain and identifies
itself with a contact email in the User-Agent header. SERP rankings are fetched
through public commercial APIs (Brave Search, SerpApi) within their free-tier ToS.
No login walls, no paywalls, no PII scraped. The full crawler policy lives in
[`data/README.md`](data/README.md).
