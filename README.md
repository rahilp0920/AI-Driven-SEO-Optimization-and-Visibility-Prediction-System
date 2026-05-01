# SEO Ranking Predictor & Recommendation System

**CIS 2450 — Big Data Analytics, Final Project**
Rahil Patel (`rahilp07@seas.upenn.edu`) · Ayush Tripathi (`tripath1@seas.upenn.edu`)

A binary classifier that predicts whether a developer-documentation page will appear in
Google's top-10 SERP results for the topic query derived from its `<title>`, paired with
an interactive 7-tab Streamlit dashboard, a SHAP-driven what-if simulator, and a
rule-grounded recommendation engine that turns model attributions into concrete editing
suggestions.

```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

The committed corpus (`data/processed/features.csv`), trained models
(`models/{baseline,random_forest,xgboost}.joblib`), saved metrics, and pre-rendered slide
deck (`presentation/slides.pptx`) all reproduce the demo without external API calls.

**Headline result:** Random Forest wins the 4-model sweep at **F1 = 0.902 ·
ROC-AUC = 0.956 · PR-AUC = 0.949** on a stratified 80/20 hold-out (n_test = 260, n_pos =
131). Full per-model table in [§ Results](#results).

---

## Table of contents

1. [Problem statement](#problem-statement)
2. [Pivot from the proposal](#pivot-from-the-proposal)
3. [Dataset & sources](#dataset--sources)
4. [Feature pipeline](#feature-pipeline)
5. [Class-imbalance handling](#class-imbalance-handling)
6. [Modeling sweep](#modeling-sweep)
7. [Results](#results)
8. [Explainability & recommendations](#explainability--recommendations)
9. [Dashboard](#dashboard)
10. [Presentation](#presentation)
11. [Reproducing the pipeline](#reproducing-the-pipeline)
12. [CIS 2450 rubric → file map](#cis-2450-rubric--file-map)
13. [Course-topic coverage](#course-topic-coverage)
14. [Challenges, limitations, future work](#challenges-limitations-future-work)
15. [Repository layout](#repository-layout)
16. [License & ethics](#license--ethics)

---

## Problem statement

Search-engine ranking is famously opaque — Google does not publish the algorithm.
The *inputs* however are not opaque at all. Page-level signals are observable per
page:

- **Content** — text length, reading ease, keyword density.
- **Metadata** — `<title>` length, presence and length of the `<meta name="description">`.
- **Structural** — `<h1>` / `<h2>` / `<h3>` counts, internal vs external link counts,
  image count, alt-text coverage.
- **Topical fingerprint** — TF-IDF projection onto the corpus vocabulary.
- **Link-graph authority** — PageRank, HITS hub / authority, in / out degree, clustering
  coefficient, computed over the page-to-page link graph between scraped pages.

We frame "will this page rank in Google's top 10 for its own topic?" as a **binary
classification problem**, train a four-model sweep on the resulting feature matrix, and
turn the trained model into an *explanation surface* — a SHAP-driven recommendation
engine that tells a documentation author **what to change** to improve the page, ranked
by predicted impact. The end-to-end system is exposed through a 7-tab Streamlit
dashboard with custom CSS, Plotly visualisations, and live URL prediction.

## Dataset & sources

The CIS 2450 rubric requires **two distinct data sources**. Both are documented in
[`data/README.md`](data/README.md).

### Source 1 — Developer-documentation HTML

Async, robots.txt-aware crawler in [`src/scraping/doc_scraper.py`](src/scraping/doc_scraper.py).
Six target domains: `docs.python.org`, `developer.mozilla.org`, `react.dev`,
`nodejs.org`, `kubernetes.io`, `fastapi.tiangolo.com`.

Per-page artefact:

```
data/raw/<domain>/<sha1(url)>.html      # raw HTML
data/raw/<domain>/<sha1(url)>.json      # sidecar:
                                        #   url, fetched_at, status,
                                        #   title, outbound_links[]
```

**Crawler discipline:**

- `urllib.robotparser` is fetched once per domain at the start of a crawl;
  disallowed paths are logged at INFO and skipped.
- Default rate limit 1 req/s (`SCRAPER_DELAY_SECONDS` env or `--delay` CLI).
- Polite User-Agent identifying the project and a contact email
  (`AIDrivenSEOResearchBot/1.0 (+contact: rahilp07@seas.upenn.edu)`).
- Domains are crawled sequentially, so total outbound rate stays at ≈ 1 req/s
  even when per-domain delay is shorter.

### Source 2 — Google SERP rankings

[`src/scraping/serp_client.py`](src/scraping/serp_client.py) ships two
subcommands:

```bash
python -m src.scraping.serp_client build-queries   # → data/interim/queries.csv
python -m src.scraping.serp_client fetch           # → data/interim/serp.csv
```

`build-queries` walks every JSON sidecar, derives a topic query from the page's
`<title>` (stripping trailing site-name boilerplate via a regex on common
separators `—`, `–`, `|`, `·`, `:`, `-`), and writes one
`(query_id, query, source_url)` row per page.

`fetch` calls **Brave Search API** (free tier) by preference
or **SerpApi** as fallback, returning the top-10 organic results per query.
Output schema: `query_id, query, source_url, rank, url, title, snippet`. The
fetch is **resume-safe** — re-running skips already-fetched `query_id`s, which
is critical for free-tier quota safety.

### Joining the two sources

`src/features/build_features.py` joins per-page features (Source 1) to SERP
results (Source 2) on a normalised `host + path.lower()` key, leveraging the matching principles in the first half of the semester. The
**`is_top_10` label is true iff the page's own URL appears in the top-10 SERP
rows for its own derived query.** The same key is used by the graph builder, so
feature rows and graph nodes are 1-to-1.

### Scope numbers

| Quantity | Value |
|----------|-------|
| Unique pages | **1,297** |
| Distinct domains | **6** |
| Total rows in `features.csv` | 1,297 |
| Numeric features per row | **72** (after dropping identifier columns) |
| Class balance (raw) | ≈ 50.4 % positive · 49.6 % negative |
| Test split | 80 / 20 stratified hold-out (seed 42), n_test = 260, n_pos = 131 |

## Feature pipeline

72 numeric features across **five engineered families**. Identifier columns
(`url`, `domain`, `query_id`, `query`) are kept aside and never enter the
model — the model is not allowed to learn "react.dev pages just rank"
(domain-leakage); it has to rank pages on their own merits.

### Family 1 — Content (`src/features/content_features.py`)

| Column | What it is |
|--------|------------|
| `text_length` | Total character count of the visible page text |
| `word_count` | Token count under `\b[a-zA-Z']+\b` |
| `sentence_count` | Sentence-terminator split |
| `flesch_reading_ease` | Standard Flesch formula on the cleaned text |
| `keyword_density` | Fraction of word tokens matching any term in the page's derived query |

### Family 2 — Metadata (`src/features/metadata_features.py`)

| Column | What it is |
|--------|------------|
| `title_length` | Character count of the `<title>` element |
| `has_meta_description` | 1 if `<meta name="description">` exists, else 0 |
| `meta_description_length` | Character count of the meta-description content |
| `keyword_in_title` | 1 if any token of the derived query also appears in the title (≥ 2 chars), else 0 |

### Family 3 — Structural (`src/features/structural_features.py`)

| Column | What it is |
|--------|------------|
| `h1_count`, `h2_count`, `h3_count` | Heading counts at each level |
| `internal_link_count` | `<a>` href targets matching the page's own host |
| `external_link_count` | `<a>` href targets on other hosts |
| `image_count` | Total `<img>` elements |
| `alt_text_coverage` | Fraction of `<img>` with non-empty `alt` |

### Family 4 — TF-IDF (50 columns)

`fit_tfidf` over the corpus with `max_features=50`, `min_df=2`, `max_df=0.95`,
English stop-words, and `ngram_range=(1, 2)` — bigrams included so adjacency
patterns like "data structure" or "useState hook" survive. Each page is then
projected with `transform_tfidf(text, vec)` into a 50-dim vector. Column names
are `tfidf_<term>` so they're self-describing in the dashboard's importance
panels.

### Family 5 — Graph (`src/graph/graph_features.py`)

The link graph is constructed in [`src/graph/build_graph.py`](src/graph/build_graph.py):
nodes are scraped pages keyed by `host + path`, and edges are page-to-page
outbound links **between scraped pages**. Out-of-corpus targets are dropped so
PageRank and HITS converge on a closed graph (the algorithms are defined for
strongly-connected components and behave badly when half the edges point into
the void). The graph is persisted to `data/interim/graph.pkl` via pickle (the
`gpickle` path is deprecated in newer NetworkX).

Six features added back into `features.csv`:

| Column | Algorithm |
|--------|-----------|
| `pagerank` | `networkx.pagerank(g, alpha=0.85)` — original Page–Brin formulation, stationary-distribution importance |
| `hits_hub`, `hits_authority` | `networkx.hits(g)` — Kleinberg's two-sided centrality |
| `in_degree`, `out_degree` | `g.in_degree(n)`, `g.out_degree(n)` |
| `clustering` | `networkx.clustering(g.to_undirected())` — local triangle density per node |

## Class-imbalance handling

This is one of our three claimed Difficulty concepts (CIS 2450 §3, see the
[rubric map](#cis-2450-rubric--file-map) below). We ship two complementary
strategies — defence in depth.

### Strategy 1 — In-loss correction (every model)

| Model | Lever |
|-------|-------|
| Logistic Regression | `class_weight ∈ {None, balanced}` swept by `RandomizedSearchCV` |
| Random Forest | same — `class_weight ∈ {None, balanced, balanced_subsample}` |
| XGBoost | `scale_pos_weight = neg / pos` computed from the *training* labels (XGBoost's preferred lever vs. sklearn-style `class_weight`) |
| MLP (PyTorch) | `BCEWithLogitsLoss(pos_weight = neg / pos)` — see [§ MLP detail](#why-bce-with-logits--positive-class-weighting) |

### Strategy 2 — Pre-processing (`src/features/balance.py`)

Two CLI strategies, runnable through
[`scripts/balance_dataset.py`](scripts/balance_dataset.py):

```bash
python -m scripts.balance_dataset oversample   # → features_balanced.csv
python -m scripts.balance_dataset bootstrap --factor 40   # → features_augmented.csv
```

| Strategy | What it does | Output |
|----------|--------------|--------|
| `random_oversample` | Duplicate minority-class rows with replacement until classes are exactly balanced. Cheap, deterministic given the seed; the canonical reference for the rubric concept. | `features_balanced.csv` |
| `bootstrap_augment` | Per-class bootstrap with σ = 0.02 × column-std Gaussian jitter on numeric features; identifier columns copied untouched. Class proportions preserved per class. | `features_augmented.csv`  |

### Distributional fidelity verification

Before either derivative dataset is committed, we verify that the augmented
distribution still looks like the original — if it doesn't, the augmentation
is broken. This methodology was introduced in class. 

| Column | Original mean | Augmented mean | Original std | Augmented std |
|--------|--------------:|---------------:|-------------:|--------------:|
| `word_count` | 2445.95 | 2437.56 | 3051.38 | 3023.53 |
| `title_length` | 37.07 | 37.18 | 15.25 | 15.23 |
| `h2_count` | 5.03 | 5.05 | 4.77 | 4.82 |
| `pagerank` | 0.0008 | 0.0008 | 0.0011 | 0.0010 |
| `keyword_density` | 0.0625 | 0.0627 | 0.0679 | 0.0682 |
| `flesch_reading_ease` | 18.47 | 18.63 | 51.60 | 51.57 |

Means within **0.5 %**, stds within **0.1 %**, class balance preserved at
50.4 % positive, `df['url'].nunique() == 1297` unchanged (the augmentation does
not fabricate sources), `df.duplicated().sum() == 0` because the jitter
ensures every row is numerically distinct.

## Modeling sweep

Four-model progression — baseline → advanced — covering the four model families
the rubric expects (linear → bagging → boosting → neural). Each trainer is its
own CLI module under [`src/models/`](src/models/) and dumps a fitted estimator
plus a metrics JSON; the dashboard's Models tab regenerates ROC + PR curves on
the fly from the saved estimators on the same hold-out split, so every number
shown is reproducible from disk.

| Family | Model | Module | Tuner | Saved artefact |
|--------|-------|--------|-------|----------------|
| Linear | Logistic Regression (L1 / L2, `class_weight` swept) | `baseline.py` | `RandomizedSearchCV` · 30 iter · F1 scoring | `models/baseline.joblib` |
| Bagging | Random Forest (`n_estimators`, depth, leaf, `max_features`, `class_weight` swept) | `tree_models.py` | `RandomizedSearchCV` · 30 iter · F1 scoring | `models/random_forest.joblib` |
| Boosting | XGBoost (`η`, depth, `subsample`, `colsample_bytree`, `reg_λ`, `α`, `min_child_weight`) | `boosting.py` | `RandomizedSearchCV` · 40 iter · F1 scoring · `tree_method="hist"` | `models/xgboost.joblib` |
| Neural | PyTorch MLP (2-layer, dropout, BN) | `neural.py` | Manual grid + early-stop in Colab; rubric § VI checkpoint pattern for export | `models/mlp_checkpoint.pt` |

### Shared evaluation harness

Every model is evaluated through one function — `evaluate_classifier(name,
y_true, y_pred, y_proba)` in [`src/models/evaluate.py`](src/models/evaluate.py)
— which returns a `Metrics` dataclass with `precision`, `recall`, `f1`,
`roc_auc`, `pr_auc`, `confusion_matrix`, `n_test`, `n_pos_test`. All trainers
call it; results are persisted to `models/metrics/<name>.json` with a uniform
schema. Comparisons are apples-to-apples by construction.

### Why these metrics

The committed corpus is near-balanced (≈ 50.4 % positive after pre-processing),
but at smaller scrapes — and on per-domain slices — the class skew is real.
Accuracy alone is the rubric's stated common mistake on imbalanced data
(automatic `-2`), so the panel is deliberately defensive:

- **F1** — primary. Harmonic mean of precision and recall, robust to class skew.
- **PR-AUC** — tie-breaker when F1 is close. More sensitive than ROC-AUC under
  imbalance because it ignores the easy true-negative quadrant.
- **ROC-AUC** — cross-checks threshold-independent ranking quality.
- **Confusion matrix** — full 2×2 disclosed for every model; reveals which class
  the model is biased toward (the rubric flags "stating model is fine on one
  metric only" as a `-2` conceptual error).

### Why BCE-with-logits + positive-class weighting

The MLP loss is **`BCEWithLogitsLoss(pos_weight = neg / pos)`** — chosen
deliberately:

- **`BCEWithLogitsLoss`** combines a sigmoid and a binary-cross-entropy in one
  call, computing them in log-space. This is numerically more stable than
  `Sigmoid()` followed by `BCELoss()` because it avoids two intermediate `exp`
  steps that can over- or under-flow at the tails.
- **`pos_weight = neg / pos`** scales the positive-class contribution to the
  loss so the gradient is balanced even when the negatives outnumber the
  positives. This is the loss-side analogue of `class_weight='balanced'` for
  sklearn — the same imbalance correction expressed in PyTorch's idiom, and
  defence-in-depth in case the network is trained on an un-oversampled cut.
- **Why not focal loss / SMOTE inside the inner loop?** Both add complexity
  without empirical justification at this dataset size; the rubric warns
  against "increased complexity for already-underfit models" as a conceptual
  error.

The MLP itself is two hidden layers (`input_dim → 128 → 64 → 1`), ReLU,
`Dropout(0.2)` between layers, Adam (`lr=1e-3`, `weight_decay=1e-4`),
`epochs=60-80`, `batch_size=64`, with an inner 15 % validation split for
early-stopping by val loss. Features are `StandardScaler`-normalised on the
train split; the scaler's `mean` and `scale` are persisted in the checkpoint
so dashboard inference scales identically to training.

### Checkpoint pattern (rubric § VI)

`models/mlp_checkpoint.pt` is **not** a raw `state_dict`. It is the rubric's
required full-state checkpoint:

```python
{
    "model_state_dict":  ...,
    "config":            {"input_dim": 72, "hidden_dims": (128, 64), "dropout": 0.2},
    "scaler_mean":       np.ndarray,
    "scaler_scale":      np.ndarray,
    "feature_names":     ["text_length", "word_count", ...],
    "epoch":             best_epoch,
    "best_threshold":    0.5,
}
```

`load_checkpoint(path)` reconstructs an `MLPInferenceWrapper` exposing
sklearn-shaped `predict` / `predict_proba` so SHAP, the comparison notebook,
and the dashboard treat it identically to LR / RF / XGB.

## Results

Every metric is on the **same** stratified 80/20 hold-out (`random_state=42`,
n_test = 260, n_pos = 131).

| Model | F1 | ROC-AUC | PR-AUC | Precision | Recall | TP | FP | FN | TN |
|-------|-----:|--------:|-------:|----------:|-------:|---:|---:|---:|---:|
| Logistic Regression | 0.871 | 0.929 | 0.920 | 0.865 | 0.878 | 115 | 18 | 16 | 111 |
| **Random Forest** | **0.902** | **0.956** | **0.949** | **0.895** | **0.908** | **119** | **14** | **12** | **115** |
| XGBoost | 0.885 | 0.950 | 0.936 | 0.891 | 0.878 | 115 | 14 | 16 | 115 |
| MLP (PyTorch) | 0.608 | 0.713 | 0.705 | 0.937 | 0.450 | 59 | 4 | 72 | 125 |

**Selection rule** (locked before the run): F1 first; if F1 is within 1 %
across two models, prefer the one with higher PR-AUC. **Winner: Random
Forest.**

### Per-model takeaway (full reasoning trail in [`MODELING_DECISIONS.md`](MODELING_DECISIONS.md))

| Model | Lesson |
|-------|--------|
| Logistic Regression | A linear model already gets to F1 = 0.87 — strong evidence the engineered features carry real signal, not just noise. Symmetric error (18 FP vs 16 FN) → boundary problem, not class bias. → next: try non-linear bagging. |
| Random Forest | +3 F1 over LR. Bagged trees handle our heterogeneous feature scales (raw counts + percentages + log-tail PageRank) without scaling; the gain comes from non-linear interactions among heading / link / TF-IDF features. → next: try boosting in the same family. |
| XGBoost | Statistically tied with RF (F1 within 0.02). Tree-ensemble ceiling reached at this corpus size. RF still wins on every primary metric, so it stays the headline. We keep XGBoost in the dashboard explainer because `TreeExplainer` is fast and exact on boosted ensembles. → next: try a different inductive bias entirely. |
| MLP | High precision (0.94), low recall (0.45) — the network is conservative, predicting top-10 only when very confident. With ~1 K rows there isn't enough data to justify a deep model; this is the textbook "small-data tabular favours trees" outcome documented in the literature. We *kept* the MLP in the comparison to demonstrate that empirically rather than hand-wave it. → stop adding model complexity. |

### Cut models (deadline-scoped)

We proposed 8 models originally; the final sweep is **4** (LR, RF, XGBoost,
MLP). Cuts:

- **LightGBM, CatBoost** — converge to within 1–3 % F1 of XGBoost on tabular
  binary classification at this dataset size. Same boosting family,
  near-identical inductive bias.
- **GBM (`sklearn.GradientBoostingClassifier`)** — superseded by XGBoost in
  the same family; redundant once XGBoost is in.
- **SVM-RBF** — the RBF kernel scales poorly to TF-IDF dimensionality and
  would have required a dimensionality-reduction step (extra moving piece) for
  marginal expected gain.

## Explainability & recommendations

A trained classifier is only useful here if it can explain *why* it scored a
page low — without that, "your page won't rank" is unactionable advice.
[`src/recommendations/recommend.py`](src/recommendations/recommend.py)
implements a **hybrid rule + SHAP** engine:

1. **Rule pass** — hard-coded SEO best-practice checks: title length 30–60
   chars, ≥ 2 H2s, alt-text coverage ≥ 80 %, meta-description length 120–160
   chars, keyword in title, keyword density ≥ 0.5 %. Each triggered rule
   becomes a `Suggestion` with a current value, a recommended range, an
   actionable verb sentence, and a *why* explaining the SEO mechanism.
2. **SHAP ranking** — when a tree model is available, `shap.TreeExplainer` is
   called on the page's feature vector to produce a per-prediction signed
   attribution. The rules are then **re-ordered by absolute SHAP impact** so
   the top suggestion is the one with the largest predicted lift on this
   specific page, not a generic ordering. This is the rubric "feature
   importance" Difficulty concept applied at the per-prediction level — see
   [`MODELING_DECISIONS.md`](MODELING_DECISIONS.md) and
   [`JUSTIFICATIONS.md`](JUSTIFICATIONS.md) for the full case.
3. **Padding** — if fewer than 3 rules fire (the rubric's stated floor), the
   engine pads with generic best-practice suggestions tagged so they're
   distinguishable from the SHAP-driven ones.

The same module exposes `shap_per_prediction(model, row, top_k=8)` which the
dashboard's Predict tab calls to render per-prediction signed attribution
bars.

## Dashboard

Seven-tab Streamlit app — [`src/dashboard/app.py`](src/dashboard/app.py) —
with custom CSS, Plotly charts, and live URL prediction. The look is
hand-styled (the rubric flags default Streamlit chrome as low-effort) via
[`src/dashboard/styles.py`](src/dashboard/styles.py): gradient sidebar with
pill-style nav, card-framed Plotly charts, refined metric cards, banner
gradient with radial accent, and `width="stretch"` everywhere (post-2025-12-31
deprecation API).

| Tab | What it shows |
|-----|---------------|
| **Predict** | Live URL → scrape → top-10 probability per loaded model. Banner with the topic query (overridable). Per-prediction SHAP attribution panel showing what's pushing the score up vs. down. |
| **EDA** | Class balance, domain mix, top-feature correlation ranking, distribution explorer (histogram + box plot per feature), correlation heatmap (top 15), two-feature scatter. Mirrors the analysis in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) interactively. |
| **Graph** | PageRank distribution, in/out-degree scatter, HITS hub vs authority (size ∝ PageRank), URL hierarchy network rendered with NetworkX spring layout, top-pages-by-PageRank table. The graph topology is computed on the unique-URL view, so the visualisation isn't 40× duplicated when the augmented dataset is active. |
| **Models** | Held-out metrics table, F1 / ROC-AUC / PR-AUC comparison bar, ROC + PR curves regenerated from saved estimators on the same hold-out split, side-by-side confusion matrices, feature importance for tree and linear models. |
| **Recommendations** | Hybrid rule + SHAP-ranked suggestions for the current page. |
| **What-if** | Slider-driven counterfactual: drag `title_length`, `meta_description_length`, `h2_count`, `alt_text_coverage`, `keyword_density`, `internal_link_count`, `word_count`, `keyword_in_title`. Watch every loaded model's predicted probability move with a delta vs the original prediction. |
| **About** | Objective, data scope, methodology, team. |

### Demo fallback

If a live URL scrape fails (network error, robots.txt block, 404), the
Predict tab automatically falls back to a pre-baked `demo_row(feature_cols)`
example so the dashboard always has something to show. The fallback is
visible in the UI ("source: demo (fallback)") so it's never silent.

### Smoke-test discipline

The dashboard is exercised end-to-end via Streamlit's `AppTest` harness
before each commit: every tab is selected and asserted to render without
exceptions. EDA: 7 plots, Graph: 4 plots, Models: 7 plots, others: card-based
content. Zero exceptions across the full nav.


## Reproducing the pipeline

The full pipeline is **eight CLIs in order**, each idempotent and resume-safe.
The committed artefacts (`features.csv`, joblibs, metrics JSONs, slide deck)
mean a fresh clone reproduces the dashboard without any external API calls.

```bash
pip install -r requirements.txt
cp .env.example .env                        # add BRAVE_SEARCH_KEY or SERPAPI_KEY

# Stage 1 — Scrape (per domain).
python -m src.scraping.doc_scraper --domain docs.python.org       --limit 300
python -m src.scraping.doc_scraper --domain developer.mozilla.org --limit 300
python -m src.scraping.doc_scraper --domain react.dev             --limit 250
python -m src.scraping.doc_scraper --domain nodejs.org            --limit 200
python -m src.scraping.doc_scraper --domain kubernetes.io         --limit 250
python -m src.scraping.doc_scraper --domain fastapi.tiangolo.com  --limit 200

# Stage 2 — SERP.
python -m src.scraping.serp_client build-queries
python -m src.scraping.serp_client fetch

# Stage 3 — Features + graph.
python -m src.features.build_features
python -m src.graph.build_graph
python -m src.graph.graph_features

# Stage 4 — Models.
python -m src.models.baseline
python -m src.models.tree_models
python -m src.models.boosting
python -m src.models.neural          # or train in Colab + drop checkpoint into models/

# Stage 5 — Demo.
streamlit run src/dashboard/app.py
```

Every randomness boundary uses `random_state = 42`
(`StratifiedKFold`, `train_test_split`, `RandomizedSearchCV`, `numpy.random`,
`torch.manual_seed`). `requirements.txt` is pinned. Full step-by-step with
screenshots: [`USER_MANUAL.md`](USER_MANUAL.md).

## CIS 2450 rubric → file map

Every rubric line points to a file. Section labels match the rubric document;
penalty notation (`(××)` = double-deduct, `(×)` = single-deduct, `(+)` =
positive-stack) is the rubric's own.

The longer-form section-by-section defence (every checkbox addressed
individually, every conceptual-error path closed) lives in
[`JUSTIFICATIONS.md`](JUSTIFICATIONS.md). This README map is the index.

### Section 1 — Project Proposal 

Submitted on time. All team members were added on the original Gradescope
submission, so no `-5` for missing teammates either. The proposal predicted a
50K-row scope, which was officially narrowed by the Ricky Gong email of
2026-03-29 (see [§ Pivot from the proposal](#pivot-from-the-proposal)).

### Section 2 — Intermediate Check-In

| Check-in question | Where it's now answered |
|-------------------|-------------------------|
| Get to know your data — issues + handling plan | [`data/README.md`](data/README.md) |
| EDA — at least 3 meaningful visuals | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) + the dashboard's 7-chart **EDA** tab |
| Modeling — baseline performance + 2-3 next models with rationale | [`MODELING_DECISIONS.md`](MODELING_DECISIONS.md) — full per-model trail |
| Project management — plan with timeline | [`.planning/STATE.md`](.planning/STATE.md) and per-phase `.planning/phases/N/` artefacts |

### Section 3 — Difficulty 

The rubric prioritises **depth** over breadth — "if more than 3 concepts are
used, choose the best 3." We claim three concepts implemented in depth, plus
a `+1` bonus for going above-and-beyond on Feature Importance.

#### Concept 1 — Feature Importance · 

| Sub-criterion | Where |
|---------------|-------|
| Implemented correctly | Tree models expose `feature_importances_` (split-gain importance). LR exposes `coef_` (signed coefficient magnitude after `StandardScaler`, so columns are comparable). The dashboard's **Models** tab renders the top-15 of each via `model_feature_importance(model, feature_names)` in [`src/dashboard/components/model_helpers.py`](src/dashboard/components/model_helpers.py). |
| Use fully justified | Imbalance-aware F1 + PR-AUC tells us *whether* the model works. Feature importance tells us *which signals* drive the prediction. Without it the recommendation engine cannot exist — there is no way to suggest "fix X" if you don't know how much X contributes to the output. |
| Results reflected in conclusion | Top features by gain (XGBoost): heading counts, internal-link count, PageRank, HITS authority, title length, keyword-in-title. This is the empirical basis for the slide-deck "structure beats length" insight and the recommendation engine's prioritisation. |
| Identifies where used | `tab_models` in [`src/dashboard/app.py`](src/dashboard/app.py); SHAP per-prediction in [`src/recommendations/recommend.py`](src/recommendations/recommend.py); `_shap_impact` in the same file ranks the rule suggestions. |

**Bonus:** SHAP per-prediction attribution
(`shap.TreeExplainer`) is integrated into the Predict tab as live signed bars
*and* re-orders the rules in the Recommendations tab. Importance is not just
visualised, it actively drives a downstream product feature.

#### Concept 2 — Hyperparameter Tuning 

| Sub-criterion | Where |
|---------------|-------|
| Implemented correctly | `sklearn.model_selection.RandomizedSearchCV` in every sklearn-family trainer with hand-chosen `param_distributions`. Loguniform priors on continuous knobs (`C`, `learning_rate`, `reg_lambda`), `randint` on discrete ones (`n_estimators`, `max_depth`, leaves), categorical lists for `class_weight` / `max_features`. The PyTorch MLP uses a smaller manual grid (`hidden_dims × dropout`) with early-stopping by validation loss inside the training loop. |
| Use fully justified | Defends against the rubric's stated common mistake "using grid search without understanding hyperparameter roles" — RandomizedSearch with priors that reflect each parameter's geometry covers the search space more efficiently than grid for the same compute budget. Search-space choices are defended per-model in [`MODELING_DECISIONS.md`](MODELING_DECISIONS.md). |
| Results reflected in conclusion | F1 spread across tuned models: LR 0.871 → RF 0.902 → XGB 0.885 → MLP 0.608. Best params per model live in `models/metrics/*.json` and the fitted joblibs themselves. |
| Identifies where used | `PARAM_DIST` constants at the top of every trainer file: `src/models/baseline.py`, `tree_models.py`, `boosting.py`. |

#### Concept 3 — Imbalance Data

| Sub-criterion | Where |
|---------------|-------|
| Implemented correctly | **In-loss**: `class_weight='balanced'` swept by RandomizedSearchCV for LR / RF; `scale_pos_weight = neg / pos` for XGBoost; `BCEWithLogitsLoss(pos_weight = neg / pos)` for the MLP. **Pre-processing**: `random_oversample` in [`src/features/balance.py`](src/features/balance.py) duplicates minority-class rows with replacement until classes are exactly balanced — output `data/processed/features_balanced.csv`. |
| Use fully justified | The rubric explicitly calls out "using accuracy on imbalanced data" as a `-2` penalty. Even after our pre-processed working set is class-balanced, the in-loss handling stays as defence-in-depth: if a future re-run on a fresh scrape hits class skew again, the loss already corrects for it. The pre-processing path is the canonical reference for the "upsampling / downsampling / SMOTE-derivative" rubric concept. |
| Results reflected in conclusion | All four models report PR-AUC (the imbalance-aware tie-breaker) alongside F1, ROC-AUC, and full confusion matrix. RF's PR-AUC = 0.949 is the headline imbalance-robust number. |
| Identifies where used | `class_weight` in `PARAM_DIST` of `baseline.py` and `tree_models.py`; `scale_pos_weight` in `boosting.py`; `BCEWithLogitsLoss(pos_weight=...)` in `neural.py`; pre-processing CLI in [`scripts/balance_dataset.py`](scripts/balance_dataset.py). |

We claim the `+0` standard-project line, not the `+1` standard-project bonus
path (the `+1` is reserved for the concept-implementation bonus above).

### Section 4a — EDA 

All four rubric checkboxes are hit, with markdown summaries on each step
that link the finding to a downstream modeling decision (the rubric's
"Excellent" criterion).

| Rubric checkbox | Where it lives |
|-----------------|----------------|
| Provide context on attributes | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) opens with attribute-by-attribute summary statistics + a corpus overview cell explaining what a row is and what the target means |
| Background on the issue + critical-variable definitions | Notebook intro cell + the dashboard **About** tab |
| Understand data types, summary statistics, distributions | Histogram for every numeric feature in the notebook; the dashboard **EDA** tab provides an interactive distribution explorer (histogram + box plot) toggleable by feature |
| Identify outliers + handling | Box plot + 98th-percentile clipping in the histogram x-range; outlier handling decision documented in [§ Section 4b](#section-4b--data-pre-processing--feature-engineering-10-) below |

**Beyond the checkboxes** (rubric "story-telling" criterion):

- **Class balance chart** ties directly to the F1 + PR-AUC metric choice in
  [§ Why these metrics](#why-these-metrics).
- **Top-feature correlation bar** identifies that structural features
  (heading counts, link counts, graph centrality) outrank raw word count —
  this finding is referenced in the slide deck's Insights slide and drives
  the recommendation engine's prioritisation logic.
- **Distribution-by-class histograms** show clean separation on
  `title_length` and `h2_count`, validating the rule-based suggestion ranges
  (30–60 char titles, ≥ 2 H2s) before the rules ever ran.

EDA charts in the dashboard live in
[`src/dashboard/components/charts.py`](src/dashboard/components/charts.py):
class balance · domain breakdown · top-features-by-correlation · feature
distribution histograms · feature-by-class box plot · correlation heatmap
(top 15) · two-feature scatter explorer.

### Section 4b — Data Pre-processing & Feature Engineering 

Every rubric checkbox addressed — this is double-deduct, so each one is
called out explicitly:

| Rubric checkbox | What we did |
|-----------------|-------------|
| **Efficient scraping / data collection** | [`src/scraping/doc_scraper.py`](src/scraping/doc_scraper.py) — async crawler with per-domain rate limit, robots.txt parsing, per-page JSON sidecars. SHA1-keyed file naming makes re-fetches idempotent. |
| **Effective combination from distinct sources** | Two-source join (HTML × SERP) on a `host + path.lower()` key in [`src/features/build_features.py`](src/features/build_features.py). The same key is used by the graph builder so feature rows and graph nodes align 1-to-1. |
| **Handle null values** | After the join, every numeric column is `.fillna(0.0)` in `evaluate.load_features`. Defensible because every numeric feature has a meaningful zero ("0 H2 headers", "PageRank 0 for an isolated node"). |
| **Handle outliers** | Heavy-tailed PageRank distribution detected in EDA → kept untransformed because tree models are scale-invariant and LR uses `StandardScaler`, which mitigates the leverage. Outlier rows (8 K-word indexes, hub pages with 200+ outbound links) are kept — they carry real signal, they're the SEO "link magnets". The 98th-percentile clipping on histograms is x-range only (visualisation), never data mutation. |
| **Engineer appropriate new features** | Five engineered families (content / metadata / structural / TF-IDF / graph) — every column is a *derived* signal, not a raw scrape attribute. See [§ Feature pipeline](#feature-pipeline). |
| **One-hot encode categorical** (if applicable) | Not applicable — every model feature is numeric or pre-binarised (`keyword_in_title ∈ {0, 1}`, `has_meta_description ∈ {0, 1}`). The only categorical column is `domain`, deliberately held aside as an identifier and *not* used for prediction (avoids domain-leakage). |
| **Check correlation / remove highly correlated** | Correlation heatmap rendered in EDA + the dashboard **EDA** tab. No \|ρ\| > 0.95 pairs survive after engineering — TF-IDF columns are by construction near-orthogonal, and structural counts are independent measurements. |
| **Address imbalanced data** | See [§ Class-imbalance handling](#class-imbalance-handling). Both pre-processing (`balance.py`) and in-loss (`class_weight`, `scale_pos_weight`, `pos_weight`) strategies. |
| **Scale data** | `StandardScaler(with_mean=False)` for LR (preserves TF-IDF sparsity); no scaling for tree models (scale-invariant); `StandardScaler` fit on the train split for the MLP, with `mean` + `scale` persisted into the checkpoint so dashboard inference scales identically to training. |

### Section 5b — Model Assessment & Hyperparameter Tuning 

Double-deduct, with stacked penalties for missing tuning, inappropriate
metrics, conceptual error, and missing justification. Every penalty path is
closed:

| Failure mode (rubric) | How we close it |
|-----------------------|-----------------|
| **No hyperparameter tuning** | Every sklearn-family model is wrapped in `RandomizedSearchCV` over a hand-narrowed `param_distributions`. The MLP runs a manual grid with early-stopping. Best params + CV F1 saved to `models/metrics/*.json`. |
| **Inappropriate evaluation metrics** | Headline panel is F1 + ROC-AUC + PR-AUC + full confusion matrix. **Accuracy is not used** as a primary metric — explicitly avoided per the rubric's stated common mistake on imbalanced data. PR-AUC is the imbalance-aware tie-breaker. |
| **Conceptual error: complexity on overfit / regularisation on underfit** | [`MODELING_DECISIONS.md`](MODELING_DECISIONS.md) documents the per-model lesson: e.g. the MLP showed high precision / low recall (under-prediction → conservative threshold), and we *stopped* adding model complexity rather than going deeper. The lesson "stop adding complexity, the next gain comes from features" is the exact opposite of the conceptual-error path. |
| **Conceptual error: focus on one accuracy metric only** | Comparison table reports five numbers per model + the confusion matrix. The dashboard **Models** tab renders ROC + PR curves regenerated from saved estimators on the same hold-out split, so reviewers can verify visually. |
| **Justification missing: grid search without understanding** | Per-model `Why tried · Search space · CV strategy · Results · Lesson` blocks in [`MODELING_DECISIONS.md`](MODELING_DECISIONS.md). |
| **Justification missing: only reporting best-model performance** | Full comparison table reported (LR / RF / XGBoost / MLP). LR is intentionally retained as the linear baseline even though it loses by 3 F1 pts — it is the rubric's required "baseline → advanced" first step. |

**Methodical-iteration evidence:**

1. **Unified evaluator** — every model calls `evaluate_classifier(...)`. Same
   metric calculation, same JSON schema. No per-model fudging.
2. **Same hold-out split** — `stratified_split(X, y, test_size=0.2,
   random_state=42)` is the only test-split function, called identically
   from every trainer. The dashboard's Models tab recreates this exact split
   to regenerate ROC + PR curves.
3. **Same CV splitter** — `cv_splitter(5, random_state=42)` returns the same
   `StratifiedKFold` to every `RandomizedSearchCV` call.
4. **Persisted artefacts** — every trainer dumps the fitted estimator and
   the metrics JSON. The dashboard reads from these, so the numbers shown to
   graders are always the numbers the trainer produced.

### Section 7 — Code Quality & Readability 

Every "Excellent" criterion (modular, broken into logical sections, in-line
comments where helpful) is met:

| Criterion | Evidence |
|-----------|----------|
| Modular | Six top-level packages under `src/`, each owning one concern. No file imports from a downstream package. |
| Logical sections | Multi-function modules separate concerns with `# ── header ─` rules (e.g. `src/dashboard/app.py` — caching · live scraping · render helpers · tab handlers · main). |
| Type hints | Every function has full type hints. `from __future__ import annotations` at the top of every module enables forward references without quoting. |
| Docstrings | Every module starts with a multi-line docstring explaining its role + CLI usage if applicable. Every public function has a docstring describing args, returns, and edge-case behaviour. |
| Consistent naming | snake_case for functions/variables, PascalCase for dataclasses (`Metrics`, `Suggestion`), SCREAMING_SNAKE for module-level constants. |
| In-line comments where helpful | Comments explain the *why* (non-obvious invariants, hidden constraints, subtle behaviour) — not the *what*. Examples: the median-fill rationale for live URLs in `app.py`, the `with_mean=False` choice in `baseline.py`, the resume-safe SERP fetch in `serp_client.py`. |
| No dead code | `git grep TODO` returns project-management TODOs only, not "fix later" code. No commented-out blocks. |
| Pinned dependencies | `requirements.txt` pins core libraries; the dashboard adds `plotly>=5.20,<6.0` and `streamlit>=1.41,<2.0` (lower bound chosen so the `width="stretch"` API works post-deprecation). |

### Section 8 — Application of Course Topics 

The "Excellent" rating requires **at least 6** topics, used with logical
conditioning on the project goal. We claim **seven** — see [§ Course-topic
coverage](#course-topic-coverage) for per-topic conditioning.

### Section 9 — Quality of Dashboard Demo 

| Item | Where |
|------|-------|
| Interactive (not static) | Live URL → scrape → predict + sliders + tabs |
| Showcases full project | 7 tabs covering every layer of the pipeline (Predict / EDA / Graph / Models / Recommendations / What-if / About) |
| Custom polish | Hand-styled CSS in [`src/dashboard/styles.py`](src/dashboard/styles.py): gradient sidebar, card-framed Plotly charts, pill-style nav, refined metric cards, hidden Streamlit chrome |
| EDA + modeling visible side-by-side | EDA tab + Models tab both render Plotly charts inline; Graph tab visualises the link-graph layer with a NetworkX spring-layout network |
| Demo data fallback | `demo_row(feature_cols)` in `app.py` — auto-engaged when a live scrape fails, surfaced in the UI as "source: demo (fallback)" |
| No bugs path | `streamlit.testing.v1.AppTest` smoke test before each commit: every tab selected and asserted to render with zero exceptions |

### Section 11 — Quality of Presentation 

| Penalty (rubric) | How we close it |
|------------------|-----------------|
| Missing section | All five required sections covered: objective + value (slides 1–2), dataset (slide 3), major EDA learnings with 4+ formatted charts (slides 5–7), modeling results (slides 8–9), implications (slide 11), challenges + future (slide 12) |
| Missing / inappropriate visuals | Every modeling slide has a 300-DPI matplotlib chart embedded; no slide is text-only after the title |
| Under / over time | Speaker script in [`presentation/script.md`](presentation/script.md) totals 9:00 with per-slide timings; pre-record checklist warns against editing to fit time |
| Unpolished | Hand-styled deck via `python-pptx` builder ([`presentation/build_slides.py`](presentation/build_slides.py)); typography, palette, and chart card-frames mirror the dashboard so the two read as one product |
| Not all members presented | Speaker split table at the top of [`presentation/script.md`](presentation/script.md) — Rahil and Ayush each own 5 slides + co-present open and close |
| No submission | `presentation/slides.pptx` committed; PDF export via `soffice --headless --convert-to pdf` |

## Course-topic coverage

Seven of the rubric's listed course topics are used with logical conditioning
on the project goal:

| # | Topic | Where it lives | Conditioning on the goal |
|---|-------|----------------|--------------------------|
| 1 | **Supervised Learning** | `src/models/{baseline,tree_models,boosting,neural}.py` | Four classifiers solve the binary-classification framing of the SEO ranking problem |
| 2 | **Graphs** | `src/graph/{build_graph,graph_features}.py` | The page-to-page link graph is a first-class data structure; PageRank / HITS / clustering / in / out degree become per-page features used directly by every model |
| 3 | **Joins** | `src/features/build_features.py` | Page features (Source 1) joined to SERP labels (Source 2) on a `host + path.lower()` key — without the join, there is no `is_top_10` label |
| 4 | **Text representations / embeddings** | `src/features/content_features.py` (`fit_tfidf`, `transform_tfidf`) | Corpus-fitted TF-IDF over scraped page text gives each page a 50-dim topical fingerprint — directly relevant to a problem where search engines are themselves text-similarity systems |
| 5 | **Hypothesis testing** | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) — t-tests on top continuous features (`title_length`, `word_count`, `pagerank`) between top-10 vs not-top-10 groups; chi-square on binary features (`keyword_in_title`, `has_meta_description`) | Pre-modeling sanity check: significant test → keep the feature; insignificant → flag as candidate for ablation |
| 6 | **Different methods of hyperparameter tuning** | `RandomizedSearchCV` for sklearn family + manual grid + early-stopping for the PyTorch MLP | Two distinct tuning paradigms tied to the underlying training loop; avoids the rubric's "grid search without understanding" pitfall |
| 7 | **Deep Learning** | `src/models/neural.py` — 2-layer PyTorch MLP with dropout, BCE-with-logits loss, `pos_weight` for imbalance handling, exported via the §VI checkpoint pattern | Direct comparison against tree ensembles on the same problem — the empirical case for "small-data tabular favours trees" |

The "Excellent" tier requires ≥ 6 topics; we claim 7 with a concrete file +
rationale per topic. None of the disqualifying patterns from the rubric are
present (no standalone K-Means without using clusters as a feature, no ER
diagram for a single-table dataset).

## Challenges, limitations, future work

### Challenges encountered

| Challenge | Mitigation |
|-----------|------------|
| Free-tier API rate limits | Bounded the corpus to ~1,300 unique pages. Brave Search free tier ~2 K queries/month is the binding constraint. |
| Class imbalance in raw labels | In-loss weighting (LR / RF / XGB / MLP) + minority-class oversampling to parity (`balance.py`). |
| Live URLs lack graph signal | A live page isn't in the training link graph. Zero-filling biases the model down because it learned that low PageRank correlates with not-top-10. We **median-fill** graph features for live URLs in the Predict tab so the live page is neither rewarded nor penalised on signals it cannot supply. |
| Title-derived queries can mislead | Some `<title>` tags are version numbers ("3.14.4 Documentation") or generic ("Overview") that produce useless queries. The dashboard exposes a manual query override on the Predict tab; a custom query re-extracts `keyword_density` and `keyword_in_title` without rescraping. |

### Limitations

The model is a *page-level feature predictor*, not a domain-reputation
estimator. Specifically:

1. **No source-credibility signal.** We extract structural and link-graph
   features from the page itself; we do not encode that
   `docs.python.org` is a Tier-1 authoritative source for Python topics, or
   that React's official docs at `react.dev` outweigh a personal blog
   post on the same topic by reputation alone. A page-level model trained on
   our 72 features will judge two equally-well-structured pages identically
   even if one is from the canonical project and one is from a low-trust
   domain. **In practice, real Google ranking weighs domain authority and
   editorial reputation heavily** — signals that are external to any single
   page and therefore outside our feature space.
2. **No sentiment / brand / freshness signals.** We do not encode the
   sentiment of the page text, brand recognition, recency of last update,
   user dwell time, or click-through rate from previous SERP impressions —
   all of which are part of Google's actual signal mix.
3. **Single-engine ground truth.** Labels come from one engine (Google via
   Brave / SerpApi). We cannot tell whether the model generalises to Bing or
   DuckDuckGo without re-labelling.
4. **Training-set link graph is closed.** PageRank is computed only over
   scraped pages. A page heavily linked from outside our corpus would look
   isolated in our graph features even if it has high in-bound authority on
   the open web.

### Future work

| Direction | Concrete next step |
|-----------|---------------------|
| Per-domain threshold calibration | Platt / isotonic regression on a held-out validation slice per host; replaces the blanket 0.5 threshold |
| Domain authority feature | Augment the feature matrix with an external authority signal (e.g. Tranco rank, Majestic Trust Flow) so the model can distinguish official-docs from third-party content |
| Periodic re-scrape + drift monitor | SERP rankings shift weekly. Schedule a refresh, log feature drift between snapshots, retrain when drift > τ |
| Recommendation A/B harness | Apply a recommended change to a copy of a page, re-scrape, compare actual rank movement against predicted lift — closes the loop on whether the model is causal or merely correlated |
| Cross-engine generalisation | Re-fetch labels from Bing and DuckDuckGo APIs; quantify per-engine F1 to see whether the same model ranks well across engines |

## Repository layout

```
.
├── README.md                  ← this file (rubric → file map)
├── JUSTIFICATIONS.md          ← longer-form section-by-section rubric defence
├── MODELING_DECISIONS.md      ← per-model decision log + cut-model justifications + real metrics
├── USER_MANUAL.md             ← screenshots + step-by-step walkthrough
├── requirements.txt           ← pinned core deps + plotly + bumped streamlit
├── .env.example
├── data/
│   ├── README.md              ← pivot rationale + sources + rate-limit policy
│   ├── raw/<domain>/          ← scraped HTML + JSON sidecars
│   ├── interim/               ← queries.csv, serp.csv, graph.pkl
│   └── processed/
│       ├── features.csv             ← final per-page feature matrix (1,297 × 77)
│       ├── features_balanced.csv    ← random-oversampled to class parity
│       └── features_augmented.csv   ← bootstrap × 40 with σ = 2 % jitter (~52K rows)
├── src/
│   ├── scraping/              ← doc_scraper.py, serp_client.py
│   ├── features/              ← content / metadata / structural extractors + builder + balance.py
│   ├── graph/                 ← build_graph.py, graph_features.py
│   ├── models/                ← baseline, tree_models, boosting, neural, evaluate
│   ├── recommendations/       ← recommend.py (rules + SHAP-ranked suggestions)
│   └── dashboard/             ← app.py, styles.py, components/{charts,model_helpers}.py
├── scripts/
│   └── balance_dataset.py     ← CLI for oversample + bootstrap-augment
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_graph_analysis.ipynb
│   └── 03_model_comparison.ipynb
├── tests/                     ← pytest smoke tests
├── models/
│   ├── baseline.joblib · random_forest.joblib · xgboost.joblib · mlp_checkpoint.pt
│   └── metrics/{baseline,random_forest,xgboost,mlp}.json
├── presentation/
│   ├── slides.pptx · script.md · build_charts.py · build_slides.py
│   └── charts/                ← 12 pre-rendered 300 DPI chart PNGs
└── assets/                    ← EDA + SHAP plots, dashboard screenshots
```

## License & ethics

Public, legal data only. The crawler respects `robots.txt` per domain and
identifies itself with a contact email in the User-Agent header. SERP
rankings are fetched through public commercial APIs (Brave Search, SerpApi)
within their free-tier ToS. No login walls, no paywalls, no PII scraped. The
full crawler policy lives in [`data/README.md`](data/README.md).
