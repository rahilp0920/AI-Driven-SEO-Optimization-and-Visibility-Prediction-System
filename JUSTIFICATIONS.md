# Justifications

Implementation walkthrough and CIS 2450 rubric defence for the SEO Ranking
Predictor & Recommendation System. Every rubric line we are graded on is
addressed below with the specific file path, decision rationale, and result.

**Authors:** Rahil Patel · Ayush Tripathi
**Course:** CIS 2450 — Big Data Analytics, Final Project
**Companion docs:** [`README.md`](README.md) · [`MODELING_DECISIONS.md`](MODELING_DECISIONS.md) ·
[`USER_MANUAL.md`](USER_MANUAL.md) · [`data/README.md`](data/README.md)

---

## 0. Executive summary

| Item | Value |
|------|-------|
| Problem framing | Binary classification: will a developer-doc page rank in Google's top-10 for the topic query derived from its `<title>`? |
| Sources | 2 distinct — scraped HTML + Google SERP rankings |
| Corpus | 1,297 unique pages across 6 developer-doc domains |
| Feature matrix | 72 numeric columns across 5 families (content, metadata, structural, TF-IDF, graph) |
| Models | LR · Random Forest · XGBoost · PyTorch MLP — all tuned with RandomizedSearchCV on a shared `StratifiedKFold(5, seed=42)` |
| Headline metric panel | F1 + ROC-AUC + PR-AUC + confusion matrix · PR-AUC is the imbalance-aware tie-breaker |
| Winner | **Random Forest** · F1 = 0.902 · ROC-AUC = 0.956 · PR-AUC = 0.949 |
| Dashboard | 7-tab Streamlit app with custom CSS, Plotly visualisations, live URL prediction, SHAP-driven recommendations |
| Reproducibility | One-line CLI per stage, pinned `requirements.txt`, deterministic seeds everywhere |

---

## 1. Implementation walkthrough

The pipeline is **eight CLIs in order**, each idempotent and resume-safe. Every
stage writes a single artifact under `data/` or `models/`, and every artifact
is versioned in git so a fresh clone reproduces the dashboard without any
external API calls.

### 1.1 Scraping — `src/scraping/doc_scraper.py`

Async, robots.txt-aware crawler over six developer-documentation domains.
Each page is persisted as `data/raw/<domain>/<sha1>.html` with a `<sha1>.json`
sidecar holding `url`, `fetched_at`, `status`, `title`, and `outbound_links`.

Key decisions:
- **`urllib.robotparser` per domain** — `robots.txt` is fetched once at the
  start of every crawl; disallowed paths are logged at INFO and skipped.
- **1 req/s default** (`SCRAPER_DELAY_SECONDS` env var, `--delay` CLI flag) —
  polite to free-tier hosts, conservative against rate limits.
- **Polite User-Agent** identifying the project + a contact email
  (`AIDrivenSEOResearchBot/1.0 (+contact: rahilp07@seas.upenn.edu)`).
- **Sequential domain crawls** — keeps total outbound rate at ~1 req/s overall
  even when per-domain delay is shorter.
- **Out-of-corpus links discarded** before graph construction so PageRank
  converges on a closed graph (see §1.4).

### 1.2 SERP labelling — `src/scraping/serp_client.py`

Two subcommands:

```bash
python -m src.scraping.serp_client build-queries    # → data/interim/queries.csv
python -m src.scraping.serp_client fetch            # → data/interim/serp.csv
```

`build-queries` walks every JSON sidecar, derives a topic query from the
page's `<title>` (strips trailing site-name boilerplate via regex on common
separators `—`, `–`, `|`, `·`, `:`, `-`), and writes one `(query_id, query,
source_url)` row per page.

`fetch` calls Brave Search API (preferred, free tier ~2K req/month) or SerpApi
(fallback) for the top-10 organic results per query. Output:
`data/interim/serp.csv` with `(query_id, query, source_url, rank, url, title,
snippet)`. **Resume-safe**: re-running skips already-fetched `query_id`s by
reading the existing output file, important for free-tier quota safety.

### 1.3 Feature build — `src/features/build_features.py`

Joins five feature families on a `host + path.lower()` page key into the
final `data/processed/features.csv`. The same key is used by the graph
builder, so feature rows and graph nodes are 1-to-1.

The `is_top_10` label is computed at this stage: true iff the page's own URL
appears in the top-10 SERP rows for its own derived query.

### 1.4 Link graph — `src/graph/build_graph.py`, `src/graph/graph_features.py`

`build_graph.py` constructs a `networkx.DiGraph`. Nodes are scraped pages
(keyed by `host + path`); edges are page-to-page outbound links **between
scraped pages**. Out-of-corpus targets are dropped so PageRank and HITS
converge on a closed graph. The graph is persisted to
`data/interim/graph.pkl`.

`graph_features.py` runs three algorithms over the graph and left-joins the
results back into `features.csv`:

- **PageRank** — `networkx.pagerank(g, alpha=0.85)`, the original Page-Brin
  formulation. Stationary-distribution importance per node.
- **HITS** — `networkx.hits(g)`, Kleinberg's hub / authority decomposition.
  Two columns added: `hits_hub`, `hits_authority`.
- **Clustering coefficient** — `networkx.clustering(g.to_undirected())`. Local
  triangle density per node.

Plus `in_degree` and `out_degree` derived directly from the graph.

### 1.5 Class balance — `src/features/balance.py`

Class-imbalance handling is a Difficulty rubric concept (§3, see §4.3 below).
Two strategies are exposed, both runnable via
`scripts/balance_dataset.py`:

1. **`random_oversample`** — duplicate minority-class rows with replacement
   until classes are exactly balanced. Output:
   `data/processed/features_balanced.csv` (1,308 rows, 654 / 654 parity).
2. Augmentation pass for the larger working set — output:
   `data/processed/features_augmented.csv` (~52K rows, class proportions
   preserved). Used as a sanity-check pass alongside the headline 1,297-row
   sweep.

Distributional fidelity is verified before commit — every numeric column's
mean is within 0.5 % of the original and every std within 0.1 %; class
balance is preserved at 50.4 %; URL identity is unchanged
(`df['url'].nunique() == 1297`).

### 1.6 Modeling sweep — `src/models/{baseline,tree_models,boosting,neural,evaluate}.py`

Four-model progression — baseline → advanced — with a shared evaluator and
identical CV strategy. See §4.6 for full justification.

| Model | Module | Tuner |
|-------|--------|-------|
| Logistic Regression (linear) | `baseline.py` | RandomizedSearchCV, 30 iter, F1 scoring |
| Random Forest (bagging) | `tree_models.py` | RandomizedSearchCV, 30 iter, F1 scoring |
| XGBoost (boosting) | `boosting.py` | RandomizedSearchCV, 40 iter, F1 scoring |
| PyTorch MLP (neural) | `neural.py` | Manual sweep, Colab, exported via §VI checkpoint pattern |

Every model writes a fitted joblib (or `.pt` checkpoint) plus a metrics
JSON to `models/metrics/`.

### 1.7 Evaluation — `src/models/evaluate.py`

A single `Metrics` dataclass with `precision`, `recall`, `f1`, `roc_auc`,
`pr_auc`, `confusion_matrix`, `n_test`, `n_pos_test`. Every model in the
sweep calls `evaluate_classifier(name, y_true, y_pred, y_proba)` and writes
the result via `save_metrics`. The shared evaluator means the comparison
table in [`MODELING_DECISIONS.md`](MODELING_DECISIONS.md) is apples-to-apples
by construction.

### 1.8 Recommendations — `src/recommendations/recommend.py`

Hybrid rules + SHAP. The `recommend(features_row, model, query)` function:

1. Runs a hard-coded SEO rule pass (`title_length` 30-60 chars, `≥ 2 H2`,
   `alt_text_coverage ≥ 80 %`, `meta_description_length` 120-160 chars,
   keyword in title, etc.) — every triggered rule becomes a `Suggestion`.
2. If a tree model is available, calls `shap.TreeExplainer` to compute
   per-prediction attribution, and **ranks** the suggestions by absolute
   SHAP impact on the page in question. The order is "biggest predicted
   lift first", not arbitrary.

The same module exposes `shap_per_prediction(model, row, top_k)` which the
dashboard's Predict tab uses for the per-prediction SHAP attribution panel.

### 1.9 Dashboard — `src/dashboard/`

Seven-tab Streamlit app — see §4.9 for the full rubric mapping.

---

## 2. Decisions log — locked before the run

These were locked at the planning stage to keep the run reproducible and
prevent post-hoc justification:

| Decision | Value | Why |
|----------|-------|-----|
| Random seed | `42` everywhere | `StratifiedKFold`, `train_test_split`, `RandomizedSearchCV`, `numpy`, `torch` — single source of randomness |
| Test split | 80 / 20 stratified hold-out | Standard binary-classification split; same split for every model means comparison-table numbers are apples-to-apples |
| CV strategy | `StratifiedKFold(5, shuffle=True, random_state=42)` | Preserves class proportions in every fold; 5 folds is a reasonable trade-off between bias and runtime |
| Tuner | `RandomizedSearchCV` (n_iter ≥ 30) | Faster than grid, defensible against the rubric's "grid search without understanding" warning; loguniform priors reflect actual hyperparameter geometry |
| Primary metric | F1 | Imbalance-aware composite of precision + recall — accuracy alone is the rubric's stated common mistake |
| Tie-breaker | PR-AUC | Better signal than ROC-AUC under class imbalance — places more positives high in the ranked list |
| Decision threshold | 0.5 | Default; future-work item is per-domain calibration (Platt / isotonic) |

---

## 3. CIS 2450 rubric — section-by-section justification

Section labels match the rubric document. Penalty notation is the rubric's
own convention (`(××)` = double-deduct, `(×)` = single-deduct, `(+)` =
positive-stack).

### 3.1 Section 1 — Project Proposal `(+5, ×)`

**Submitted on time.** All team members were added to the original Gradescope
submission, so there is no `-5` deduction for missing teammates either.

The proposal predicted a 50K-row scope, which was officially narrowed by
TA Ricky Gong (email of 2026-03-29) — see §3.4 / §4.5 below. The narrowing is
sanctioned, not a deviation.

### 3.2 Section 2 — Intermediate Check-In `(+5, ××)`

The intermediate check-in covered the four required questions:

| Question | Where the answer lives now |
|----------|----------------------------|
| **Get to know your data** — recognised issues + handling plan | `data/README.md` (sources, robots policy, rate limit) + the `## Pivot rationale` paragraph at the top of [`README.md`](README.md) |
| **EDA** — at least 3 meaningful visuals | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) plus the dashboard's **EDA** tab (7 Plotly charts) |
| **Modeling** — baseline performance + 2-3 next models with rationale | [`MODELING_DECISIONS.md`](MODELING_DECISIONS.md) — full comparison table + per-model "why tried / search space / results / lesson" sections |
| **Project management** — plan with timeline | [`.planning/STATE.md`](.planning/STATE.md) and the per-phase `.planning/phases/N/` artifacts |

### 3.3 Section 3 — Difficulty `(+13)`

The rubric prioritises **depth** over breadth — "if more than 3 concepts are
used, choose the best 3." We claim three concepts implemented in depth, plus
a `+1` bonus for going above-and-beyond on Feature Importance.

#### Concept 1 — Feature Importance (4 / 4)

| Sub-criterion | Where it lives |
|---------------|----------------|
| Implemented correctly | Tree models (`RandomForest`, `XGBoost`) expose `feature_importances_` (split-gain importance). LR exposes `coef_` (signed coefficient magnitude after `StandardScaler` so columns are comparable). The dashboard's **Models** tab renders the top-15 of each via `model_feature_importance(model, feature_names)` in [`src/dashboard/components/model_helpers.py`](src/dashboard/components/model_helpers.py). |
| Use fully justified | Imbalance-aware F1 + PR-AUC tells us *whether* the model works; feature importance tells us *which signals* drive the prediction. Without it the recommendation engine can't be built — there is no way to suggest "fix X" if you don't know how much X contributes to the output. |
| Results reflected in conclusion | Top features by gain (XGBoost): heading counts, internal-link count, PageRank, HITS authority, title length, keyword-in-title. This is the empirical basis for the slide-deck "structure beats length" insight. |
| Identifies where used | [`src/dashboard/app.py`](src/dashboard/app.py) Models tab + [`src/recommendations/recommend.py`](src/recommendations/recommend.py) where SHAP per-prediction attribution drives the suggestion ranking. |

**Bonus +1 — went above and beyond:** SHAP per-prediction attribution
(via `shap.TreeExplainer`) is integrated into the Predict tab as live bars
*and* used to **rank** the rules in the Recommendations tab. Importance is
not just visualised, it actively drives a downstream product feature.

#### Concept 2 — Hyperparameter Tuning (4 / 4)

| Sub-criterion | Where it lives |
|---------------|----------------|
| Implemented correctly | `sklearn.model_selection.RandomizedSearchCV` in every sklearn-family trainer with hand-chosen `param_distributions`. Loguniform priors on continuous knobs (`C`, `learning_rate`, `reg_lambda`), `randint` on discrete (`n_estimators`, `max_depth`, leaves), categorical lists for `class_weight` / `max_features`. PyTorch MLP uses a smaller manual grid (`hidden_dims × dropout`) with early-stopping by validation loss inside the training loop. |
| Use fully justified | Defensible against the rubric's stated common mistake "using grid search without understanding hyperparameter roles" — RandomizedSearch with priors that reflect the parameters' geometry (loguniform for log-scale knobs, integer for tree depths) covers the search space more efficiently than grid for the same compute budget. Documented in [`MODELING_DECISIONS.md`](MODELING_DECISIONS.md) per model. |
| Results reflected in conclusion | Comparison table in §3.7 below shows the F1 spread of tuned models — RF wins by ~3 pts over LR. The best params per model are saved in `models/metrics/*.json` and the fitted joblibs themselves. |
| Identifies where used | `PARAM_DIST` constants at the top of every trainer file: `src/models/baseline.py`, `tree_models.py`, `boosting.py`. |

#### Concept 3 — Imbalance Data (4 / 4)

| Sub-criterion | Where it lives |
|---------------|----------------|
| Implemented correctly | Two complementary strategies. **In-loss handling**: `class_weight='balanced'` swept by RandomizedSearchCV for LR / RF; `scale_pos_weight = neg / pos` for XGBoost; `BCEWithLogitsLoss(pos_weight=neg/pos)` for the MLP. **Pre-processing**: `random_oversample` in [`src/features/balance.py`](src/features/balance.py) duplicates minority-class rows with replacement until classes are exactly balanced — output `data/processed/features_balanced.csv`. |
| Use fully justified | The rubric explicitly calls out "using accuracy on imbalanced data" as a `-2` penalty. Even after our augmented working set is class-balanced, the in-loss handling is kept as defence-in-depth: if a future re-run on a fresh scrape hits class skew again, the loss already corrects for it without changing code. The pre-processing path is the canonical reference for the "upsampling / downsampling / SMOTE-derivative" rubric concept. |
| Results reflected in conclusion | All four models report PR-AUC (the imbalance-aware tie-breaker) alongside F1, ROC-AUC, and full confusion matrix. RF's PR-AUC = 0.949 is the headline imbalance-robust number. |
| Identifies where used | `class_weight` argument visible in `PARAM_DIST` of [`src/models/baseline.py`](src/models/baseline.py) and [`src/models/tree_models.py`](src/models/tree_models.py); `scale_pos_weight` in [`src/models/boosting.py`](src/models/boosting.py); `BCEWithLogitsLoss(pos_weight=...)` in [`src/models/neural.py`](src/models/neural.py); pre-processing CLI in [`scripts/balance_dataset.py`](scripts/balance_dataset.py). |

**Standard-project line:** the project does **not** follow the standard
homework outline — full data pipeline + 4-model sweep + Streamlit
dashboard + SHAP recommendation engine is well beyond a homework's scope.
We claim the `+0` standard-project line, not the `+1` bonus path
(reserved for the concept-implementation bonus above).

---

### 3.4 Section 4a — EDA `(+10, ×)`

The rubric's four checkboxes are all hit, with markdown summaries on each
step that link the finding to a downstream modeling decision (the rubric's
"Excellent" criterion).

| Rubric checkbox | Where it lives |
|-----------------|----------------|
| Provide context on attributes in relation to overall data | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) opens with attribute-by-attribute summary statistics + a corpus overview cell explaining what a row is (one developer-doc page) and what the target means (top-10 in Google for the page's own derived query). |
| Background on the issue statement + high-level critical-variable definitions | Notebook intro cell + the dashboard **About** tab. |
| Understand data types, summary statistics, distributions | Histogram for every numeric feature in the notebook; the dashboard **EDA** tab provides an interactive distribution explorer (histogram + box plot) toggleable by feature. |
| Identify outliers + handling | Box plot + 98th-percentile clipping in the histogram x-range; outlier handling decision documented in §4.5 below. |

**Beyond the checkboxes** (rubric "story-telling" criterion):

- **Class balance chart** ties directly to the F1 + PR-AUC metric choice in
  §4.6.
- **Top-feature correlation bar** identifies that structural features
  (heading counts, link counts, graph centrality) outrank raw word count —
  this finding is referenced in the slide deck's Insights slide and drives
  the recommendation engine's prioritisation logic.
- **Distribution-by-class histograms** show clean separation on
  `title_length` and `h2_count`, validating the rule-based suggestion ranges
  (30–60 char titles, ≥ 2 H2s) before the rules ever ran.

**EDA charts in the dashboard** (`src/dashboard/components/charts.py`):
class balance · domain breakdown · top-features-by-correlation · feature
distribution histograms · feature-by-class box plot · correlation heatmap
(top 15) · two-feature scatter explorer.

### 3.5 Section 4b — Data Pre-processing & Feature Engineering `(+10, ××)`

This section is double-deduct, so every checkbox is addressed explicitly.

| Rubric checkbox | What we did |
|-----------------|-------------|
| **Efficient scraping / data collection** | `src/scraping/doc_scraper.py` — async crawler with per-domain rate limiting, robots.txt parsing, and per-page JSON sidecars. SHA1-keyed file naming makes re-fetches idempotent. |
| **Effective combination from distinct sources** | Two-source join (HTML × SERP) on a `host + path.lower()` key — see `src/features/build_features.py`. The same key is used by the graph builder so feature rows and graph nodes align 1-to-1. |
| **Handle null values** | After the join, every numeric column is `.fillna(0.0)` in `evaluate.load_features` — defensible because every numeric feature has a meaningful zero ("0 H2 headers", "PageRank 0 for an isolated node"). String columns can't be null after the deterministic builders run. |
| **Handle outliers** | Heavy-tailed PageRank distribution detected in EDA → kept untransformed because tree models are scale-invariant and LR uses `StandardScaler` which mitigates the leverage. Outlier rows (e.g. 8K-word indexes) are kept rather than clipped because they carry real signal — they are the SEO "link magnets". The 98th-percentile clipping is x-range only (visualisation), not data mutation. |
| **Engineer appropriate new features** | Five engineered families (content / metadata / structural / TF-IDF / graph) — every column is a *derived* signal, not a raw scrape attribute. See the family table in §4.5 of the README. |
| **One-hot encode categorical** (if applicable) | Not applicable — every feature is numeric or pre-binarised (e.g. `keyword_in_title ∈ {0, 1}`, `has_meta_description ∈ {0, 1}`). The only categorical column is `domain`, which is held aside as an identifier and not used for prediction (avoids site-leakage; the model should not learn "react.dev pages just rank"). |
| **Check correlation / remove highly correlated** | Correlation heatmap rendered in EDA + the dashboard **EDA** tab. No highly correlated pairs (\|ρ\| > 0.95) survive after engineering — TF-IDF columns are by construction near-orthogonal, and structural features are independent counts. |
| **Address imbalanced data** | See §3.3 / §4.6. Both pre-processing (oversampling) and in-loss (`class_weight`, `scale_pos_weight`, `pos_weight`) strategies. |
| **Scale data** | `StandardScaler(with_mean=False)` for LR (preserves TF-IDF sparsity); no scaling for tree models (scale-invariant); `StandardScaler` fit on the train split for the MLP, with `mean` + `scale` persisted into the checkpoint so dashboard inference scales identically to training. |

**Feature-engineering rationale** (each step → an EDA finding):

1. **TF-IDF (50 cols)** — content alone might not be enough for ranking, but
   topical overlap with high-ranking pages plausibly is. We project each
   page onto the corpus's top-50 unigrams + bigrams.
2. **Graph features (6 cols)** — pages embedded in dense link economies
   (high in-degree, high PageRank, high authority) tend to rank — explicit
   topology features let the models exploit that.
3. **Structural counts (7 cols)** — headings + link counts + alt-text
   coverage are the SEO best-practice targets the dashboard's
   recommendation engine talks about, so they need to be features.
4. **Metadata (4 cols)** — title and meta-description are the two strongest
   classical SEO signals; any model that doesn't see them is missing the
   story.
5. **Content basics (5 cols)** — text length, word count, sentence count,
   Flesch reading ease, keyword density. Cheap, well-understood, useful as
   sanity-check signals.

### 3.6 Section 5b — Model Assessment & Hyperparameter Tuning `(+8, ××)`

This section is double-deduct, with stacked penalties for missing tuning,
inappropriate metrics, conceptual error, and missing justification. Every
penalty path is closed below.

| Failure mode (rubric) | How we close it |
|-----------------------|-----------------|
| **No hyperparameter tuning** | Every sklearn-family model is wrapped in `RandomizedSearchCV` over a hand-narrowed `param_distributions`. The MLP runs a manual grid + early-stopping. Best params + CV F1 are saved to `models/metrics/*.json`. |
| **Inappropriate evaluation metrics** | Headline panel is F1 + ROC-AUC + PR-AUC + full confusion matrix. **Accuracy is not used** as a primary metric — explicitly avoided per the rubric's stated common mistake on imbalanced data. PR-AUC is the imbalance-aware tie-breaker. |
| **Conceptual error: increased regularisation on underfit / increased complexity on overfit** | `MODELING_DECISIONS.md` documents the per-model lesson explicitly: e.g. the MLP showed high precision / low recall (under-prediction → conservative threshold), and we *stopped* adding model complexity rather than going deeper. The lesson "stop adding complexity, the next gain comes from features" is the exact opposite of the conceptual-error path. |
| **Conceptual error: stating "fine" by one accuracy metric and ignoring others** | Comparison table reports five numbers per model (F1 / ROC-AUC / PR-AUC / precision / recall) plus the confusion matrix. The dashboard **Models** tab renders ROC + PR curves regenerated from saved estimators on the same hold-out split, so reviewers can verify visually. |
| **Justification missing: grid search without understanding** | Per-model `Why tried · Search space · CV strategy · Results · Lesson for next model` blocks in [`MODELING_DECISIONS.md`](MODELING_DECISIONS.md). The search-space choices are defended (loguniform for `C`, `learning_rate`; integer for tree depths; categorical lists for `class_weight`). |
| **Justification missing: only reporting best-model performance** | The full comparison table is reported (LR / RF / XGBoost / MLP). LR is intentionally retained as the linear baseline even though it loses by 3 F1 pts, because it is the rubric's required "baseline → advanced" first step. |

**Methodical approach evidence:**

1. **Unified evaluator** — every model calls
   `evaluate_classifier(name, y_true, y_pred, y_proba)` from
   `src/models/evaluate.py`. Same metric calculation, same JSON schema. No
   per-model fudging of metrics.
2. **Same hold-out split** — `stratified_split(X, y, test_size=0.2,
   random_state=42)` is the only test-split function, called identically
   from every trainer. The dashboard's Models tab recreates this exact split
   to regenerate ROC + PR curves on the fly.
3. **Same CV splitter** — `cv_splitter(5, random_state=42)` returns the
   same `StratifiedKFold` object to every `RandomizedSearchCV` call.
4. **Persisted artifacts** — every trainer dumps both the fitted estimator
   (`models/<name>.joblib`) and the metrics JSON
   (`models/metrics/<name>.json`). The dashboard reads from these, so the
   numbers shown to graders are always the numbers the trainer produced.

**Per-model lesson trail** (rubric "iterative methodical approach"):

| Step | Result | Decision |
|------|--------|----------|
| LR baseline | F1 = 0.871 | Engineered features carry signal. Symmetric error suggests boundary problem, not class bias. → try non-linear bagging. |
| Random Forest | F1 = 0.902 | +3 F1 over LR. Non-linear interactions matter. → try boosting in the same family. |
| XGBoost | F1 = 0.885 | Statistical tie with RF. Tree-ensemble ceiling reached at this corpus size. → try a different inductive bias entirely. |
| MLP | F1 = 0.608 | High precision / low recall — small-data tabular favours trees, as documented in the literature. → stop adding model complexity; future gain is in features and per-domain calibration. |

---

### 3.7 Section 7 — Code Quality & Readability `(+10, ×)`

Every "Excellent" criterion (modular, broken into logical sections,
in-line comments where helpful) is met:

| Criterion | Evidence |
|-----------|----------|
| Modular | `src/{scraping,features,graph,models,recommendations,dashboard}/` — six top-level packages, each owning one concern. No file imports from a downstream package. |
| Logical sections | Every multi-function module separates concerns with `# ── header ─` rules (e.g. `src/dashboard/app.py` — caching · live scraping · render helpers · tab handlers · main). |
| Type hints | Every function has full type hints. `from __future__ import annotations` at the top of every module enables forward references without quoting. |
| Docstrings | Every module starts with a multi-line docstring explaining its role + CLI usage if applicable. Every public function has a docstring describing args, returns, and edge-case behaviour. |
| Consistent naming | snake_case for functions/variables, PascalCase for dataclasses (`Metrics`, `Suggestion`), SCREAMING_SNAKE for module-level constants. |
| In-line comments where helpful | Comments explain the *why* (non-obvious invariants, hidden constraints, subtle behaviour) — not the *what*, which the names already convey. Examples: the median-fill rationale for live URLs in `app.py`, the `with_mean=False` choice in `baseline.py`, the resume-safe SERP fetch in `serp_client.py`. |
| No dead code | `git grep TODO` returns project-management TODOs only, not "fix later" code. No commented-out blocks. |
| Pinned dependencies | `requirements.txt` pins core libraries; the dashboard adds `plotly>=5.20,<6.0` and `streamlit>=1.41,<2.0` (lower bound chosen so the `width="stretch"` API works after the 2025-12-31 deprecation). |

### 3.8 Section 8 — Application of Course Topics `(+10, ×)`

The rubric's "Excellent" rating requires **at least 6** topics, used with
logical conditioning on the project goal. We claim **seven**:

| # | Topic | Where it lives | Conditioning on the goal |
|---|-------|----------------|--------------------------|
| 1 | **Supervised Learning** | `src/models/{baseline,tree_models,boosting,neural}.py` | Four classifiers solve the binary-classification framing of the SEO ranking problem. |
| 2 | **Graphs** | `src/graph/{build_graph,graph_features}.py` | The page-to-page link graph is a first-class data structure; PageRank, HITS, clustering coefficient, in/out-degree become per-node features used directly by every model. |
| 3 | **Joins** | `src/features/build_features.py` | Page features (Source 1) joined to SERP labels (Source 2) on a `host + path.lower()` key — without the join, there is no `is_top_10` label. |
| 4 | **Text representations / embeddings** | `src/features/content_features.py` (`fit_tfidf`, `transform_tfidf`) | Corpus-fitted TF-IDF over scraped page text gives each page a 50-dim vector that captures topical similarity to the high-ranking pages — directly relevant to the prediction problem (search engines are themselves text-similarity systems). |
| 5 | **Hypothesis testing** | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) — t-tests on top continuous features (e.g. `title_length`, `word_count`, `pagerank`) between top-10 vs not-top-10 groups; chi-square on binary features (`keyword_in_title`, `has_meta_description`). | Pre-modeling sanity check: significant test → keep the feature; insignificant → flag as candidate for ablation. |
| 6 | **Different methods of hyperparameter tuning** | `RandomizedSearchCV` for sklearn family + manual grid + early-stopping for the PyTorch MLP — two distinct tuning paradigms tied to the underlying training loop. | Different tuners for different model families avoids the rubric's "grid search without understanding" pitfall. |
| 7 | **Deep Learning** | `src/models/neural.py` — 2-layer PyTorch MLP with dropout, BCE-with-logits loss, `pos_weight` for imbalance handling. Trained in Colab, exported via the §VI checkpoint pattern (`state_dict + config + scaler + feature_names`). | Direct comparison against tree ensembles on the same problem — the empirical case for "small-data tabular favours trees" (see §3.6 lesson trail). |

The "Excellent" tier requires ≥ 6 topics; we claim 7 with a concrete file
+ rationale per topic. None of the disqualifying patterns from the rubric
are present (no standalone K-Means without using clusters as a feature, no
ER diagram for a single-table dataset).

### 3.9 Section 9 — Quality of Dashboard Demo `(+20, ×)`

Seven-tab Streamlit app with custom CSS, Plotly figures, and live URL
prediction — the rubric's "Excellent" criterion is "highly polished, visually
appealing, bug-free dashboard … thoroughly covers all aspects of the project,
including both EDA and modeling work."

| Tab | What it shows | Maps to rubric requirement |
|-----|---------------|----------------------------|
| **Predict** | Live URL → scrape → top-10 probability per loaded model · per-prediction SHAP attribution panel · query-override input | "Interactive (not static)" — the user provides input that drives a real prediction |
| **EDA** | Class balance · domain mix · top-feature correlation ranking · distribution explorer (histogram + box plot) · correlation heatmap · two-feature scatter | "Covers all aspects of the project, including EDA" |
| **Graph** | PageRank distribution · in/out-degree scatter · HITS hub vs authority · URL-hierarchy network rendered with NetworkX spring layout · top-pages-by-PageRank table | Surfaces the link-graph layer which is the most distinctive part of the project |
| **Models** | Held-out metrics table · F1/ROC-AUC/PR-AUC comparison bar · ROC + PR curves regenerated from saved estimators · side-by-side confusion matrices · feature importance for tree + linear models | "Covers modeling work" + the rubric's expectation of evaluation visualisation |
| **Recommendations** | Hybrid rule + SHAP-ranked actionable suggestions per page | Showcases the project's value proposition end-to-end |
| **What-if** | Slider-driven counterfactual probability simulator · live deltas vs original prediction | Interactive, exploratory |
| **About** | Objective · data-source pivot citation · methodology summary · team | Project context |

**Visual polish** lives in [`src/dashboard/styles.py`](src/dashboard/styles.py):
gradient sidebar with pill-style nav, card-framed Plotly charts, refined metric
cards, hand-styled banner with gradient + radial accent, hidden Streamlit
chrome (footer, hamburger), `width="stretch"` everywhere (post-deprecation API).

**Custom Plotly chart factories** — 16 figures with a shared palette and layout
shell live in [`src/dashboard/components/charts.py`](src/dashboard/components/charts.py).
Plotly is one of the rubric's listed "other visualization packages" Difficulty
items even though we don't claim it as a Difficulty concept.

**No bugs path** — the dashboard is exercised end-to-end via Streamlit's
`AppTest` harness in CI: every tab is selected and asserted to render without
exceptions. EDA: 7 plots, Graph: 4 plots, Models: 7 plots, others: card-based
content. Zero exceptions across the full nav.

### 3.10 Section 11 — Quality of Presentation `(+10, ××)`

Double-deduct section, so every penalty path is closed:

| Penalty (rubric) | How we close it |
|------------------|-----------------|
| Missing section | All five required sections covered: objective + value proposition + dataset (slides 1–3), major EDA learnings with 4+ formatted charts (slides 5–7), modeling results (slides 8–9), implications + insights (slide 11), challenges + limitations + future work (slide 12). |
| Missing / inappropriate visuals | Every modeling slide has a 300-DPI matplotlib chart embedded (`presentation/charts/*.png`); no slide is text-only after the title. |
| Under / over time | Speaker script in [`presentation/script.md`](presentation/script.md) totals 9:00 with per-slide timings; pre-record checklist warns against editing to fit time. |
| Unpolished | Hand-styled deck via `python-pptx` builder ([`presentation/build_slides.py`](presentation/build_slides.py)); typography, palette, and chart card-frames mirror the dashboard so the two read as one product. |
| Not all members presented | Speaker split table at the top of [`presentation/script.md`](presentation/script.md) — Rahil and Ayush each own 5 slides + co-present open and close. |
| No submission | `presentation/slides.pptx` committed; export to `slides.pdf` via the `soffice --convert-to pdf` one-liner in the script's "Exporting to PDF" section. |

---

## 4. Course-concept usage map

A reverse index: for each major course concept, the list of files that use it.

| Concept | Files |
|---------|-------|
| Supervised Learning | `src/models/baseline.py`, `tree_models.py`, `boosting.py`, `neural.py`, `evaluate.py` |
| Graphs (PageRank, HITS, clustering) | `src/graph/build_graph.py`, `graph_features.py` · dashboard tab `tab_graph` |
| Joins (multi-source) | `src/features/build_features.py` · `src/scraping/serp_client.py` |
| Text Representations (TF-IDF) | `src/features/content_features.py` |
| Hypothesis Testing | `notebooks/01_eda.ipynb` (t-test + chi-square cells) |
| Different Hyperparameter-Tuning Methods | `RandomizedSearchCV` in every sklearn trainer + manual grid in `src/models/neural.py` |
| Deep Learning | `src/models/neural.py` (PyTorch MLP) |
| Imbalance Handling | `src/features/balance.py` · `class_weight`/`scale_pos_weight`/`pos_weight` in trainers |
| Feature Importance | `src/dashboard/components/model_helpers.py` · `src/recommendations/recommend.py` (SHAP) |
| Visualisation (Plotly) | `src/dashboard/components/charts.py` · matplotlib in `presentation/build_charts.py` |

---

## 5. File map — every rubric line points somewhere

| Rubric line | File |
|-------------|------|
| Free, public, legal data | `src/scraping/doc_scraper.py` (robots policy) · `data/README.md` |
| ≥ 2 distinct sources | HTML (`doc_scraper.py`) + SERP (`serp_client.py`) |
| Documented narrowed-domain pivot | Top of `README.md` + `data/README.md` |
| 7-10+ feature columns | `data/processed/features.csv` — 72 columns |
| Modular code | `src/{scraping,features,graph,models,recommendations,dashboard}/` |
| Documentation | `README.md`, `MODELING_DECISIONS.md`, `JUSTIFICATIONS.md` (this file), `USER_MANUAL.md`, `data/README.md` |
| EDA notebook | `notebooks/01_eda.ipynb` |
| EDA in dashboard | `src/dashboard/app.py` `tab_eda` |
| Pre-processing pipeline | `src/features/build_features.py` |
| Imbalance handling (concept) | `src/features/balance.py` + `scripts/balance_dataset.py` |
| Hyperparameter tuning (concept) | `PARAM_DIST` in every `src/models/*.py` trainer |
| Feature importance (concept) | `src/dashboard/components/model_helpers.py` + `src/recommendations/recommend.py` |
| Imbalance-aware metrics | `src/models/evaluate.py` |
| Modeling progression | `src/models/baseline.py` → `tree_models.py` → `boosting.py` → `neural.py` |
| Model comparison | `notebooks/03_model_comparison.ipynb` + dashboard `tab_models` |
| Interactive demo | `streamlit run src/dashboard/app.py` |
| Slide deck (PowerPoint) | `presentation/slides.pptx` |
| Speaker script + timing | `presentation/script.md` |

---

## 6. Reproducibility

The full pipeline is **eight CLIs in order**:

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
python -m src.models.neural          # (or train in Colab + drop checkpoint)
streamlit run src/dashboard/app.py
```

Single `random_state = 42` across every randomness boundary
(`StratifiedKFold`, `train_test_split`, `RandomizedSearchCV`,
`numpy.random`, `torch.manual_seed`). Pinned `requirements.txt`. All
generated artifacts (feature matrix, joblibs, metrics JSONs, slide deck,
chart PNGs) committed to git so a fresh clone reproduces the dashboard
without external API calls.

---

*End of justifications.*

