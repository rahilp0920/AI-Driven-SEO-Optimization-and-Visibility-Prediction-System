# User Manual

**AI-Driven SEO Ranking Predictor & Recommendation System**
NETS 1500 (HW5) + CIS 2450 final project — Rahil Patel & Ayush Tripathi

This manual walks through installing, running the data pipeline, training the models, and using
every feature of the Streamlit dashboard. Screenshots correspond to a clean clone with the full
pipeline run on 5 developer-documentation domains.

---

## 1. Install

Tested on Python 3.11 (macOS / Linux). Apple-Silicon Macs work; CUDA optional for the MLP.

```bash
git clone https://github.com/rahilp0920/AI-Driven-SEO-Optimization-and-Visibility-Prediction-System.git
cd AI-Driven-SEO-Optimization-and-Visibility-Prediction-System
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` and add **one** of:

```
BRAVE_SEARCH_KEY=...     # https://brave.com/search/api/  (free tier preferred — 2K req/month)
SERPAPI_KEY=...          # https://serpapi.com            (free tier 100 req/month — fallback)
```

Both providers return Google's organic top-10 in the same shape; the client picks whichever key
is present (Brave preferred).

---

## 2. Run the data pipeline

The pipeline is four CLIs run in order. Each step writes to `data/` and is **idempotent** — safe
to re-run after a crash.

### 2.1 — Scrape per-domain HTML (~1500 pages total)

```bash
python -m src.scraping.doc_scraper --domain docs.python.org       --limit 300
python -m src.scraping.doc_scraper --domain developer.mozilla.org --limit 300
python -m src.scraping.doc_scraper --domain react.dev             --limit 250
python -m src.scraping.doc_scraper --domain nodejs.org            --limit 200
python -m src.scraping.doc_scraper --domain kubernetes.io         --limit 250
python -m src.scraping.doc_scraper --domain fastapi.tiangolo.com  --limit 200
```

Each invocation writes HTML + a JSON sidecar (URL, status, fetched_at, title, outbound_links)
to `data/raw/<domain>/`. `robots.txt` is fetched once per domain at the start of the crawl —
disallowed paths are logged and skipped. Default rate limit: 1 second between requests
(configurable via `--delay` or the `SCRAPER_DELAY_SECONDS` env var).

**Screenshot — `data/raw/` after a full scrape:**
*(Run the commands above, then take a screenshot of `ls -lh data/raw/` showing one
subdirectory per domain. Save as `assets/screenshots/01_data_raw.png`.)*

### 2.2 — Build queries from page titles

```bash
python -m src.scraping.serp_client build-queries
```

Walks every JSON sidecar in `data/raw/`, derives one query per page from the `<title>` tag
(strips trailing site-name boilerplate like " — Python 3.12"), and writes
`data/interim/queries.csv` (`query_id, query, source_url`).

### 2.3 — Fetch SERP rankings (top-10 per query)

```bash
python -m src.scraping.serp_client fetch
```

Reads `queries.csv`, hits Brave Search (or SerpApi), and writes `data/interim/serp.csv` with
one row per `(query_id, rank, url, title, snippet)`. The fetch is **resume-safe** — re-running
skips already-fetched query_ids, important for protecting the free-tier quota if the run is
interrupted.

### 2.4 — Build features and graph

```bash
python -m src.features.build_features         # → data/processed/features.csv
python -m src.graph.build_graph                # → data/interim/graph.pkl
python -m src.graph.graph_features             # left-joins graph features into features.csv
```

`features.csv` ends up with ~78 numeric columns: 5 content + 4 metadata + 7 structural + 50
TF-IDF + 6 graph + the `is_top_10` target. Identifier columns (`url`, `domain`, `query_id`,
`query`) are kept aside, not in the feature space.

**Screenshot — `data/processed/features.csv` head:**
*(`head -5 data/processed/features.csv` → `assets/screenshots/02_features_head.png`.)*

---

## 3. Train the models

Each script saves a fitted model + a metrics JSON (P/R/F1/ROC-AUC/PR-AUC/confusion-matrix) to
`models/` and `models/metrics/`.

```bash
python -m src.models.baseline      # Logistic Regression, RandomizedSearchCV (n_iter=30)
python -m src.models.tree_models   # Random Forest,        RandomizedSearchCV (n_iter=30)
python -m src.models.boosting      # XGBoost,              RandomizedSearchCV (n_iter=40)
python -m src.models.neural        # MLP (local fallback). For Colab — see §3.1.
```

### 3.1 — MLP via Colab (rubric §VI checkpoint pattern)

Open `notebooks/03_model_comparison.ipynb` in Colab, mount Drive, copy
`data/processed/features.csv` over, `!pip install -r requirements.txt`, then run:

```python
from src.models.neural import train
train(csv_path="features.csv", out_checkpoint="mlp_checkpoint.pt", epochs=80)
# files.download("mlp_checkpoint.pt")
```

Place the downloaded `mlp_checkpoint.pt` into `models/` locally. The dashboard's
`load_checkpoint()` reconstructs the wrapper with the saved scaler + feature_names, so local
inference returns the exact predictions the Colab session produced.

### 3.2 — Compare and pick the winner

```bash
jupyter nbconvert --execute notebooks/03_model_comparison.ipynb --to notebook --inplace
```

Prints the comparison table, plots metrics + confusion matrices, picks the winner by
`(F1 desc, PR-AUC desc)`, and writes `assets/charts/06_shap_summary.png`.

**Screenshot — comparison table + winner:**
*(After `jupyter nbconvert`, screenshot the bar chart cell + the "Winner: ..." printout. Save
as `assets/screenshots/03_model_comparison.png`.)*

---

## 4. Run the dashboard

**Prerequisite:** `data/processed/features.csv` must exist (from §2.4 / `make features`). The
app reads it to recover feature column order and to refit TF-IDF on corpus text from
`data/raw/` when available. Trained model files under `models/` (see §3) are required for
predictions.

```bash
streamlit run src/dashboard/app.py
```

Opens at `http://localhost:8501`. The sidebar has four tabs.

### 4.1 — Predict tab

Paste a documentation URL, then click **Scrape & predict**. The dashboard fetches the page
over HTTP (no SERP API key needed for this step), extracts the same feature set used in
training (zero-fills graph features since the live page is not in the corpus graph), and
runs every loaded model.

**Headline output — SEO score (0–100).** The dashboard converts each model's `P(top-10)` for
the live page into a **percentile rank** within that model's distribution over the training
feature matrix, then averages those percentiles into a single SEO score. Higher = stronger
SEO signals relative to the corpus, **not** a literal probability of ranking. A verdict
bucket (Strong ≥ 70, Moderate 40–69, Weak < 40) and a **model agreement** indicator (high /
mixed / low spread across models) are shown beside the score. Per-model probabilities and
their percentiles are still available under "Per-model breakdown".

**Topic query:** By default the query string matches the training pipeline (stripped from
`<title>`). If that string is misleading (e.g. a docs root title that becomes only a version
number), use **Topic query override** and either scrape again or click **Re-predict with query
only** to re-featurize the last page with your query (keyword features update; SERP is not
re-fetched).

If the URL is empty, you get a warning. If the fetch fails (network error, timeout, HTTP
error), you get an error and no placeholder scores — fix connectivity or try another URL.

**Screenshot:** `assets/screenshots/04_dashboard_predict.png` *(use a successful live scrape
for the rubric; avoid empty error states in submission captures.)*

### 4.2 — Recommendations tab

After a **successful** prediction on the Predict tab, this tab shows ≥3 actionable suggestions ranked by SHAP impact.
Each card shows the action ("Title is 80 chars — shorten to 30-60"), the responsible feature
(tag), and a "why" line explaining the SEO rationale. Suggestions are computed by
`src/recommendations/recommend.py` — rules-based core, SHAP-ranked when a tree model is loaded.

**Screenshot:** `assets/screenshots/05_dashboard_recommendations.png`

### 4.3 — What-if tab

Requires a successful prediction first (same session state as Predict). Eight live sliders
(title length, meta description length, H2 count, alt-text coverage, keyword density,
internal-link count, word count, keyword-in-title binary). Move any slider — the headline
**SEO score** and verdict update instantly, and a **Δ vs original** card shows the change in
score (green = improvement, red = regression). Per-model probability deltas remain available
under "Per-model breakdown".

**Screenshot:** `assets/screenshots/06_dashboard_whatif.png`

### 4.4 — About tab

Project description, the pivot rationale verbatim (Ricky Gong 3/29 email), AI-usage disclosure
broken down by category (what we used AI for, what we did NOT use AI for), and team work split.

**Screenshot:** `assets/screenshots/07_dashboard_about.png`

---

## 5. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `no SERP API key found` on `serp_client fetch` | Add `BRAVE_SEARCH_KEY` or `SERPAPI_KEY` to `.env`. |
| Scraper hangs on a single domain | Check the domain's `robots.txt` — some sites disallow `/`; raise `--delay` or skip the domain. |
| Dashboard shows "No trained models loaded" | Run §3 first; at minimum train XGBoost. |
| Dashboard error about missing `data/processed/features.csv` | Run §2.4 (`make features` or the three `python -m` commands) before starting Streamlit. |
| MLP checkpoint won't load | Confirm the file came from the *same* `requirements.txt` — torch version matters. Use `--evaluate-only` to test load. |
| Streamlit looks like default white box | Browser may have cached an old CSS — hard refresh (cmd-shift-R / ctrl-shift-R). |

---

## 6. Where things live (quick reference)

```
src/scraping/      doc_scraper.py + serp_client.py
src/features/      content_features.py, metadata_features.py, structural_features.py, build_features.py
src/graph/         build_graph.py, graph_features.py
src/models/        baseline.py, tree_models.py, boosting.py, neural.py, evaluate.py
src/recommendations/  recommend.py
src/dashboard/     app.py, styles.py
notebooks/         01_eda.ipynb, 02_graph_analysis.ipynb, 03_model_comparison.ipynb
data/raw/          per-domain HTML + JSON sidecars
data/interim/      queries.csv, serp.csv, graph.pkl
data/processed/    features.csv (final feature matrix)
models/            *.joblib + mlp_checkpoint.pt + metrics/*.json
assets/charts/     EDA charts (≥300 DPI) + 06_shap_summary.png
```

For deeper architecture / rubric→file map, see `README.md`. For modeling decisions and cut
models, see `MODELING_DECISIONS.md`.
