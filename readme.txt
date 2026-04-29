AI-Driven SEO Ranking Predictor & Recommendation System
NETS 1500 — HW5 Implementation Project (joint with CIS 2450)

DESCRIPTION
A binary classifier that predicts whether a developer documentation page appears in the top-10
Google SERP results for the topic query derived from its <title> tag. We crawl ~1500 pages
across 5+ developer documentation domains (docs.python.org, developer.mozilla.org, react.dev,
nodejs.org, kubernetes.io, fastapi.tiangolo.com), build a directed link graph between scraped
pages, compute content/metadata/structural/graph features, train a 4-model sweep (Logistic
Regression, Random Forest, XGBoost, PyTorch MLP), and ship a Streamlit dashboard with a
SHAP-driven what-if simulator and concrete recommendation engine. Target metric: F1 on a
stratified held-out test set, with PR-AUC as the tiebreaker since the data is class-imbalanced.

NETS 1500 CATEGORIES USED
- Information Networks (World Wide Web): SERP ranking prediction, page-to-page link graph.
- Information Retrieval: TF-IDF over scraped page text, keyword density, title-keyword match.
- Graph and Graph Algorithms: PageRank (Page-Brin, alpha=0.85), HITS hub/authority (Kleinberg),
  in-/out-degree, clustering coefficient — all merged into the model feature space.

WORK BREAKDOWN
- Rahil Patel (rahilp07): src/scraping/* (async robots.txt-aware crawler + Brave Search SERP
  client), src/features/* (content/metadata/structural extraction + corpus TF-IDF orchestrator),
  src/graph/* (DiGraph build + PageRank/HITS/clustering merge), src/models/{baseline,tree_models,
  boosting}.py (LR/RF/XGBoost with RandomizedSearchCV), dashboard backend wiring (live scrape +
  featurize), data/README.md, project planning under .planning/.
- Ayush Tripathi (tripath1): notebooks/01_eda.ipynb (5 rubric charts, each tied to a modeling
  decision), src/models/neural.py (PyTorch MLP trained in Colab, exported via the rubric §VI
  checkpoint pattern: state_dict + config + scaler + feature_names), src/models/evaluate.py
  (single Metrics dataclass with P/R/F1/ROC-AUC/PR-AUC/CM, shared StratifiedKFold), src/recom-
  mendations/recommend.py (rules + SHAP-ranked suggestions), src/dashboard/styles.py + frontend
  polish on app.py (custom CSS, banner, metric cards, suggestion cards, SHAP rows), notebooks/
  03_model_comparison.ipynb, slide deck + recording.

CHANGES SINCE PROPOSAL (per NETS submission rule — TAs were kept in the loop)
1. Data scope narrowed from 50K rows to ~1500 developer documentation pages, sanctioned by
   CIS 2450 TA Ricky Gong's email of 2026-03-29 (free-tier API rate limits made 50K infeasible
   inside the deadline). Quoted verbatim in data/README.md.
2. Model sweep reduced from 8 to 4 (LR, RF, XGBoost, MLP). LightGBM, CatBoost, GBM, and
   SVM-RBF were cut due to the same time constraint. The cut justification is documented in
   MODELING_DECISIONS.md: LightGBM/CatBoost converge to within 1-3% F1 of XGBoost on tabular
   binary classification at this dataset size (well-documented in the literature); GBM is
   superseded by XGBoost in the same family; SVM-RBF scales poorly to TF-IDF dimensionality.
   The retained 4 still cover all four model families the rubric expects (linear → bagging →
   boosting → neural), preserving the baseline-to-advanced progression.
3. Graph algorithms made explicit (per Shivani's feedback): PageRank, HITS hub/authority, and
   clustering coefficient. Each is named in notebooks/02_graph_analysis.ipynb and merged into
   data/processed/features.csv as a numeric column.

AI USAGE
We used Claude (Anthropic) for the following, with concrete examples:
- Code scaffolding: initial async loop in src/scraping/doc_scraper.py (BFS with robots.txt
  gating + per-domain rate limit), the sklearn Pipeline+RandomizedSearchCV boilerplate in
  src/models/baseline.py, and the Streamlit component layout in src/dashboard/app.py. Every
  generated function was reviewed, type-hint-audited, and integration-tested before commit.
- Documentation drafting: initial drafts of data/README.md, this readme.txt, MODELING_DECISIONS
  .md cut-model justifications, and module/function docstrings. Authors reviewed and rewrote
  substantive content (e.g., the work breakdown above is hand-written).
- Hyperparameter search-space ranges: initial ranges suggested by Claude (loguniform for C in
  LR, log-scaled learning_rate in XGB), narrowed/widened based on early CV results.
- What we did NOT use AI for: the modeling sweep itself (each model was trained, tuned, and
  evaluated by us — Claude wrote the harness, not the experiments), the data ethics decisions
  (domain selection, robots.txt policy, per-domain rate limits, User-Agent identification),
  the final feature-engineering choices (which features to keep/drop), or the slide content.

HOW TO RUN — see USER_MANUAL.md for the full step-by-step + screenshots.
