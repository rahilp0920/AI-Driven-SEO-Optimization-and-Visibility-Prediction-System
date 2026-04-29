# Requirements: AI-Driven SEO Ranking Predictor

**Defined:** 2026-04-29
**Core Value:** Score maximally on the CIS 2450 rubric while satisfying NETS 1500's WWW + IR + Graph-Algorithms course-topic integration with a single shared codebase.

Every requirement maps to a rubric line in CIS 2450 (`grade_against_cis2450`) or NETS 1500 (`grade_against_nets1500`). REQ-IDs are referenced in PRs, MODELING_DECISIONS.md, and the final README.md rubric-to-file map.

## v1 Requirements

### Data Layer (DATA)

- [ ] **DATA-01**: Async, rate-limited, robots.txt-respecting scraper for ~1500 developer documentation pages from docs.python.org, react.dev, Stripe docs, MDN, and at least one other dev-doc domain *(CIS 2450: §2 Data Source · raw web data; satisfies "2 distinct sources" half)*
- [ ] **DATA-02**: SERP rankings collected via SerpApi free tier OR Brave Search API for each page's primary topic query (query derived from `<title>`) *(CIS 2450: §2 Data Source · web API; satisfies "2 distinct sources" half)*
- [ ] **DATA-03**: Raw HTML stored verbatim in `data/raw/`; cleaned text + parsed metadata in `data/interim/`; final feature matrix in `data/processed/` *(CIS 2450: codebase modularity)*
- [ ] **DATA-04**: `data/README.md` documents source, scrape date, scrape script command, page count, robots.txt compliance, and quotes Ricky's 3/29 email verbatim as pivot rationale *(CIS 2450: data documentation; addresses Ricky feedback directly)*
- [ ] **DATA-05**: Stretch — augment row count via (page, query) pairs (~7 queries per page → ~10K rows) to defend against the rubric's "50K+ rows after cleaning" preference *(CIS 2450: §IX Data Source Requirements; mitigates known compliance risk)*
- [ ] **DATA-06**: Public + legal data only; respect rate limits and ToS *(CIS 2450: §2 hard requirement, failure = automatic 0)*

### Feature Engineering (FEAT)

- [ ] **FEAT-01**: Content features — TF-IDF vectors, keyword density, Flesch readability, text length *(CIS 2450: §IV.A; NETS 1500: Information Retrieval category)*
- [ ] **FEAT-02**: Metadata features — title length, meta description length/presence, keyword-in-title flag *(CIS 2450: §IV.A)*
- [ ] **FEAT-03**: Structural features — H1/H2/H3 counts, internal/external link counts, image count, alt-text coverage ratio *(CIS 2450: §IV.A; NETS 1500: WWW category)*
- [ ] **FEAT-04**: Final feature matrix has ≥10 numeric columns excluding target *(CIS 2450: §IX "7-10+ columns")*
- [ ] **FEAT-05**: Module-level docstrings + per-function docstrings (Args/Returns) + type hints throughout `src/features/` *(CIS 2450: codebase docs)*

### Graph Layer (GRAPH) — *also satisfies NETS 1500 graph-algorithms requirement*

- [ ] **GRAPH-01**: NetworkX directed graph from internal links between scraped pages, persisted to disk *(NETS 1500: Graph algorithms — explicit Shivani feedback)*
- [ ] **GRAPH-02**: PageRank computed and merged into feature matrix *(NETS 1500: graph algorithms; CIS 2450: feature engineering)*
- [ ] **GRAPH-03**: HITS hub + authority scores computed and merged *(NETS 1500: graph algorithms)*
- [ ] **GRAPH-04**: In-degree, out-degree, clustering coefficient computed and merged *(NETS 1500: graph algorithms)*
- [ ] **GRAPH-05**: `notebooks/02_graph_analysis.ipynb` shows PageRank distribution, link structure summary stats *(CIS 2450: EDA depth)*

### EDA (EDA)

- [ ] **EDA-01**: `notebooks/01_eda.ipynb` produces 3-5 well-formatted charts informing modeling, each saved as ≥300 DPI PNG to `assets/charts/` *(CIS 2450: §VI Presentation requires "top 3-5 well-formatted charts that deepened your understanding and informed your model")*
- [ ] **EDA-02**: Class balance plot — justifies F1/ROC-AUC choice over accuracy *(EDA chart #1, ties to MODEL-EVAL)*
- [ ] **EDA-03**: Per-feature distribution plots with top-10 vs not-top-10 hue *(EDA chart #2)*
- [ ] **EDA-04**: Correlation heatmap of features vs target *(EDA chart #3)*
- [ ] **EDA-05**: PageRank distribution histogram on log-scale x-axis *(EDA chart #4; justifies graph features)*
- [ ] **EDA-06**: TF-IDF top-terms-by-class chart *(EDA chart #5)*
- [ ] **EDA-07**: Every chart has a markdown caption explicitly stating "this is why we made decision X in modeling" — no silent EDA *(CIS 2450: EDA must inform modeling, per rubric explicit wording)*
- [ ] **EDA-08**: `seaborn.set_style("whitegrid")` or custom matplotlib stylesheet — no default styling *(presentation polish)*

### Modeling (MODEL)

- [ ] **MODEL-01**: Logistic regression baseline (`src/models/baseline.py`) — answers "is the signal learnable?" *(CIS 2450: baseline-to-advanced progression)*
- [ ] **MODEL-02**: Random Forest (`src/models/tree_models.py`) — non-linearity + feature importance
- [ ] **MODEL-03**: Gradient Boosting sklearn (`src/models/tree_models.py`) — typically beats RF on tabular
- [ ] **MODEL-04**: XGBoost (`src/models/boosting.py`) — handles imbalance via `scale_pos_weight`
- [ ] **MODEL-05**: LightGBM (`src/models/boosting.py`) — speed comparison vs XGBoost
- [ ] **MODEL-06**: CatBoost (`src/models/boosting.py`) — native categorical handling
- [ ] **MODEL-07**: SVM with RBF kernel (`src/models/kernel.py`) — non-tree sanity check
- [ ] **MODEL-08**: Shallow MLP in PyTorch (`src/models/neural.py`) — Colab-trained, checkpoint loaded into dashboard *(CIS 2450: §VI Dashboard explicitly mentions Colab + checkpoint loading)*
- [ ] **MODEL-09**: All models use `StratifiedKFold` CV justified by class imbalance from EDA-02 *(modeling rigor)*
- [ ] **MODEL-10**: All models use `RandomizedSearchCV` with explicit `n_iter` and `cv` documented *(per proposal hyperparameter tuning commitment)*
- [ ] **MODEL-11**: Single unified evaluator in `src/models/evaluate.py` returns precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix *(CIS 2450: appropriate metrics for imbalanced data)*
- [ ] **MODEL-12**: `MODELING_DECISIONS.md` has a section per model documenting why-tried, hyperparameter search space + method, CV strategy, final metrics, kept/dropped decision tied to a prior result or EDA finding — **no silent choices anywhere** *(CIS 2450: modeling depth + readability)*
- [ ] **MODEL-13**: `notebooks/03_model_comparison.ipynb` ends with a comparison table of all 8 models on the same metrics, a clear winner, and a paragraph defending the choice tied to specific metric trade-offs *(CIS 2450: presentation §VI)*
- [ ] **MODEL-14**: SHAP analysis run on the winner; SHAP summary plot saved to `assets/charts/` *(dashboard interpretability + presentation depth)*

### Recommendation System (REC)

- [ ] **REC-01**: `src/recommendations/recommend.py` maps low/extreme feature values to actionable text suggestions ("add 2 H2 headers", "title is 80 chars, shorten to 50-60", "add alt text to 4 images") *(per proposal §IV.B; NETS 1500: recommendations advanced topic)*
- [ ] **REC-02**: Recommendations are concrete and actionable, never generic ("improve content quality" is not acceptable) *(per proposal challenges §V; quality bar)*

### Dashboard (DASH) — *20 pts CIS 2450*

- [ ] **DASH-01**: Streamlit app entry at `src/dashboard/app.py` with sidebar nav: Predict, Recommendations, What-if, About *(usability)*
- [ ] **DASH-02**: URL input box performs live feature extraction (scrape on demand) *(interactive demo)*
- [ ] **DASH-03**: Predicted top-10 probability rendered as a gauge or progress bar (not raw text) *(polish — comparable to homework dashboards)*
- [ ] **DASH-04**: SHAP-based feature breakdown panel showing top 5 contributing features *(interpretability)*
- [ ] **DASH-05**: Recommendation panel renders REC-01 output as actionable text *(value-add)*
- [ ] **DASH-06**: What-if simulator with sliders for editing features → live re-prediction showing prob delta *(rubric differentiator; original proposal §IV.B "simulation step")*
- [ ] **DASH-07**: Custom CSS via `st.markdown(unsafe_allow_html=True)` — no default Streamlit styling *(polish)*
- [ ] **DASH-08**: Demo data fallback so live demo never breaks if scraping fails on stage *(presentation safety)*
- [ ] **DASH-09**: "About" tab includes AI-usage disclosure (which steps used Claude/Cursor + how validated) *(CIS 2450 §2.b: "in your final deliverables (presentation & dashboard demo), you must document how you utilized help from AI"))*
- [ ] **DASH-10**: PyTorch MLP loaded via checkpoint pattern from rubric §VI *(CIS 2450 §VI Dashboard explicit code snippet)*

### NETS 1500 Deliverables (NETS) — *due 2026-04-29 23:59*

- [ ] **NETS-01**: `readme.txt` (one page) containing: project name + 4-5 sentence description, categories used (WWW + IR + Graph algorithms), work breakdown (Rahil/Ayush split), concrete AI usage details *(NETS 1500 hard requirement)*
- [ ] **NETS-02**: `USER_MANUAL.md` with screenshots of every dashboard feature + run instructions *(NETS 1500 hard requirement)*
- [ ] **NETS-03**: Submission to NETS 1500 Gradescope by 2026-04-29 23:59 with both group members added *(failure = teammate scored 0)*

### Documentation & Codebase Polish (DOCS) — *bulk of CIS 2450 83-pt codebase grade*

- [ ] **DOCS-01**: `README.md` (CIS 2450 main) maps each rubric requirement to the file path that satisfies it (a literal table) *(CIS 2450: codebase readability + traceability)*
- [ ] **DOCS-02**: `MODELING_DECISIONS.md` is committed and current — no TODOs left *(CIS 2450: documentation)*
- [ ] **DOCS-03**: Every Python module has a module-level docstring; every function has a docstring with Args/Returns; type hints throughout *(CIS 2450: documentation)*
- [ ] **DOCS-04**: Notebooks have markdown cells explaining each step's purpose *(CIS 2450: notebook readability)*
- [ ] **DOCS-05**: Git history shows commits from both Rahil and Ayush, atomic per task *(CIS 2450: equal contributions + version control hard requirements)*
- [ ] **DOCS-06**: `requirements.txt` pinned, `.env.example` shipped, no secrets committed
- [ ] **DOCS-07**: Tests in `tests/test_features.py`, `tests/test_graph.py`, `tests/test_models.py` — at minimum smoke tests for happy paths *(codebase quality)*
- [ ] **DOCS-08**: Both members complete CIS 2450 Project Contribution Form *(rubric §VIII)*
- [ ] **DOCS-09**: Submission to CIS 2450 Gradescope by 2026-04-30 23:59 with both group members added *(failure = teammate scored 0)*

### Presentation (PRES) — *10 pts CIS 2450*

- [ ] **PRES-01**: Slides as PDF in `presentation/slides.pdf` — no code shown *(CIS 2450 §VI explicit)*
- [ ] **PRES-02**: Recording 8-10 min in `presentation/recording.mp4` — outside window = penalty *(CIS 2450 §VI hard window)*
- [ ] **PRES-03**: Both team members deliver part of the presentation *(CIS 2450 §VI)*
- [ ] **PRES-04**: No sped-up audio (-2 pts if detected) *(CIS 2450 §VI explicit)*
- [ ] **PRES-05**: Coverage in order: objective + value (~30s), data + pivot rationale (~1m), top 3-5 EDA charts (~2m), modeling progression with comparison table (~3m), implications (~1m), limitations (~1m), live dashboard demo (~1m) *(CIS 2450 §VI required content)*
- [ ] **PRES-06**: Charts exported at ≥300 DPI; consistent fonts; clean slide template; seaborn whitegrid or custom matplotlib stylesheet *(presentation polish)*
- [ ] **PRES-07**: AI-usage disclosure slide *(CIS 2450 §2.b mandates this in deliverables, not just readme.txt)*

### Compliance Loop (COMPL)

- [ ] **COMPL-01**: `full_compliance_check` MCP tool called after every phase; output saved to `.planning/compliance/phase-N-check.md` *(rubric drift prevention)*
- [ ] **COMPL-02**: Any P0 flag returned by compliance becomes a blocker added to current phase before advancing *(quality gate)*
- [ ] **COMPL-03**: Final pre-submission compliance check on shipped artifacts *(failsafe)*

## v2 Requirements

Deferred — not in current roadmap, surfaced during planning if time permits.

### Stretch (STRETCH)

- **STRETCH-01**: Embedding-based content features (sentence-transformers) on top of TF-IDF
- **STRETCH-02**: Multi-class extension (top-3 / top-10 / 11-30 / unranked)
- **STRETCH-03**: Time-series ranking deltas (would require multi-day SERP scrape)
- **STRETCH-04**: Active-learning loop in dashboard (recommend → re-scrape → re-predict)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Backlink data | Original proposal §V — uncapturable without paid APIs; documented limitation |
| Real-time SERP scraping in dashboard | Free-tier API quota; one-time scrape only |
| Multi-language pages | English doc-sites only — keeps scraper + TF-IDF simpler |
| Regression on rank position | Per Shivani's "be specific" feedback — binary top-10 is the chosen target |
| Mobile/Lighthouse scoring | Out of feature scope to ship in 36 hours |
| Literal 50K+ rows of distinct pages | Narrowed-domain pivot per Ricky's 3/29 email; documented in DATA-04 |
| Late submissions | CIS 2450 + NETS 1500 both have zero late days |

## Traceability

Filled during roadmap creation (see ROADMAP.md). Updated as phases complete.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| DATA-05 | Phase 1 (stretch) | Pending |
| DATA-06 | Phase 1 | Pending |
| FEAT-01 | Phase 2 | Pending |
| FEAT-02 | Phase 2 | Pending |
| FEAT-03 | Phase 2 | Pending |
| FEAT-04 | Phase 2 | Pending |
| FEAT-05 | Phase 2 | Pending |
| GRAPH-01 | Phase 3 | Pending |
| GRAPH-02 | Phase 3 | Pending |
| GRAPH-03 | Phase 3 | Pending |
| GRAPH-04 | Phase 3 | Pending |
| GRAPH-05 | Phase 3 | Pending |
| EDA-01 | Phase 4 | Pending |
| EDA-02 | Phase 4 | Pending |
| EDA-03 | Phase 4 | Pending |
| EDA-04 | Phase 4 | Pending |
| EDA-05 | Phase 4 | Pending |
| EDA-06 | Phase 4 | Pending |
| EDA-07 | Phase 4 | Pending |
| EDA-08 | Phase 4 | Pending |
| MODEL-01 | Phase 5 | Pending |
| MODEL-02 | Phase 5 | Pending |
| MODEL-03 | Phase 5 | Pending |
| MODEL-04 | Phase 5 | Pending |
| MODEL-05 | Phase 5 | Pending |
| MODEL-06 | Phase 5 | Pending |
| MODEL-07 | Phase 5 | Pending |
| MODEL-08 | Phase 5 | Pending |
| MODEL-09 | Phase 5 | Pending |
| MODEL-10 | Phase 5 | Pending |
| MODEL-11 | Phase 5 | Pending |
| MODEL-12 | Phase 5 | Pending |
| MODEL-13 | Phase 5 | Pending |
| MODEL-14 | Phase 6 | Pending |
| REC-01 | Phase 6 | Pending |
| REC-02 | Phase 6 | Pending |
| DASH-01 | Phase 7 | Pending |
| DASH-02 | Phase 7 | Pending |
| DASH-03 | Phase 7 | Pending |
| DASH-04 | Phase 7 | Pending |
| DASH-05 | Phase 7 | Pending |
| DASH-06 | Phase 7 | Pending |
| DASH-07 | Phase 7 | Pending |
| DASH-08 | Phase 7 | Pending |
| DASH-09 | Phase 7 | Pending |
| DASH-10 | Phase 7 | Pending |
| NETS-01 | Phase 8 | Pending |
| NETS-02 | Phase 8 | Pending |
| NETS-03 | Phase 8 | Pending |
| DOCS-01 | Phase 9 | Pending |
| DOCS-02 | Phase 9 | Pending |
| DOCS-03 | Phase 9 (audit) | Pending |
| DOCS-04 | Phase 9 (audit) | Pending |
| DOCS-05 | Continuous | Pending |
| DOCS-06 | Phase 9 | Pending |
| DOCS-07 | Phase 9 | Pending |
| DOCS-08 | Phase 10 | Pending |
| DOCS-09 | Phase 10 | Pending |
| PRES-01 | Phase 10 | Pending |
| PRES-02 | Phase 10 | Pending |
| PRES-03 | Phase 10 | Pending |
| PRES-04 | Phase 10 | Pending |
| PRES-05 | Phase 10 | Pending |
| PRES-06 | Phase 10 | Pending |
| PRES-07 | Phase 10 | Pending |
| COMPL-01 | All phases | Pending |
| COMPL-02 | All phases | Pending |
| COMPL-03 | Phase 10 | Pending |

**Coverage:**
- v1 requirements: 71 total
- Mapped to phases: 71
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-29*
*Last updated: 2026-04-29 after initial definition (compliance-grounded against CIS 2450 + NETS 1500 rubrics + TA email feedback)*
