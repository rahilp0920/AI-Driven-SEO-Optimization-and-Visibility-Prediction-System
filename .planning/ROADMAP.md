# Roadmap: AI-Driven SEO Ranking Predictor

**Created:** 2026-04-29
**Total phases:** 10
**Hard deadlines:**
- **NETS 1500 (85 pts)** — Phases 1-8 ship by **2026-04-29 23:59** (~10 hours from now)
- **CIS 2450 (143 pts)** — Phases 9-10 (polish + presentation) ship by **2026-04-30 23:59**

**Compliance loop:** Every phase ends with `mcp__project-compliance__full_compliance_check` against the current state. Output → `.planning/compliance/phase-N-check.md`. Any P0 flag is added to the *current* phase as a blocker, not deferred.

**Granularity:** Fine. **Mode:** YOLO. **Parallelization:** Sequential (phases have hard data dependencies).

---

## Track 1 — NETS 1500 ship-tonight track (Phases 1-8)

### Phase 1 — Data Layer

**Goal:** Two distinct data sources flowing end-to-end. Scraped doc-site HTML in `data/raw/` + SERP rankings in `data/interim/serp.csv`. ~1500 pages with `is_top_10` labels. `data/README.md` explains where data came from + quotes Ricky's pivot rationale verbatim.

**Owner:** Rahil (primary)

**Requirements covered:** DATA-01, DATA-02, DATA-03, DATA-04, DATA-06 *(DATA-05 stretch if SERP quota allows)*

**Success criteria:**
1. `python -m src.scraping.doc_scraper --domain docs.python.org --limit 300` produces HTML files in `data/raw/`
2. `python -m src.scraping.serp_client --queries data/interim/queries.csv` produces SERP top-10 rankings in `data/interim/serp.csv`
3. ~1500 total pages scraped across ≥5 dev-doc domains
4. `data/README.md` exists, names every domain, quotes Ricky's 3/29 email verbatim, includes scrape date + rate-limit policy
5. Robots.txt compliance verified for each domain (logged)

**Compliance gate:** `full_compliance_check` confirms 2 distinct sources documented + pivot rationale present.

---

### Phase 2 — Feature Engineering

**Goal:** Per-page feature matrix in `data/processed/features.csv` with ≥10 numeric columns, target column `is_top_10`.

**Owner:** Rahil (primary)

**Requirements covered:** FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05

**Success criteria:**
1. `src/features/content_features.py` extracts TF-IDF vectors (top-100 dim or sparse), keyword density, Flesch readability, text length
2. `src/features/metadata_features.py` extracts title length, meta description length/presence, keyword-in-title flag
3. `src/features/structural_features.py` extracts H1/H2/H3 counts, link counts (internal/external), image count, alt-text coverage
4. Final `features.csv` has ≥10 numeric columns + `is_top_10` target
5. Module + function docstrings + type hints throughout

**Compliance gate:** `full_compliance_check` confirms ≥7-10 columns, AI-usage notes captured for any boilerplate generated.

---

### Phase 3 — Graph Layer (also satisfies NETS 1500's Graph Algorithms requirement)

**Goal:** Directed link graph across scraped pages with PageRank + HITS + degree + clustering features merged into the feature matrix.

**Owner:** Rahil (primary)

**Requirements covered:** GRAPH-01, GRAPH-02, GRAPH-03, GRAPH-04, GRAPH-05

**Success criteria:**
1. `src/graph/build_graph.py` constructs `nx.DiGraph` from internal links between scraped pages
2. `src/graph/graph_features.py` computes PageRank, HITS hub + authority, in-degree, out-degree, clustering coefficient — merges into `data/processed/features.csv`
3. `notebooks/02_graph_analysis.ipynb` shows PageRank distribution + degree distribution + summary stats
4. Graph features clearly labeled in feature matrix (e.g., `pagerank`, `hits_hub`, `hits_authority`, `in_degree`, `out_degree`, `clustering`)

**Compliance gate:** `full_compliance_check` confirms graph algorithms are concrete (Shivani feedback resolved).

---

### Phase 4 — EDA + Rubric Charts

**Goal:** 5 high-DPI charts in `notebooks/01_eda.ipynb` exported to `assets/charts/`, each with a markdown caption tying it to a downstream modeling decision.

**Owner:** Ayush (primary)

**Requirements covered:** EDA-01 through EDA-08

**Success criteria:**
1. `assets/charts/01_class_balance.png` — class balance plot, caption justifies F1/ROC-AUC choice
2. `assets/charts/02_feature_distributions.png` — per-feature distributions with top-10 hue
3. `assets/charts/03_correlation_heatmap.png` — correlation heatmap of features vs target
4. `assets/charts/04_pagerank_distribution.png` — PageRank histogram, log-scale x
5. `assets/charts/05_tfidf_top_terms.png` — TF-IDF top terms by class
6. All charts ≥300 DPI, seaborn whitegrid or custom matplotlib stylesheet
7. Each chart has markdown caption: "this informed decision X in modeling"
8. Notebook has markdown cells explaining each step

**Compliance gate:** `full_compliance_check` confirms 3-5 charts requirement met + each chart informs modeling.

---

### Phase 5 — Modeling Sweep (8 models)

**Goal:** All 8 models trained, tuned, evaluated; comparison table; clear winner with defended rationale.

**Owner:** Rahil (LR, RF, GB, XGBoost, LightGBM, CatBoost) + Ayush (SVM-RBF, MLP, evaluator)

**Requirements covered:** MODEL-01 through MODEL-13

**Success criteria:**
1. `src/models/baseline.py` (LR), `src/models/tree_models.py` (RF + GB), `src/models/boosting.py` (XGBoost + LightGBM + CatBoost), `src/models/kernel.py` (SVM-RBF), `src/models/neural.py` (PyTorch MLP) all callable + each writes a serialized model + metrics JSON
2. `src/models/evaluate.py` is the single evaluator returning precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix
3. All models use StratifiedKFold CV (justified by EDA-02 class balance) + RandomizedSearchCV with documented `n_iter` and `cv`
4. PyTorch MLP trained in Colab; checkpoint exported via the rubric §VI snippet pattern; checkpoint pulled into local repo at `models/mlp_checkpoint.pt`
5. `MODELING_DECISIONS.md` has a section per model: why-tried, search space, CV, metrics, kept/dropped + tied to a prior result
6. `notebooks/03_model_comparison.ipynb` ends with: comparison table, clear winner, defending paragraph

**Compliance gate:** `full_compliance_check` confirms baseline-to-advanced progression + appropriate metrics for imbalanced data + no silent modeling choices.

---

### Phase 6 — Recommendations + SHAP

**Goal:** Recommendation engine produces actionable per-page suggestions; SHAP analysis on the winning model.

**Owner:** Ayush (primary)

**Requirements covered:** REC-01, REC-02, MODEL-14

**Success criteria:**
1. `src/recommendations/recommend.py` takes a feature row + model and returns ≥3 actionable text suggestions with concrete numbers ("title is 80 chars, shorten to 50-60", "add 2 H2 headers", "add alt text to 4 of 7 images")
2. SHAP analysis run on the winning model from Phase 5; SHAP summary plot saved to `assets/charts/06_shap_summary.png` ≥300 DPI
3. SHAP values can be queried per-prediction (for dashboard's per-page breakdown)
4. Tests in `tests/test_features.py` cover recommendation edge cases (e.g., already-good page returns "no changes recommended")

**Compliance gate:** `full_compliance_check` confirms recommendations are concrete (not generic).

---

### Phase 7 — Streamlit Dashboard

**Goal:** Polished dashboard at `src/dashboard/app.py` with 4 sidebar sections, what-if simulator, demo data fallback. Comparable polish to homework dashboards.

**Owner:** Ayush (frontend + styling) + Rahil (backend wiring)

**Requirements covered:** DASH-01 through DASH-10

**Success criteria:**
1. `streamlit run src/dashboard/app.py` boots without error on a clean clone after `pip install -r requirements.txt`
2. Sidebar nav: Predict / Recommendations / What-if / About
3. URL input → live scraping → feature extraction → prediction with gauge/progress bar
4. SHAP top-5 features panel renders for the input page
5. What-if sliders modify features in place, prediction updates live, delta shown explicitly
6. Custom CSS via `st.markdown(unsafe_allow_html=True)` — no default Streamlit look
7. Demo data fallback: if scraping fails (e.g., network), dashboard loads a pre-baked example seamlessly
8. PyTorch MLP loaded via the rubric §VI checkpoint pattern (proves we hit that requirement)
9. About tab includes AI-usage disclosure

**Compliance gate:** `full_compliance_check` confirms dashboard is interactive + showcases full capabilities + AI-usage in dashboard (not just readme).

---

### Phase 8 — NETS 1500 Submission Deliverables (TONIGHT 2026-04-29 23:59)

**Goal:** Ship to NETS 1500 Gradescope.

**Owner:** Rahil (readme.txt) + Ayush (USER_MANUAL.md)

**Requirements covered:** NETS-01, NETS-02, NETS-03

**Success criteria:**
1. `readme.txt` (1 page max) contains: project name, 4-5 sentence description, NETS categories used (Information Networks WWW + Information Retrieval + Graph algorithms), work breakdown (Rahil/Ayush split), concrete AI usage details with examples
2. `USER_MANUAL.md` contains: how to install (pip install, env vars), how to run (streamlit command), screenshots of every dashboard feature with captions
3. Both group members added to the Gradescope submission
4. Submitted by 2026-04-29 23:59 — confirm submission timestamp

**Compliance gate:** `full_compliance_check` against NETS 1500 rubric returns no MISSING items. Both members confirm Gradescope group-add.

---

## Track 2 — CIS 2450 ship-tomorrow track (Phases 9-10)

### Phase 9 — CIS 2450 Codebase Polish

**Goal:** Codebase grade-ready (83 pts). README maps every rubric line to a file path. MODELING_DECISIONS.md has no TODOs. Tests pass. Pinned requirements.txt. Atomic commits from both members.

**Owner:** Rahil + Ayush (split — Rahil README + tests, Ayush MODELING_DECISIONS cleanup)

**Requirements covered:** DOCS-01, DOCS-02, DOCS-03, DOCS-04, DOCS-05 (continuous), DOCS-06, DOCS-07

**Success criteria:**
1. `README.md` opens with project blurb + has explicit "Rubric → File Map" table mapping each CIS 2450 rubric requirement to the file path that satisfies it
2. `README.md` includes pivot rationale (Ricky 3/29 email quote)
3. `MODELING_DECISIONS.md` has no TODOs, all 8 models documented with kept/dropped reasoning
4. Module docstrings + function docstrings (Args/Returns) + type hints audit complete (DOCS-03 sweep across all `src/`)
5. Notebooks have markdown cells explaining each step (DOCS-04 audit)
6. `requirements.txt` pinned versions, `.env.example` shipped, no secrets in git
7. `pytest` runs green (DOCS-07: smoke tests for features, graph, models)
8. Git history shows commits attributed to both `Rahil Patel` and `Ayush Tripathi` (DOCS-05 verification)

**Compliance gate:** `full_compliance_check` against CIS 2450 codebase rubric returns no MISSING items.

---

### Phase 10 — Presentation + Final Submission (TOMORROW 2026-04-30 23:59)

**Goal:** Slides + 8-10 min recording. Both Gradescopes finalized. Contribution form filled.

**Owner:** Ayush (slides + recording lead) + Rahil (review + record together)

**Requirements covered:** PRES-01 through PRES-07, DOCS-08, DOCS-09, COMPL-03

**Success criteria:**
1. `presentation/slides.pdf` — no code on slides, ≥300 DPI charts, consistent font, clean template
2. `presentation/recording.mp4` — 8-10 min hard window (verify duration), both members audibly speak, no sped-up audio
3. Coverage in order: objective + value (~30s), data + pivot rationale (~1m), top 3-5 EDA charts (~2m), modeling progression with comparison table (~3m), implications (~1m), limitations (~1m), live dashboard demo (~1m)
4. AI-usage disclosure slide present
5. Both members fill out CIS 2450 Project Contribution Form
6. CIS 2450 Gradescope submission with both group members added — submitted by 2026-04-30 23:59
7. Final `full_compliance_check` against shipped artifacts returns no MISSING items

**Compliance gate:** `full_compliance_check` against full CIS 2450 + NETS 1500 rubric. P0 flags = block submission, fix immediately.

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| SERP API quota exhaustion mid-scrape | HIGH | Free-tier headroom check in Phase 1; Brave Search as fallback; cache aggressively |
| Robots.txt disallows critical domain | HIGH | Pre-check in Phase 1; swap domain if blocked |
| 1500-page count drops below ~1000 due to errors | MED | Pad target to 1800 in scraper; document final count in data/README.md |
| 50K-row rubric preference unmet | MED (documented) | Ricky's pivot email + DATA-05 (page,query) stretch + breadth/depth defense |
| Compliance MCP flags CIS rubric gap | HIGH | P0 blocker, fix before advancing per COMPL-02 |
| Presentation runs >10min or <8min | HIGH (penalty) | Rehearse with timer; cut content > over-cut audio |
| One member's git commits missing | HIGH (rubric) | Track per-phase commit attribution in `.planning/compliance/contributions.md` |
| Streamlit dashboard breaks during live demo | MED | DASH-08 demo data fallback |
| MLP doesn't beat boosting | LOW (acceptable) | Rubric values *capability* (Colab + checkpoint loading), not winning |

## Compliance Gate Protocol

After each phase:
1. Call `mcp__project-compliance__full_compliance_check` with current project state
2. Save output to `.planning/compliance/phase-N-check.md`
3. Walk through CIS 2450 + NETS 1500 + proposal-coverage sections
4. Any item marked MISSING or 🟡 partial that's in the *current* phase's scope = P0 blocker, fix before advancing
5. Items missing from *future* phases = expected, no action
6. Add commit message: `chore(compliance): phase N gate passed`

## Coverage Validation

| REQ-ID Block | Total | Covered |
|--------------|-------|---------|
| DATA-01..06 | 6 | 6 (Phase 1) |
| FEAT-01..05 | 5 | 5 (Phase 2) |
| GRAPH-01..05 | 5 | 5 (Phase 3) |
| EDA-01..08 | 8 | 8 (Phase 4) |
| MODEL-01..13 | 13 | 13 (Phase 5) |
| MODEL-14 + REC-01..02 | 3 | 3 (Phase 6) |
| DASH-01..10 | 10 | 10 (Phase 7) |
| NETS-01..03 | 3 | 3 (Phase 8) |
| DOCS-01..07 | 7 | 7 (Phase 9) |
| DOCS-08..09 + PRES-01..07 + COMPL-03 | 10 | 10 (Phase 10) |
| COMPL-01..02 | 2 | 2 (Continuous) |
| **Total** | **72** | **72 ✓** |

---
*Roadmap created: 2026-04-29*
