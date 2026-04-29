# Roadmap (token-optimized — 3 phases, no planning loop)

**Restructured:** 2026-04-29. Mode: build-fast, no `/gsd-plan-phase` spawns. Compliance MCP is the only verification gate.

**Deadlines:** NETS 1500 — 2026-04-29 23:59 (Phases 1-2). CIS 2450 — 2026-04-30 23:59 (Phase 3).

---

## Phase 1 — BUILD (single shot, all technical work)

**Owner:** Rahil primary, Ayush models/EDA/dashboard frontend.

**Deliverables (one commit per file group, no inter-file iteration unless broken):**

1. `src/scraping/doc_scraper.py` + `src/scraping/serp_client.py` — async scraper, robots-aware, ~1500 pages across ≥5 dev-doc domains. Output: `data/raw/*.html`, `data/interim/serp.csv`.
2. `data/README.md` — domains list, Ricky 3/29 quote verbatim, scrape date, rate-limit policy.
3. `src/features/{content,metadata,structural}_features.py` → `data/processed/features.csv` (≥10 numeric cols + `is_top_10`).
4. `src/graph/{build_graph,graph_features}.py` — DiGraph + PageRank/HITS/degree/clustering merged into features.csv.
5. `notebooks/01_eda.ipynb` → 5 charts (`assets/charts/01-05`, ≥300 DPI), each with caption tying to a modeling decision.
6. `notebooks/02_graph_analysis.ipynb` → PageRank + degree distributions.
7. **Models (4 — LogReg, RF, XGBoost, MLP):** `src/models/{baseline,tree_models,boosting,neural}.py`, single `src/models/evaluate.py` returning P/R/F1/ROC-AUC/PR-AUC/CM. StratifiedKFold + RandomizedSearchCV. MLP trained in Colab, checkpoint at `models/mlp_checkpoint.pt`.
8. `notebooks/03_model_comparison.ipynb` → comparison table + winner + defending paragraph.
9. `src/recommendations/recommend.py` — ≥3 actionable concrete suggestions per page. SHAP summary → `assets/charts/06_shap_summary.png`.
10. `src/dashboard/app.py` — sidebar (Predict / Recs / What-if / About), live URL → predict, what-if sliders, custom CSS, demo data fallback, MLP loaded via rubric §VI pattern, AI-usage in About.
11. Module + function docstrings + type hints throughout.

**Gate:** `mcp__project-compliance__full_compliance_check` → save to `.planning/compliance/phase-1-check.md`. Any P0 = fix before moving to Phase 2.

---

## Phase 2 — NETS SHIP (TONIGHT)

1. `readme.txt` (1pg): name, 4-5 sentence desc, NETS categories (Info Networks WWW + IR + Graph algorithms), Rahil/Ayush split, AI usage examples.
2. `USER_MANUAL.md`: install + run + screenshots of every dashboard feature with captions.
3. Both members on Gradescope group; submit by 23:59.

**Gate:** `full_compliance_check` against NETS rubric.

---

## Phase 3 — CIS POLISH + PRESENT (TOMORROW)

1. `README.md` — blurb + Rubric→File Map table + Ricky pivot quote.
2. `MODELING_DECISIONS.md` — 4 kept models (why-tried, search space, CV, metrics, kept) + 4 cut models (deadline-scoped justification).
3. Docstring/type-hint sweep across `src/`. Pinned `requirements.txt`. `.env.example`. `pytest` green (`tests/test_{features,graph,models}.py`).
4. Verify both `Rahil Patel` + `Ayush Tripathi` commits in `git log`.
5. `presentation/slides.pdf` (no code on slides, ≥300 DPI charts) + `presentation/recording.mp4` (8-10 min hard, both speak, no sped-up audio). AI-usage disclosure slide.
6. CIS 2450 Project Contribution Form filled. Gradescope submit by 2026-04-30 23:59.

**Gate:** Final `full_compliance_check` against full CIS + NETS rubric. P0 = block submit, fix immediately.

---

## Risk Register (abridged)

| Risk | Mitigation |
|------|------------|
| SERP quota exhaustion | Brave fallback, cache aggressively |
| Robots disallows | Pre-check, swap domain |
| <1000 pages scraped | Pad target to 1800 |
| 4-model sweep questioned | MODELING_DECISIONS.md cites deadline + literature |
| Compliance MCP P0 flag | Block phase, fix |
| Recording >10min/<8min | Rehearse with timer |
| One member's commits missing | Track in `.planning/compliance/contributions.md` |
| Dashboard breaks live | Demo data fallback (DASH-08) |

## Coverage

All 72 REQ-IDs map to Phase 1 (DATA/FEAT/GRAPH/EDA/MODEL/REC/DASH = 63), Phase 2 (NETS-01..03 = 3), Phase 3 (DOCS/PRES/COMPL-03 = 17 incl. continuous). COMPL-01..02 are continuous gates.

---
*Restructured: 2026-04-29 — 10→3 phases for token budget. Skip `/gsd-plan-phase` and `/gsd-execute-phase` ceremony; write code directly per file list above.*
