# Phase 1 — Compliance Gate Report

**Date:** 2026-04-29
**Hours to NETS 1500 deadline:** ~7-8h
**Hours to CIS 2450 deadline:** ~31-32h
**Verdict:** Phase 1 BUILD code is **PASS** for what Phase 1 promised (technical scaffolding). But several P0 items for Phase 2 NETS Ship are **imminent** and must be closed tonight.

## Proposal coverage (Original proposal vs current state)

| Section | Status | Evidence / Action |
|---------|--------|-------------------|
| Group members & responsibilities | ✅ covered (code) / 🟡 (git attribution) | Code split matches proposal. Only Rahil git commits so far — Ayush attribution required before CIS ship. |
| Data sources (2 distinct) | ✅ documented / 🟡 not yet acquired | Scraped dev-doc HTML + Brave SERP. Pivot rationale verbatim in `data/README.md`. ~1500 pages not yet scraped. |
| Objective & value (specific target) | ✅ covered | "Top-10 SERP for query derived from `<title>`" — directly addresses Shivani's "be specific" feedback. |
| Modeling plan | ✅ covered with documented deviation | Originally LR → RF → XGBoost/LightGBM. Current: **LR + RF + XGBoost + MLP**; LightGBM/CatBoost/GBM/SVM cut, documented in `MODELING_DECISIONS.md`. **The change from the proposal MUST be disclosed in `readme.txt`** (NETS 1500 explicit rule). |
| Feature engineering (content/meta/structural/technical) | ✅ covered | All 4 categories present (`src/features/{content,metadata,structural}_features.py` + graph features). |
| Recommendation system + simulation step | ✅ covered (bonus) | `src/recommendations/recommend.py` + dashboard what-if tab. |
| Challenges acknowledgment | 🟡 partial | Risk register in `ROADMAP.md`. Surface in `readme.txt`. |

## Shivani's feedback (NETS 1500)

| Item | Status | Where addressed |
|------|--------|-----------------|
| Specific prediction target | ✅ | data/README, README, dashboard About: "top-10 SERP for derived query". |
| Specific data sources | ✅ | data/README.md domain table, rate-limit + robots policy. |
| Specific graph algorithms | ✅ | PageRank (α=0.85), HITS hub+authority, clustering coefficient — explicit in `notebooks/02_graph_analysis.ipynb`. |
| Backup plan if scraping hard | ✅ | Pivot documented (Ricky 3/29 email quote verbatim). |

## Ricky's feedback (CIS 2450)

| Item | Status | Note |
|------|--------|------|
| Validate data volume early | 🟡 → run scrape NOW | Pivot to ~1500 documented; actual volume only validates once scrape runs. |
| Have narrower-domain backup plan | ✅ | Executed — ≥5 dev-doc domains, 1500-page total, sanctioned. |
| Intermediate check-in scheduled (5 pts) | ✅ | Email thread shows meeting scheduled; **confirm attendance**. |

## CIS 2450 grading (143 pts) — tentative scores

| Component | Pts | Tentative | Gaps |
|-----------|-----|-----------|------|
| Proposal | 5 | **5/5** | Done. |
| Intermediate Check-in | 5 | **5/5** if attended | Confirm meeting attendance. |
| **Codebase** | **83** | **70-78/83** | See sub-scores below. |
| Dashboard Demo | 20 | **17-20/20** | Polish solid; needs real (not demo) data flow during demo. |
| Presentation + Recording | 10 | **0/10** today | Not yet built. Phase 3 work. |
| **Total est. ceiling today** | **143** | **~97-108/143** | Closes to ~135+ once Phase 2/3 deliverables ship. |

### Codebase sub-scores

- **Modularity** — full marks; clean `src/{scraping,features,graph,models,recommendations,dashboard}` tree.
- **Documentation** — partial; module + function docstrings + type hints throughout, but `README.md` needs rubric→file map.
- **Readability** — full; no dead code, consistent naming, `from __future__ import annotations` everywhere.
- **Equal contributions** — **P0 GAP**: 8 commits, all by Rahil Patel. CIS 2450 hard requirement; Ayush MUST commit at least 2-3 attributed commits before tomorrow.
- **Version control** — atomic commits, no dump commit. Good.
- **EDA** — full; 5 rubric charts with markdown captions tying each to a modeling decision.
- **Modeling** — full; baseline → bagging → boosting → neural progression, F1/ROC-AUC/PR-AUC for imbalance, RandomizedSearchCV documented per model.
- **Interactive demo** — full; Streamlit boots with demo fallback.

## CIS 2450 hard requirements (failing any = automatic 0)

| Requirement | Status |
|-------------|--------|
| Free, public, legal data | ✅ robots.txt-aware crawler, public dev docs only. |
| ≥2 distinct sources | ✅ scraped HTML + Brave SERP. |
| 50K rows OR documented narrower-domain pivot | ✅ pivot documented verbatim per Ricky 3/29. |
| 7-10+ feature columns | ✅ 22 core + 50 TF-IDF + 6 graph = ~78 numeric columns. |
| AI usage documented in detail | 🟡 in dashboard About; **must also be in `readme.txt` and a presentation slide**. |

## NETS 1500 grading (85 pts) — tentative

| Component | Status |
|-----------|--------|
| Code (well-organized, runnable) | ✅ |
| User manual with screenshots | ❌ **P0 — missing** (`USER_MANUAL.md`) |
| readme.txt (1 page mandatory) | ❌ **P0 — missing** |
| Course-topic integration (≥2 categories) | ✅ Info Networks WWW + IR + Graph algorithms (3 of 7). |
| Both members on Gradescope group | 🟡 verify before submission. |
| Submitted by 2026-04-29 23:59 | 🟡 pending. |

---

## PRIORITIZED PUNCH LIST

### P0 — block NETS ship tonight (2026-04-29 23:59)
1. **`readme.txt`** — 1 page. Project name + 4-5 sentence description + NETS categories used (Info Networks WWW, IR, Graph algorithms) + work breakdown + concrete AI usage details + **change-from-proposal note** (LightGBM/CatBoost/SVM cut, MLP added — required by NETS 1500 explicit rule).
2. **`USER_MANUAL.md`** with screenshots. Requires the dashboard running on real data → run scrape first.
3. **Actual data scrape** — `python -m src.scraping.doc_scraper` for ≥5 domains × ~300 each. Then `serp_client build-queries` + `fetch`. Then `build_features` + `graph_features`.
4. **Train at least one model** so the dashboard predicts, not falls back to demo. XGBoost is cheapest end-to-end.
5. **Verify both members on Gradescope group** before submission.

### P0 — block CIS ship tomorrow (2026-04-30 23:59)
1. **Ayush git commits** — at least 2-3 attributed commits. Switch via `git config user.name "Ayush Tripathi"` + `git config user.email "tripath1@seas.upenn.edu"` per CLAUDE.md.
2. **`README.md`** with explicit rubric → file map table.
3. **`MODELING_DECISIONS.md`** — fill in the 4 kept-model sections with actual measured metrics + best params + cv_f1.
4. **pytest** — green suite covering features/graph/models (DOCS-07).
5. **`presentation/slides.pdf`** — no code on slides, ≥300 DPI charts, AI-usage disclosure slide, both Rahil and Ayush named.
6. **`presentation/recording.mp4`** — 8-10 min hard window (PENALTY if outside), both speak audibly, no sped-up audio (-2 pts if detected).
7. **CIS 2450 Project Contribution Form** — both members fill out.

### P1 — shouldn't block, will help
1. Push to `origin/main` for graders who check the GitHub repo.
2. Surface risk register / limitations in `readme.txt` + slides.

### P2 — nice
1. Pin `joblib` and `scipy` explicitly in `requirements.txt` (currently transitive via sklearn; pinning makes the install deterministic).

---

## Phase 1 gate verdict

**Phase 1 BUILD passes** — the codebase scaffolding promised by Phase 1's success criteria is complete:
- Two distinct data sources wired (scraper + SERP).
- Pipeline produces `features.csv` with ≥10 columns once data flows.
- Graph algorithms concrete (PageRank, HITS, clustering).
- 4-model sweep code complete with shared evaluator.
- Dashboard interactive with live URL + recommendations + what-if + About.

No P0 gaps **inside Phase 1's scope**. Proceed to Phase 2 immediately — NETS deadline is in ≤8 hours.

*Next: scrape data → train one model → write `readme.txt` → run dashboard → screenshot → write `USER_MANUAL.md` → submit to Gradescope.*
