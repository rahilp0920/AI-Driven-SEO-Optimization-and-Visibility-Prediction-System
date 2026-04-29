# AI-Driven SEO Ranking Predictor & Recommendation System

## What This Is

A binary classifier that predicts whether a developer documentation page (docs.python.org, react.dev, Stripe docs, MDN, etc.) appears in the top-10 Google SERP results for its primary topic query, plus a recommendation engine that turns model output into actionable page-improvement suggestions. Built as the joint final project for two Penn courses: **CIS 2450 (Big Data Analytics, 143 pts, due 2026-04-30 23:59)** and **NETS 1500 (HW5 Project, 85 pts, due 2026-04-29 23:59)**. Optimized hard against the CIS 2450 rubric (90% of effort); NETS 1500 ships at minimum-viable bar.

## Core Value

**Score maximally on the CIS 2450 rubric — codebase polish (83 pts), dashboard polish (20 pts), modeling depth, and presentation craft — while satisfying NETS 1500's WWW + IR + Graph-Algorithms course-topic integration with a single shared codebase.**

If anything else fails, the rubric checkboxes must still be hit. The compliance MCP (`project-compliance.full_compliance_check`) is treated as ground truth; any gap it flags after a phase is a P0 blocker.

## Requirements

### Validated

(None yet — proposal scored 5/5 by Ricky on 2026-03-29 and Shivani on 2026-04-11 acknowledged the project; everything implementation-side is unbuilt.)

### Active

<!-- See REQUIREMENTS.md for the full atomic list. Summary by deliverable: -->

- [ ] **Data layer** — 2 distinct sources (scraped doc-site HTML + SERP API), ~1500 pages narrowed-domain pivot per Ricky's 3/29 email, 7-10+ feature columns
- [ ] **Feature engineering** — content (TF-IDF, keyword density, readability, length), metadata, structural, graph features
- [ ] **Graph layer** — NetworkX directed link graph, PageRank + HITS + in/out-degree + clustering coefficient
- [ ] **EDA** — 3-5 high-DPI charts with markdown captions tying each to a modeling decision
- [ ] **Modeling sweep** — 8 models (LR, RF, GB, XGBoost, LightGBM, CatBoost, SVM-RBF, PyTorch MLP) with StratifiedKFold + RandomizedSearchCV, unified evaluator, MODELING_DECISIONS.md log per model, comparison table, SHAP on winner
- [ ] **Recommendation system** — feature → suggestion mapping
- [ ] **Streamlit dashboard** — URL input, probability gauge, SHAP panel, recommendations, what-if simulator, custom CSS, demo fallback
- [ ] **NETS 1500 deliverables** — readme.txt (1 page) + USER_MANUAL.md with screenshots, ship to Gradescope by 2026-04-29 23:59
- [ ] **CIS 2450 polish** — README.md mapping every rubric item to file path, MODELING_DECISIONS.md cleanup, codebase review, AI-usage documentation in dashboard + presentation
- [ ] **Presentation** — 8-10 min hard window, both members speak, no code on slides, no sped-up audio, ≥300 DPI charts, PDF slides + recording.mp4

### Out of Scope

- **Backlink data** — original proposal acknowledged this is uncapturable without paid APIs; documented limitation
- **Real-time SERP scraping** — covered by SerpApi/Brave free tier on a one-time scrape, not live
- **Multi-language pages** — English documentation only
- **Beyond top-10 ranking granularity** — binary classification, not regression on rank position (per Shivani's "be specific about target" feedback)
- **Mobile-friendly Lighthouse scoring** — not part of feature set; out of scope to keep scope shippable
- **50K+ rows of literal pages** — narrowed-domain pivot is documented (Ricky's 3/29 email). Stretch: augment row count via (page, query) pairs to defend volume on the CIS 2450 rubric

## Context

**Source-of-truth grounding.** Two external rubrics + two TA emails govern this project; both are stored verbatim in the `project-compliance` MCP server. Any disagreement between Claude's judgment and the rubric is resolved by the rubric. The MCP tool `full_compliance_check` is called:
- Before any planning artifact is written (already done — informed this PROJECT.md)
- After every phase completes (gate to advance)
- Before final submission

**Pivot rationale (must appear in README.md and data/README.md verbatim):**
> Per CIS 2450 TA Ricky Gong's email of 2026-03-29: *"The main risk here is data collection scale. Acquiring 50K+ labeled webpages with corresponding SERP ranking data through a free-tier API could be slow and rate-limited. I'd prioritize getting the data pipeline working first, validate your data volume early, and have a backup plan (e.g., focusing on a specific domain or topic area) in case the full dataset is hard to collect."*
> We executed this guidance by narrowing scope to ~1500 developer documentation pages from docs.python.org, react.dev, Stripe docs, MDN, and similar high-authority dev-doc domains.

**Shivani's three flags (NETS 1500, 2026-04-11) — all addressed:**
1. Specific prediction target → binary top-10 SERP, query from `<title>`
2. Specific data sources → enumerated dev-doc domains
3. Specific graph algorithms → PageRank, HITS hub/authority, in/out-degree, clustering coefficient

**Course-topic integration (NETS 1500 floor):**
- Information Networks (WWW) — by construction
- Document Search / Information Retrieval — TF-IDF + ranking target
- Graph algorithms — PageRank + HITS over scraped link graph

**AI usage (must be documented in BOTH readme.txt AND dashboard/presentation per CIS 2450 rubric):** Claude Code drove scaffolding, scraper boilerplate, feature-extraction utilities, EDA chart code, and modeling boilerplate. Every AI-generated artifact gets a hand-validation note. Specific examples logged in `readme.txt` and surfaced via an "AI Usage" panel in the dashboard's About section + a presentation slide.

**Codebase structure (non-negotiable, weighted heavily by CIS 2450 codebase rubric — 83 pts):**
```
/
├── README.md                  # CIS 2450 main; rubric-to-file map
├── readme.txt                 # NETS 1500 1-pager
├── USER_MANUAL.md             # NETS 1500 user manual + screenshots
├── MODELING_DECISIONS.md      # Living log per model
├── requirements.txt
├── .env.example
├── data/{raw,interim,processed}/, data/README.md
├── src/
│   ├── scraping/{doc_scraper.py, serp_client.py}
│   ├── features/{content_features.py, metadata_features.py, structural_features.py}
│   ├── graph/{build_graph.py, graph_features.py}
│   ├── models/{baseline.py, tree_models.py, boosting.py, kernel.py, neural.py, evaluate.py}
│   ├── recommendations/recommend.py
│   └── dashboard/{app.py, components/, styles.py}
├── notebooks/{01_eda, 02_graph_analysis, 03_model_comparison}.ipynb
├── tests/{test_features, test_graph, test_models}.py
└── presentation/{slides.pdf, recording.mp4}
```
Module-level docstrings + per-function docstrings (Args/Returns) + type hints throughout.

**Work breakdown (commits attributed accordingly):**
- **Rahil Patel** (`rahilp0920@gmail.com` / `rahilp07@seas.upenn.edu`) — `src/scraping/`, `src/features/`, `src/graph/`, `src/models/{baseline,tree_models,boosting}.py`, dashboard backend
- **Ayush Tripathi** (`tripath1@seas.upenn.edu`) — `notebooks/01_eda.ipynb`, `src/models/{kernel,neural}.py`, `notebooks/03_model_comparison.ipynb`, `src/models/evaluate.py`, `src/recommendations/`, dashboard frontend + styling, slide deck

**Intermediate check-in (CIS 2450, 5 pts):** Confirmed via Ricky's 2026-04-10 13:34 email providing meet.google.com link. Both members must verify they attended; loss = -5 pts per absent member.

## Constraints

- **Deadline (NETS 1500):** 2026-04-29 23:59 — *tonight*. Hard cutoff, no late days.
- **Deadline (CIS 2450):** 2026-04-30 23:59 — *tomorrow night*. Hard cutoff, no late days.
- **Tech stack:** Python 3.11+, scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch (shallow MLP), NetworkX, Streamlit, SHAP, BeautifulSoup/lxml, async HTTP (aiohttp/httpx), SerpApi or Brave Search free tier.
- **Data legality:** Public dev-doc sites only. Respect `robots.txt`. Respect rate limits. PA legal obligation per CIS 2450 rubric — failure = automatic 0.
- **Cost:** SERP API on free tier only. Course doesn't reimburse paid API calls.
- **Presentation length:** 8-10 minutes — outside this window = penalty. No sped-up audio (-2 pts). No code on slides.
- **Both members must speak in presentation.**
- **Compliance loop:** Run `full_compliance_check` MCP tool after every phase. Any flagged gap = P0 blocker before advancing.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Narrow domain to ~1500 dev-doc pages (vs proposal's 50K) | Ricky's 3/29 email explicitly sanctions narrower-domain pivot when full-scale collection is rate-limited | — Pending; cite verbatim in README.md and data/README.md |
| Binary top-10 classification (vs regression on rank) | Shivani flagged "predict how well it ranks" as too vague; binary target is specific, measurable, easy to defend | — Pending; addressed in proposal-pivot section |
| 8-model sweep (CIS rubric depth) | CIS 2450 codebase weight 83/143 rewards modeling breadth + documented justification | — Pending; MODELING_DECISIONS.md log enforces no silent choices |
| Streamlit (vs Dash) | Faster to ship a polished dashboard in <2 days; SHAP integrates cleanly | — Pending |
| Skip parallel domain-research agents during planning | Rubric IS the source of truth; external SEO best-practices research has no rubric weight and burns time we don't have | ✓ Saved ~30min of agent runtime |
| Stretch: (page, query) pairs to inflate row count | CIS rubric prefers 50K+ rows. 1500 pages × ~7 queries each ≈ 10K rows — better defense alongside Ricky's email | — Stretch in Phase 1 |
| MCP compliance gate after every phase | The rubric is external; without a programmatic check we drift | ✓ Built into ROADMAP.md as a per-phase gate |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Run `full_compliance_check` — capture output to `.planning/compliance/phase-N-check.md`
2. Any P0 flag → fix before advancing; log fix in Key Decisions
3. Requirements invalidated? → Move to Out of Scope with reason
4. Requirements validated? → Move to Validated with phase reference
5. New requirements emerged from compliance? → Add to Active
6. "What This Is" still accurate? → Update if drifted

**After milestone (final submission to both Gradescopes):**
1. Final `full_compliance_check` against shipped artifacts
2. Both members confirm CIS 2450 contribution form filled out
3. Both members confirm Gradescope group-member-add completed (failure = teammate gets 0)

---
*Last updated: 2026-04-29 after initialization*
