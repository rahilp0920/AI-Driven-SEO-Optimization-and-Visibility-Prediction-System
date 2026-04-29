# Project: AI-Driven SEO Ranking Predictor & Recommendation System

## What this project is

A binary classifier that predicts whether a developer documentation page appears in the top-10 Google SERP results for its primary topic query (query derived from `<title>`), plus a Streamlit dashboard with what-if simulator and SHAP-based recommendations. Joint final project for **CIS 2450 (Big Data Analytics, due 2026-04-30 23:59)** and **NETS 1500 (HW5 Project, due 2026-04-29 23:59)**.

## Critical project context

**This project is graded against external rubrics** stored in the `project-compliance` MCP server. Any disagreement between Claude's judgment and a rubric is resolved by the rubric. Always run `mcp__project-compliance__full_compliance_check` after any non-trivial change and treat its output as ground truth.

**Two course deadlines, prioritized:**
- NETS 1500 (85 pts) — minimum viable, ship by 2026-04-29 23:59
- CIS 2450 (143 pts) — 90% of effort goes here, due 2026-04-30 23:59

**Pivot rationale (must appear verbatim in README.md and data/README.md):** Per CIS 2450 TA Ricky Gong's email of 2026-03-29, the project was narrowed from the original 50K-row scope to ~1500 developer documentation pages because full-scale free-tier scraping is rate-limited. This is sanctioned, not a deviation.

## Workflow — GSD (Get Shit Done)

This project uses GSD planning artifacts in `.planning/`:
- `PROJECT.md` — vision, core value, requirements, constraints, key decisions
- `REQUIREMENTS.md` — atomic REQ-IDs traceable to rubric lines
- `ROADMAP.md` — 10 phases with success criteria + per-phase compliance gates
- `STATE.md` — current position, phase status, open issues
- `config.json` — workflow mode (YOLO), granularity (fine), compliance gate enabled

**Per-phase loop:** plan → execute → `full_compliance_check` → fix any P0 flag → commit → advance.

**Use the slash commands** (`/gsd-plan-phase N`, `/gsd-execute-phase N`, `/gsd-progress`) — don't replan from scratch.

## Hard rules

1. **Run `mcp__project-compliance__full_compliance_check` after every phase.** Save output to `.planning/compliance/phase-N-check.md`. Any P0 flag is a blocker for the current phase.
2. **Public, legal data only.** Respect `robots.txt` and rate limits. CIS 2450 rubric §2 — failure = automatic 0.
3. **Both members commit to git** with their own attributed commits. CIS 2450 hard requirement (equal contributions).
4. **AI usage must be documented** in (a) `readme.txt`, (b) the dashboard's About tab, AND (c) a presentation slide. CIS 2450 §2.b is explicit on this.
5. **Presentation is 8-10 minutes hard.** No code on slides. No sped-up audio (-2 pts).
6. **Module + function docstrings + type hints throughout.** CIS 2450 weights codebase polish at 83/143 points.
7. **No silent modeling choices.** Every model decision goes in `MODELING_DECISIONS.md` with prior-result reasoning.

## Codebase structure (non-negotiable)

```
/
├── README.md                  # CIS 2450 main; rubric-to-file map
├── readme.txt                 # NETS 1500 1-pager
├── USER_MANUAL.md             # NETS 1500 user manual + screenshots
├── MODELING_DECISIONS.md      # Per-model decision log
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
├── assets/charts/             # ≥300 DPI EDA/SHAP PNGs
└── presentation/{slides.pdf, recording.mp4}
```

## Tech stack

Python 3.11+, scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, NetworkX, Streamlit, SHAP, BeautifulSoup/lxml, aiohttp/httpx, SerpApi (free tier) or Brave Search API.

## Work split

- **Rahil Patel** (`rahilp07@seas.upenn.edu`) — `src/scraping/`, `src/features/`, `src/graph/`, `src/models/{baseline,tree_models,boosting}.py`, dashboard backend
- **Ayush Tripathi** (`tripath1@seas.upenn.edu`) — `notebooks/01_eda.ipynb`, `src/models/{kernel,neural}.py`, `notebooks/03_model_comparison.ipynb`, `src/models/evaluate.py`, `src/recommendations/`, dashboard frontend + styling, slide deck

When committing Ayush's portions: `git config user.name "Ayush Tripathi"` and `git config user.email "tripath1@seas.upenn.edu"` before the commit.

## What to do next

Run `/gsd-progress` to see current state, or `/gsd-plan-phase 1` to start Phase 1 (Data Layer).
