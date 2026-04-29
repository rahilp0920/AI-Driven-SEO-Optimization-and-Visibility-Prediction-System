# Handoff to Ayush — what's done, what's left, exactly how to finish

**Status as of 2026-04-29 ~17:30:** Phase 1 BUILD code is complete and pushed to
`origin/main`. A 100-page smoke-test pipeline ran with synthetic SERP labels —
trained models (LR/RF/XGBoost) and the Streamlit dashboard are committed so you
can verify locally before kicking off the real run.

**You are picking up the keyboard for:** real SERP fetch → real model training
→ docs fill-in → presentation → submit.

---

## Step 0 — sanity check the demo state (5 min)

This proves the pipeline I shipped works end-to-end on your machine before you
touch anything graded.

```bash
git pull
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -v                                  # 23 tests should pass
streamlit run src/dashboard/app.py         # opens at http://localhost:8501
```

In the dashboard:
- **Predict tab** → click **Use demo page** → 3 metric cards should show probabilities
  (XGBoost ≈ 38%, LR ≈ 31%, RF ≈ 16% on the demo asyncio page).
- **Recommendations tab** → ≥3 SHAP-ranked suggestions.
- **What-if tab** → 8 sliders, prediction deltas update live.
- **About tab** → pivot quote + AI usage disclosure.

If anything renders broken, ping Rahil before you wipe.

---

## Step 1 — get a real SERP API key (10 min, FREE)

Brave Search has a free tier that gives ~2,000 queries/month — enough for the
~1,500-page corpus. SerpApi free tier is 100/month, too small for this.

1. Go to **https://brave.com/search/api/** → sign up (just an email).
2. Create a subscription on the **Free AI** plan (no credit card required).
3. Copy the API token. Then in the project root:

```bash
echo "BRAVE_SEARCH_KEY=YOUR_TOKEN_HERE" > .env
echo "SCRAPER_USER_AGENT=AIDrivenSEOResearchBot/1.0 (+contact: tripath1@seas.upenn.edu)" >> .env
echo "SCRAPER_DELAY_SECONDS=1.0" >> .env
```

(Yes, put your contact email in the User-Agent — robots.txt etiquette.)

---

## Step 2 — run the real pipeline (~50 min total, mostly walltime)

```bash
# Wipe the demo synthetic state
rm -rf data/raw data/interim data/processed
rm -f models/*.joblib models/*.pt models/metrics/*.json

# Full pipeline (Makefile orchestrates everything)
make all
# Equivalent to:
#   make scrape    # 5 domains × ~1300 pages, ~22 min at 1 sec/page (polite)
#   make queries   # derive ~1300 queries from <title> tags
#   make serp      # Brave Search top-10 per query, ~22 min, resume-safe
#   make features  # build data/processed/features.csv
#   make graph     # build PageRank/HITS link graph + merge into features.csv
#   make train     # train LR + RF + XGBoost + MLP
```

**If `serp` gets interrupted** (rate limit, network), just re-run `make serp` —
the client skips already-fetched query_ids in the existing serp.csv, so you
don't burn quota.

**For the MLP** (`src/models/neural.py`), you have two paths:
1. **Local CPU** (`make train` runs it last) — fine for this dataset size, takes ~3 min.
2. **Colab (preferred for the rubric §VI checkpoint pattern)** — open
   `notebooks/03_model_comparison.ipynb` in Colab, mount Drive, copy
   `data/processed/features.csv` over, run the `train()` cell. Download
   `mlp_checkpoint.pt` and drop it in `models/`. The dashboard's
   `load_checkpoint()` reconstructs it identically with the saved scaler +
   feature_names. **The grader specifically wants to see the §VI pattern,
   so the Colab path scores better even if local results are identical.**

After training, the comparison notebook + SHAP summary:

```bash
jupyter nbconvert --execute notebooks/03_model_comparison.ipynb --to notebook --inplace
# This populates the comparison table, picks the winner by (F1, PR-AUC), and
# saves assets/charts/06_shap_summary.png.
```

---

## Step 3 — fill in MODELING_DECISIONS.md (15 min)

Open `MODELING_DECISIONS.md`. The "Demo run" table is filled in already from
the synthetic-label smoke test. **Replace the "Real run (TBD)" table with
real numbers** by pasting from `models/metrics/*.json`:

```bash
cat models/metrics/*.json
```

For each kept-model section (LR / RF / XGBoost / MLP), fill in:
- **Results** row — F1, ROC-AUC, PR-AUC, precision, recall, n_test
- **Lesson for next model** — one or two sentences. Pattern:
  - LR's lesson → motivates RF: "linear floor was X; the confusion matrix
    showed false negatives concentrated in long-content pages, so we tried
    a non-linear ensemble next."
  - RF's lesson → motivates XGBoost: "RF gave us interactions but variance
    on the minority class was high; sequential boosting was the obvious
    next step."
  - XGBoost's lesson → motivates MLP: "tree ensemble plateau identified;
    different inductive bias next."

Then update the `**Real-run winner:**` line at the top of the comparison
table. Selection rule: F1 first; if F1 within 1% across two models, prefer
higher PR-AUC.

---

## Step 4 — capture screenshots for USER_MANUAL.md (15 min)

`USER_MANUAL.md` references 7 screenshot paths under `assets/screenshots/`.
With the dashboard running and the real models loaded:

| Path | What to capture |
|------|-----------------|
| `assets/screenshots/01_data_raw.png` | terminal showing `ls -lh data/raw/` (one subdir per domain) |
| `assets/screenshots/02_features_head.png` | terminal showing `head -3 data/processed/features.csv` |
| `assets/screenshots/03_model_comparison.png` | the comparison-table + bar-chart cells from notebook 03 |
| `assets/screenshots/04_dashboard_predict.png` | dashboard Predict tab after a real URL scrape |
| `assets/screenshots/05_dashboard_recommendations.png` | dashboard Recommendations tab |
| `assets/screenshots/06_dashboard_whatif.png` | dashboard What-if tab with a slider moved |
| `assets/screenshots/07_dashboard_about.png` | dashboard About tab |

Use any screenshotter (Cmd-Shift-4 on macOS). Make sure no terminal/file paths
are visible in dashboard screenshots (rubric: "no code on slides" applies in
spirit to UI screenshots too).

---

## Step 5 — slides + recording (1-2 hours)

`presentation/slides_outline.md` is a 12-slide outline mapped to the rubric's
mandated coverage order. It has per-slide talking points and timings totalling
~9 minutes (target: 8-10 min HARD limit).

1. Drop the outline content into Google Slides. **No code on slides** (rubric
   penalty). Replace TBD numbers in slide 7 with the real comparison table.
2. Each chart (slides 4-6) should show the corresponding `assets/charts/*.png`.
3. Slide 11 is the AI-usage disclosure — copy from `readme.txt` and dashboard
   About tab. **Both rubrics require this.**
4. Export → `presentation/slides.pdf`.

Recording:
1. **Both members audibly speak.** Speaker handoffs are marked in the outline.
2. **8:00-10:00 hard window.** Time it twice before recording. **No sped-up
   audio (-2 pts if detected).**
3. Live-demo slide 9 — keep the dashboard visible for ~60 seconds, click
   through Predict → Recommendations → What-if (move a slider).
4. Save → `presentation/recording.mp4`.

---

## Step 6 — submit + paperwork

### NETS 1500 (TONIGHT 2026-04-29 23:59)

Already required: `readme.txt` (committed) + `USER_MANUAL.md` (committed,
needs your screenshots).

```bash
# Add screenshots
git add assets/screenshots/*.png
git commit -m "docs: USER_MANUAL screenshots for NETS submission"
git push
```

Then:
1. Confirm both members are on the Gradescope group for NETS HW5.
2. Submit the repo (or zip + upload, whichever Gradescope expects for NETS).

### CIS 2450 (TOMORROW 2026-04-30 23:59)

1. Both members fill the **Project Contribution Form** (separate from Gradescope).
2. Confirm both members on the Gradescope group for CIS 2450 final project.
3. Submit: code (push to GitHub) + `presentation/slides.pdf` + `presentation/recording.mp4`.
4. **Run a final compliance check:**
   ```bash
   # Inside Claude Code, with mcp__project-compliance configured:
   # > use mcp__project-compliance__full_compliance_check with a fresh project_summary
   ```
   Save the output to `.planning/compliance/phase-final-check.md`. Address any
   P0 flags before submitting.

---

## What's already done (don't redo this)

| Deliverable | Where | Status |
|-------------|-------|--------|
| `requirements.txt`, `.env.example`, `Makefile`, `pyproject.toml`, `conftest.py` | repo root | ✅ |
| Data ethics doc | `data/README.md` | ✅ (pivot quote verbatim) |
| Async robots.txt-aware crawler | `src/scraping/doc_scraper.py` | ✅ |
| SERP client (Brave + SerpApi) + query derivation + resume-safe fetch | `src/scraping/serp_client.py` | ✅ |
| Content/metadata/structural/TF-IDF features | `src/features/` | ✅ |
| Build features orchestrator (joins SERP labels) | `src/features/build_features.py` | ✅ |
| Link graph (PageRank/HITS/clustering merge) | `src/graph/` | ✅ |
| 4 model files + shared evaluator (F1/ROC-AUC/PR-AUC/CM, StratifiedKFold, RandomizedSearchCV) | `src/models/` | ✅ |
| MLP rubric §VI checkpoint pattern (state_dict + config + scaler + feature_names) | `src/models/neural.py` | ✅ |
| Recommendations engine (rules + SHAP-ranked) | `src/recommendations/recommend.py` | ✅ |
| Streamlit dashboard (Predict / Recs / What-if / About) with custom CSS, light theme forced, demo data fallback, MLP checkpoint loader | `src/dashboard/` | ✅ |
| 5 EDA charts + graph notebook + model comparison notebook | `notebooks/` | ✅ |
| Pytest smoke tests (synthetic inputs, no scraped data needed) | `tests/` | ✅ (23 pass) |
| `readme.txt` (NETS 1-pager) | repo root | ✅ |
| `README.md` (CIS rubric→file map) | repo root | ✅ |
| `USER_MANUAL.md` (install → pipeline → dashboard, with screenshot paths) | repo root | 🟡 needs your screenshots |
| `MODELING_DECISIONS.md` (cut models, comparison-table template, per-model framework) | repo root | 🟡 needs real-run numbers filled in |
| `presentation/slides_outline.md` (12-slide outline, talking points, timing) | `presentation/` | 🟡 needs real numbers + Google Slides export |
| Phase 1 compliance gate report | `.planning/compliance/phase-1-check.md` | ✅ |
| GitHub remote pushed | github.com/rahilp0920/AI-Driven-SEO-Optimization-and-Visibility-Prediction-System | ✅ |

---

## What you still need to do (the punch list)

### Tonight (NETS deadline 2026-04-29 23:59)
- [ ] **Get Brave API key** (Step 1, 10 min)
- [ ] **Run real pipeline** (Step 2, ~50 min walltime)
- [ ] **Capture screenshots** (Step 4, 15 min)
- [ ] **Verify both members on Gradescope group**
- [ ] **Submit to NETS Gradescope** by 23:59

### Tomorrow (CIS deadline 2026-04-30 23:59)
- [ ] **Fill in MODELING_DECISIONS.md real numbers** (Step 3, 15 min)
- [ ] **Train MLP in Colab** (rubric §VI bonus path)
- [ ] **Slides + recording** (Step 5, 1-2 hours)
- [ ] **CIS Project Contribution Form** (both members)
- [ ] **Final compliance check** (Step 6, 10 min)
- [ ] **Submit to CIS Gradescope** by 23:59

---

## Where to look if something breaks

- **Scraper hangs / 403s on a domain** → check `robots.txt` for that domain;
  swap to a different one. Update the `DOMAINS` list in the Makefile and in
  `data/README.md`.
- **`no SERP API key found`** → `.env` not loaded. Confirm:
  `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('BRAVE_SEARCH_KEY'))"`
- **Dashboard shows "No trained models loaded"** → `make train` didn't finish.
  Run `make train-fast` (XGBoost only) for the cheapest path to a working dashboard.
- **Streamlit text is invisible / white** → forced light theme is in
  `.streamlit/config.toml`; make sure that file exists. If browser caches the
  old CSS, hard-refresh (cmd-shift-R).
- **`from src.X import Y` fails when running streamlit** → the sys.path injection
  is in `src/dashboard/app.py:1-10`; do not delete it.
- **Tests fail in Colab** → expected; tests are local-only. Run them on your
  machine, not in Colab.

---

## One last thing — the AI-usage disclosure

Both rubrics require detailed AI-usage. The text is already in three places:
- `readme.txt` (NETS submission)
- Dashboard's About tab (`src/dashboard/app.py`)
- Slide 11 of the presentation outline

If you change how you used AI (e.g., used it to draft slides), update all
three to keep them consistent. The rubric explicitly checks for specific
examples ("scaffolding async loop in `src/scraping/doc_scraper.py`" is
better than "we used Claude").

---

Good luck Ayush — the codebase is ready. Most of what's left is running
commands, filling in numbers, and recording. — Rahil
