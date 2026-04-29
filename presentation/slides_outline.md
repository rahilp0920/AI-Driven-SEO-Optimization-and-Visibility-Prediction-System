# Presentation slide outline (8-10 min HARD limit)

Drop this into Google Slides → export PDF as `presentation/slides.pdf`.
Total target: **9:00** with 30-60s buffer on either side. **No code on slides.**
**No sped-up audio** (-2 pts if detected).

Coverage order is mandated by the rubric:
objective + value → data + pivot → EDA charts (3-5) → modeling progression →
implications → limitations → live demo.

---

## Slide 1 — Title (0:00-0:15)
- **AI-Driven SEO Ranking Predictor & Recommendation System**
- Rahil Patel · Ayush Tripathi · CIS 2450 + NETS 1500 final · Spring 2026
- One-line tagline: *Will this developer documentation page rank in Google's top-10? And what should you change to get there?*

**Speaker:** Rahil opens. Hand off to Ayush at slide 5.

---

## Slide 2 — Objective + value (0:15-0:45)
**Headline:** Binary classifier + actionable recommender.
- **Predict:** P(top-10 SERP) for a developer-documentation page on the topic query derived from its `<title>`.
- **Recommend:** ≥3 concrete, numerically-grounded suggestions per page (title length, H2 count, alt text coverage, etc.).
- **Why this matters:** docs that don't rank don't get used. We give writers a leading indicator + a concrete fix list before publishing.

**Speaker note:** keep this slide tight — graders want specificity here. Say "top-10" out loud, not "ranks well."

---

## Slide 3 — Data + sanctioned pivot (0:45-1:30)
- **Two distinct sources** (CIS hard requirement):
  - Scraped HTML across 5+ developer-doc domains (docs.python.org, MDN, react.dev, nodejs.org, kubernetes.io, fastapi.tiangolo.com).
  - Top-10 Google SERP rankings via the Brave Search API.
- **Robots.txt-respected, polite User-Agent, ~1 req/s rate limit per domain.**
- **Pivot:** scope narrowed from 50K to ~1500 pages — per CIS 2450 TA Ricky Gong's email of 2026-03-29 (free-tier API rate limits made 50K infeasible inside the deadline). Sanctioned, not a deviation.

**Speaker note:** read the Ricky-quote sentence verbatim — graders will be checking for it. Show one screenshot of `data/raw/<domain>/` + one of `serp.csv` head.

---

## Slide 4 — EDA highlight #1: Class balance (1:30-2:10)
- Show `assets/charts/01_class_balance.png` full-bleed.
- **Read it out:** "Roughly N% of pages are top-10 — the rest are not. Imbalanced."
- **Modeling decision:** F1 + PR-AUC primary metrics, not accuracy. PR-AUC tiebreaker because it weighs the minority class.

---

## Slide 5 — EDA highlight #2: Feature distributions (2:10-2:50)
- Show `assets/charts/02_feature_distributions.png`.
- **Read it out:** point at one visibly-separating feature (e.g., title_length or pagerank) and one with overlap (e.g., word_count).
- **Modeling decision:** strong per-class separation justifies tree-based ensembles; overlapping features motivate L1 regularization in LR to prune them.

**Speaker:** Ayush takes over at this slide and runs through to slide 9.

---

## Slide 6 — EDA highlight #3: PageRank distribution (2:50-3:30)
- Show `assets/charts/04_pagerank_distribution.png` (log scale).
- **Read it out:** heavy-tailed — a few hub pages dominate.
- **Why this matters:** confirms graph features carry signal; uniform PageRank would have meant the column is just noise.
- **Graph algorithms used (Shivani's feedback addressed):** PageRank (α=0.85), HITS hub/authority, in-/out-degree, clustering coefficient.

---

## Slide 7 — Modeling progression + comparison table (3:30-6:30)
The 3-minute meat slide. Present in three beats.

**Beat 1 (45s) — model ladder:**
- LR (linear baseline) → Random Forest (bagging) → XGBoost (boosting) → PyTorch MLP (neural).
- Reduced from 8 to 4 to fit deadline. Cuts (LightGBM, CatBoost, GBM, SVM-RBF) documented in `MODELING_DECISIONS.md`. Justification: LightGBM/CatBoost converge to within 1-3% F1 of XGBoost on tabular binary at this size — well-documented in the literature; SVM-RBF scales poorly to TF-IDF.
- All 4 model families the rubric expects are still represented.

**Beat 2 (1m) — methodology:**
- Stratified 80/20 holdout. 5-fold StratifiedKFold for hyperparameter search. RandomizedSearchCV with documented `n_iter` per model.
- Imbalance handled per-model: `class_weight=balanced` (LR, RF), `scale_pos_weight=neg/pos` (XGB), `BCEWithLogitsLoss(pos_weight=...)` (MLP).
- MLP trained in Colab; checkpoint exported via the rubric §VI pattern (`state_dict + config + scaler + feature_names`); loaded locally for the dashboard.

**Beat 3 (1m 15s) — results:**
- Show comparison table: model · F1 · ROC-AUC · PR-AUC · precision · recall.
- Name the winner. Defend the choice: F1 first, PR-AUC as tiebreaker (puts true positives near the top of the ranked list — the SEO-recommendation failure mode that matters most).
- Show `assets/charts/06_shap_summary.png` to motivate the recommendations: which features actually drive the prediction.

---

## Slide 8 — Recommendations engine (6:30-7:00)
- Rule-based core (title length, meta description, H2 count, alt text, keyword density, internal links).
- SHAP-ranked: when a tree model is loaded, suggestions are sorted by `|SHAP value|` so the highest-leverage fix shows first.
- **Concrete numbers, not vague advice:** "Title is 80 chars — shorten to 30-60." "Add alt text to 4 of 7 images." "Add 2 H2 headers."

---

## Slide 9 — Live dashboard demo (7:00-8:00)
- **Switch to screen share. ~60 seconds.**
- 1. Paste a real URL → click Scrape & predict → show 4 metric cards.
- 2. Switch to Recommendations tab → show top suggestion's "why" line.
- 3. Switch to What-if → drag the title-length slider → watch the prediction tick up; call out the delta.
- 4. Switch to About → point at the AI-usage disclosure.

**Speaker:** Rahil drives the demo (he wired the backend); Ayush narrates.

---

## Slide 10 — Implications + limitations (8:00-8:45)
**Implications (20s):**
- Concrete pre-publication leading indicator for technical-doc writers.
- Methodology generalizes to any domain where SERP rankings are observable (e-commerce, support docs, marketing pages).

**Limitations (25s):**
- Domain-restricted training set (developer docs); generalization to non-technical content is untested.
- SERP labels are point-in-time snapshots — Google's ranking is non-stationary, so periodic re-scraping is needed for production use.
- We do not observe backlinks (paid API gate); on-page signals only.

---

## Slide 11 — AI usage disclosure (8:45-9:00)
**Required by both rubrics. Be specific.**
- Used Claude (Anthropic) to scaffold async crawler boilerplate, sklearn pipeline + RandomizedSearchCV harness, Streamlit component layout. Each generated function reviewed + type-hint-audited + integration-tested before commit.
- Used Claude to draft initial documentation (this slide deck included as a structured outline, then we wrote the talking points ourselves).
- We did NOT use AI for the modeling sweep itself, the data ethics decisions (domain pick / robots.txt / rate limits), or the final feature-engineering choices.

**Speaker:** both speakers acknowledge briefly.

---

## Slide 12 — Thanks + Q&A backup
- Thanks; team contacts.
- Repo: `github.com/rahilp0920/AI-Driven-SEO-Optimization-and-Visibility-Prediction-System`
- Backup detail slides hidden behind this one if Q&A goes deep:
  - Confusion matrices per model
  - Hyperparameter search-space ranges per model
  - Per-domain page counts table
  - Robots.txt compliance log

---

## Recording checklist (do BEFORE you click Record)

- [ ] Both speakers have working mic — test with a 10-second sample.
- [ ] Timer running (Phone or `presentation/` `.mp4` clock overlay) — abort and restart if you cross 9:50.
- [ ] Screen share scaled so the dashboard text is readable (zoom in once before live demo).
- [ ] Demo URL prepared in clipboard; backup URL too in case the first 403s.
- [ ] Demo data fallback works — confirm by disabling network for 2 seconds before recording.
- [ ] No filename, terminal, or code editor visible in any slide screenshot.

## Post-recording

- [ ] Listen back at full speed. If anyone sounds sped-up, RE-RECORD (-2 pts otherwise).
- [ ] Verify total length is 8:00-10:00 (not 7:59, not 10:01).
- [ ] Export slides as PDF → `presentation/slides.pdf`.
- [ ] Save recording as `presentation/recording.mp4`.
- [ ] Both members fill out CIS 2450 Project Contribution Form.
- [ ] Submit to CIS 2450 Gradescope by 2026-04-30 23:59 with both members on the group.
