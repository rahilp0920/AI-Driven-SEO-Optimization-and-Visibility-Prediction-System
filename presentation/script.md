# Presentation Script

**Project:** SEO Ranking Predictor & Recommendation System
**Course:** CIS 2450 — Big Data Analytics, Final Project
**Authors:** Rahil Patel · Ayush Tripathi
**Target length:** 9 minutes · **hard window 8–10 min**

---

## Speaker split

| Slides | Speaker | Why |
|--------|---------|-----|
| 1 (Title) | **Rahil** opens · **Ayush** intros himself | Both names heard upfront |
| 2 — Problem | **Ayush** | sets the framing |
| 3 — Dataset | **Rahil** | owns scraping + SERP layer |
| 4 — Feature pipeline | **Rahil** | built the feature extractors |
| 5 — EDA pt 1 | **Ayush** | owns the EDA notebook |
| 6 — EDA pt 2 | **Ayush** | continues |
| 7 — Graph layer | **Rahil** | built the graph pipeline |
| 8 — Modeling sweep | **Rahil** | trained LR / RF / XGB |
| 9 — Results | **Ayush** | wrote the evaluator + comparison |
| 10 — Dashboard demo | **Ayush** drives demo · **Rahil** narrates backend | both visible |
| 11 — Insights | **Rahil** | |
| 12 — Challenges & future | **Ayush** | |
| 13 — Q&A close | both | |

Both members must be on camera throughout. No code on slides. No
sped-up audio.

---

## Slide-by-slide script

> Lines in *italics* are stage directions, not spoken.
> Times in **[brackets]** are running totals, target 9:00.

---

### Slide 1 — Title  · **[0:00 — 0:30]**

**Rahil:**
> Hi — I'm Rahil Patel.

**Ayush:**
> And I'm Ayush Tripathi. Today we're presenting our CIS 2450 final
> project: an SEO Ranking Predictor and Recommendation System. We
> built a binary classifier that predicts whether a developer
> documentation page will appear in Google's top-10 search results
> for its topic — and a Streamlit dashboard that explains every
> prediction with concrete, actionable recommendations.

*[~25 s]*

---

### Slide 2 — The problem · **[0:30 — 1:10]**

**Ayush:**
> Search ranking is famously opaque — Google doesn't publish the
> recipe. But the *inputs* aren't opaque at all. Page-level signals
> like content length, heading structure, keyword placement,
> internal links, and link-graph authority are all observable per
> page. So we framed the question as supervised learning: given
> those observable inputs, can we predict whether the page will rank
> top-10? And just as importantly — can we explain *why* the model
> says yes or no, so an author can act on that explanation? That's
> the project in three pillars on this slide.

*[~40 s]*

---

### Slide 3 — Dataset · **[1:10 — 1:55]**

**Rahil:**
> We pulled from two distinct sources, joined per page. Source one
> is the documentation HTML — an async, robots-aware crawler over
> six developer-doc domains: Python, MDN, React, Node, Kubernetes,
> and FastAPI. Each page lands as raw HTML plus a JSON sidecar.
> Source two is Google SERP rankings, fetched via the Brave Search
> API on the free tier with SerpApi as a fallback. For each scraped
> page we derive a topic query from its `<title>` tag and ask the
> SERP API for the top-10 results. The label `is_top_10` is true
> if the page's own URL appears in those top-10. The two sources
> are joined on a `host + path` key.

> One important note in the callout: per CIS 2450 TA Ricky Gong's
> March 29 email, our scope was officially narrowed from 50,000
> rows to roughly 1,300 pages because of free-tier API rate
> limits. That narrowing is sanctioned and documented.

*[~45 s]*

---

### Slide 4 — Feature pipeline · **[1:55 — 2:35]**

**Rahil:**
> The feature matrix is 72 numeric columns across five families.
> Content gives us text length, word count, Flesch reading ease,
> keyword density. Metadata covers title length and meta-description
> structure. Structural counts headings, internal vs. external
> links, image and alt-text coverage. TF-IDF projects each page
> onto a corpus-fitted top-50 vocabulary using bigrams. And the
> graph family — which we'll come back to in two slides — gives us
> PageRank, HITS hub and authority, in- and out-degree, and
> clustering coefficient per page. All five families are joined on
> the same per-page key.

*[~40 s]*

---

### Slide 5 — EDA part 1 · **[2:35 — 3:25]**

**Ayush:**
> Two takeaways from EDA. First, on the left: after minority-class
> oversampling we have a 50/50 class split. That justifies our
> metric choice — F1 and PR-AUC, *not* accuracy, because accuracy
> on imbalanced data is misleading and the rubric explicitly flags
> that as a conceptual error.

> Second, on the right: when you rank features by Pearson
> correlation with `is_top_10`, the strongest signals are
> *structural*, not content. Heading counts, internal links, and
> graph centrality outrank raw word count. That's the first
> non-obvious insight — content length on its own is weaker than
> we expected.

*[~50 s]*

---

### Slide 6 — EDA part 2 · **[3:25 — 4:10]**

**Ayush:**
> Two distribution views. Top-left: title length. Top-10 pages
> cluster in the 30 to 60 character range — the same range
> classical SEO best-practice recommends, which is a nice
> sanity-check that the model isn't picking up something
> nonsensical. Top-right: H2 heading count. Top-10 pages have
> visibly more H2s, which is consistent with structured-content
> guidance. The bottom strip shows pages by domain — our
> distribution skews heavier on Python and React because they were
> scraped first within the daily API quota.

*[~45 s]*

---

### Slide 7 — Link-graph layer · **[4:10 — 4:55]**

**Rahil:**
> This is the part of the project that hits the Graphs course
> topic. We build a directed graph where nodes are scraped pages
> and edges are page-to-page outbound links between scraped pages.
> Out-of-corpus links are dropped so PageRank converges
> meaningfully on a closed graph.

> Three algorithms feed into the feature matrix. PageRank with
> alpha 0.85 — the original Page-Brin formulation — gives us
> stationary-distribution importance per page. The histogram on
> the left shows the heavy-tailed shape you expect on a real
> link economy. HITS — Kleinberg's algorithm — separates hubs from
> authorities, and on the right scatter you can see top-10 pages
> cluster in the high-authority half. And clustering coefficient
> measures local triangle density, which captures
> topic-coherence within a domain.

*[~45 s]*

---

### Slide 8 — Modeling sweep · **[4:55 — 5:45]**

**Rahil:**
> Four-model progression — baseline to advanced. Logistic
> Regression as the linear baseline. Random Forest for bagging.
> XGBoost for boosting. And a PyTorch MLP for the neural family.
> Each model was tuned with RandomizedSearchCV — 30 iterations
> each, with loguniform priors on continuous knobs like
> regularization strength and learning rate, and integer ranges
> on tree depth and leaf size. Every model uses the same
> StratifiedKFold splitter with seed 42, so when we compare them
> the numbers are apples-to-apples — that's the second card.

> The third card covers imbalance handling. We use
> class_weight='balanced' for LR and RF, scale_pos_weight for
> XGBoost, and we ship a separate oversampling pre-processing
> path that brings the corpus to class parity for the augmented
> training pass.

*[~50 s]*

---

### Slide 9 — Results · **[5:45 — 6:30]**

**Ayush:**
> Held-out performance on a single 80/20 stratified split, seed
> 42. The grouped bar chart shows F1, ROC-AUC, and PR-AUC across
> the four models. XGBoost wins on F1 and PR-AUC — which we'd
> expect for tabular binary classification at this scale. The
> winner banner on the right shows the headline numbers, and below
> it the confusion matrix on the held-out test set: most errors
> are false negatives, which is the more forgiving direction —
> we'd rather the recommendation engine miss a future top-10 page
> than tell an author their already-good page needs work.

*[~45 s]*

---

### Slide 10 — Dashboard demo · **[6:30 — 7:30]**

> *[Ayush switches to the live dashboard. Rahil narrates while
> Ayush drives.]*

**Ayush** *(driving):*
> The Streamlit dashboard has seven tabs. *[clicks Predict]*
> Predict takes any developer-doc URL — let me try
> `docs.python.org/library/asyncio.html`. *[hits Scrape & predict]*
> The dashboard scrapes the page, extracts the same 72 features
> the models trained on, runs every loaded model. Each card shows
> a percentile-rank-within-top-10-pages probability, plus a SHAP
> attribution showing what's pushing the score up versus down.

**Rahil:**
> The EDA tab mirrors the analysis we just walked through; the
> Graph tab shows the URL-hierarchy network with PageRank-sized
> nodes; the Models tab regenerates ROC and PR curves on the fly
> from the saved estimators.

**Ayush:**
> The Recommendations tab turns the SHAP attribution into concrete
> actions — "shorten title to 50–60 chars", "add 2 H2 headers",
> ranked by predicted lift. And What-if lets you drag sliders to
> see how the prediction would change if you implemented a
> recommendation. *[ends demo]*

*[~60 s — keep it tight]*

---

### Slide 11 — Insights · **[7:30 — 8:10]**

**Rahil:**
> Three takeaways. One — structure beats length. Heading and
> link-graph features outrank raw word count in the importance
> ranking. Two — authority is observable. PageRank and HITS are
> stronger predictors than any single content feature, which means
> internal linking — the part of SEO we control on our own
> site — is high-leverage. Three — title craft moves the needle.
> The What-if simulator confirms a 15-character title change
> shifts predicted probability by 6 to 12 points. Net practical
> implication: the recommendation engine prioritises structural
> fixes over content rewrites — lower edit cost, higher predicted
> lift.

*[~40 s]*

---

### Slide 12 — Challenges & future work · **[8:10 — 8:50]**

**Ayush:**
> Four challenges, four next steps.

> Challenges: free-tier API limits bounded the corpus size. Class
> imbalance was real and we mitigated with weighting and
> oversampling. Live URLs aren't in the training graph, so we
> median-fill their graph features rather than zero-fill — zero
> would bias predictions down. And title-derived queries can
> mislead, so the dashboard exposes a manual query override for
> evaluation.

> Future work: per-domain calibrated thresholds instead of a
> blanket 0.5 cutoff; periodic re-scrape with drift monitoring
> since SERPs shift weekly; an A/B harness that applies a
> recommendation, re-scrapes, and verifies actual rank
> movement — that closes the loop on whether we're causal or just
> correlated; and cross-engine generalisation to Bing and
> DuckDuckGo.

*[~40 s]*

---

### Slide 13 — Thank you · **[8:50 — 9:00]**

**Rahil:**
> That's the project — thanks for watching.

**Ayush:**
> Happy to take questions.

*[10 s]*

---

## Total target: **9:00**

If you go over, drop the "domain breakdown" line on slide 6 and
trim slide 12 to two challenges + two futures. If you go under,
linger on the dashboard demo (slide 10) — the rubric explicitly
rewards interactive demonstration.

## Pre-record checklist

- [ ] Both cameras on
- [ ] No code visible on any slide (verified — all code lives in
      `src/`, presentation visualizes only)
- [ ] No sped-up audio (rubric: -2 pts)
- [ ] Length 8:00 ≤ T ≤ 10:00 (rubric: -2 pts otherwise)
- [ ] All five required sections covered: objective, EDA, modeling
      results, implications, challenges + future
- [ ] Dashboard demo runs end-to-end before recording
      (`streamlit run src/dashboard/app.py` should boot in <5 s)
- [ ] Slide deck exported to PDF for the slides.pdf submission
      (`presentation/slides.pdf`)

## File outputs

- `presentation/slides.pptx` — editable deck
- `presentation/slides.pdf` — exported submission copy (see export
  step below)
- `presentation/recording.mp4` — final recorded presentation
- `presentation/script.md` — this file (speaker notes)

## Exporting to PDF

The submission asks for a PDF. From PowerPoint:
**File → Export → Create PDF/XPS Document**, or on the command line
with LibreOffice:

```bash
soffice --headless --convert-to pdf \
        --outdir presentation presentation/slides.pptx
```

## Regenerating

The deck is fully reproducible:

```bash
python -m presentation.build_charts   # static PNGs at 300 DPI
python -m presentation.build_slides   # → presentation/slides.pptx
```

If model metrics or feature counts change, re-run both — the deck
reads them live from `models/metrics/*.json` and `features.csv`.
