# Modeling Decisions

Every modeling choice is captured here with the prior result that motivated it. CIS 2450 weights
codebase polish at 83/143 points and explicitly penalizes silent modeling choices.

## Comparison table

⚠️ **Two number sets coexist below.** The "demo (synthetic labels)" row was produced on a
100-page smoke-test corpus with synthetic SERP labels (`scripts/synthesize_serp.py`,
~30% positive rate, random correlation to source URL). It exists only to verify the pipeline
runs end-to-end and to seed the dashboard with non-trivial models. **For the graded
submission, replace `data/interim/serp.csv` with real Brave/SerpApi fetches and re-run the
sweep — the numbers below will change substantially.**

### Demo run (2026-04-29 — synthetic labels, n_test=20)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC | Precision | Recall | TP | FP | FN | TN |
|-------|----------|----|---------|--------|-----------|--------|----|----|----|----|
| Logistic Regression | 0.60 | 0.429 | 0.653 | 0.488 | 0.333 | 0.600 | 3 | 6 | 2 | 9 |
| Random Forest       | 0.60 | 0.200 | 0.560 | 0.294 | 0.200 | 0.200 | 1 | 4 | 4 | 11 |
| **XGBoost**         | **0.75** | **0.615** | **0.693** | **0.376** | **0.500** | **0.800** | **4** | **4** | **1** | **11** |
| MLP (PyTorch)       | not-trained-this-run | — | — | — | — | — | — | — | — | — |

**Demo winner:** **XGBoost** (F1=0.615, ROC-AUC=0.693, accuracy=0.75 on a 20-page held-out
test set with 5 positive labels). Note: accuracy alone is misleading on imbalanced data — the
all-negative classifier scores 0.75 here (15/20). F1 + PR-AUC are the metrics that matter.

### Real run (TBD — once `serp_client fetch` runs against Brave/SerpApi on the full ~1500-page corpus)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC | Precision | Recall | best_params (abridged) |
|-------|----------|----|---------|--------|-----------|--------|------------------------|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD | C, penalty, class_weight |
| Random Forest       | TBD | TBD | TBD | TBD | TBD | TBD | n_estimators, max_depth, min_samples_leaf, class_weight |
| XGBoost             | TBD | TBD | TBD | TBD | TBD | TBD | learning_rate, max_depth, n_estimators, subsample |
| MLP (PyTorch)       | TBD | TBD | TBD | TBD | TBD | TBD | hidden_dims, dropout, epochs (Colab) |

**Selection rule:** F1 first; if F1 is within 1% across two models, prefer the one with higher
PR-AUC (puts more positives near the top of the ranked list — the SEO-recommendation failure
mode that matters most).

**Real-run winner:** _to be filled after the real sweep finishes._

## Cut models (deadline-scoped)

Original proposal: 8 models (LR, RF, GBM, XGBoost, LightGBM, CatBoost, SVM-RBF, MLP).
Final sweep: **4 models — LR, RF, XGBoost, MLP.**

**Cut: GBM, LightGBM, CatBoost, SVM-RBF.** Justification:

- **LightGBM and CatBoost** — empirically converge to within 1-3% F1 of XGBoost on tabular
  binary classification at this dataset size (~1500 rows). Differences vs XGBoost are
  well-documented in the literature; running all three would consume tuning budget without
  changing the winner-class story for the comparison table. Same boosting family, near-identical
  inductive bias.
- **GBM (sklearn `GradientBoostingClassifier`)** — superseded by XGBoost in the same family;
  redundant once XGBoost is in.
- **SVM-RBF** — the RBF kernel scales poorly to TF-IDF feature dimensionality (50 columns × 1500
  rows is workable but SVM tuning is slow, and the kernel matrix grows quadratically in n).
  Would have required a dimensionality-reduction step (extra moving piece) for marginal expected
  gain. Cut to keep the evaluation pipeline uniform.

The retained 4 still cover the four model families the rubric expects (linear / bagging /
boosting / neural), preserving the baseline-to-advanced progression story.

## Kept models

Each section is structured: **Why tried · Search space · CV strategy · Results · Kept/Dropped ·
Lesson for next model.** The "Results" rows are filled after the sweep runs.

### 1. Logistic Regression — `src/models/baseline.py`

- **Why tried:** linear baseline is the cheapest model that lets us measure whether the
  feature engineering (content + metadata + structural + graph + TF-IDF) carries any signal at
  all. If LR can't beat the majority-class baseline, the features are noise and we restart
  feature design before adding model complexity.
- **Search space:** `C ~ loguniform(1e-3, 1e2)`, `penalty ∈ {l1, l2}`, `class_weight ∈ {None, balanced}`.
- **CV strategy:** StratifiedKFold(5, shuffle=True, random_state=42), `n_iter=30`,
  `scoring="f1"`. Pipeline includes `StandardScaler(with_mean=False)` so sparse TF-IDF columns
  pass through cleanly.
- **Results:** TBD.
- **Kept/Dropped:** Kept as the baseline (rubric requirement: must show baseline → advanced
  progression).
- **Lesson for next model:** TBD — fill in once metrics are in. The expected pattern: LR sets
  the floor; we look at LR's confusion-matrix to identify which class is harder to find, and
  pick the next model to attack that failure mode.

### 2. Random Forest — `src/models/tree_models.py`

- **Why tried:** moves to non-linear decision boundaries and feature interactions without
  requiring feature scaling. Bagging family. Robust to the heterogeneous feature scales we have
  (raw counts + percentages + log-tail PageRank).
- **Search space:** `n_estimators ~ randint(100, 600)`, `max_depth ∈ {None, 4, 8, 16, 32}`,
  `min_samples_split ~ randint(2, 12)`, `min_samples_leaf ~ randint(1, 8)`,
  `max_features ∈ {sqrt, log2, 0.5}`, `class_weight ∈ {None, balanced, balanced_subsample}`.
- **CV strategy:** Same StratifiedKFold(5), `n_iter=30`, `scoring="f1"`.
- **Results:** TBD.
- **Kept/Dropped:** Kept (bagging family representative).
- **Lesson for next model:** TBD — typically RF tells us where boosting will help (high-variance
  trees → lower-variance boosted ensemble lifts F1 a few points).

### 3. XGBoost — `src/models/boosting.py`

- **Why tried:** boosting family. Dominant on tabular binary classification benchmarks at this
  scale; its sequential bias correction tends to lift recall on the minority class without
  destroying precision (which RF often does when `class_weight=balanced`).
- **Search space:** `n_estimators ~ randint(100, 600)`, `max_depth ~ randint(3, 12)`,
  `learning_rate ~ loguniform(1e-3, 0.3)`, `subsample ~ uniform(0.6, 1.0)`,
  `colsample_bytree ~ uniform(0.6, 1.0)`, `reg_lambda ~ loguniform(1e-3, 10)`,
  `reg_alpha ~ loguniform(1e-4, 1)`, `min_child_weight ~ randint(1, 10)`.
- **CV strategy:** StratifiedKFold(5), `n_iter=40`, `scoring="f1"`. Class imbalance handled via
  `scale_pos_weight = neg / pos` computed from the training labels (XGBoost's preferred lever
  vs sklearn's `class_weight`). `tree_method="hist"` for speed.
- **Results:** TBD.
- **Kept/Dropped:** Expected sweep winner. Used for SHAP explanations in the dashboard
  (TreeExplainer is fast and exact for tree ensembles).
- **Lesson for next model:** XGBoost defines the tree-ensemble ceiling. The MLP is the only
  remaining model class that can plausibly beat it — different inductive bias (continuous
  decision surface vs axis-aligned splits).

### 4. PyTorch MLP — `src/models/neural.py`

- **Why tried:** different inductive bias from the tree family. Even if it doesn't win, the
  rubric values *capability* (Colab-trained checkpoint loaded locally per §VI pattern), and a
  neural baseline tells us whether continuous-decision-surface modeling adds anything.
- **Architecture:** Linear(`input_dim` → 128) → ReLU → Dropout(0.2) → Linear(128 → 64) → ReLU
  → Dropout(0.2) → Linear(64 → 1) logit. `BCEWithLogitsLoss(pos_weight = neg/pos)`.
- **Training:** Adam, `lr=1e-3`, `weight_decay=1e-4`, `epochs=60-80`, `batch_size=64`. Inner
  15% validation split for early-stopping by val loss. Features standardised once on the train
  split; the same scaler is saved into the checkpoint.
- **Search:** Hyperparameters fixed (no RandomizedSearchCV — sklearn's wrapper doesn't fit
  PyTorch cleanly). One small manual sweep on `hidden_dims ∈ {(128,64), (256,128), (64,32)}`
  + `dropout ∈ {0.1, 0.2, 0.3}` recorded as a comment in the notebook.
- **Checkpoint payload (rubric §VI pattern):** `{model_state_dict, config (input_dim,
  hidden_dims, dropout), scaler_mean, scaler_scale, feature_names, epoch, best_threshold}`.
  `load_checkpoint()` reconstructs an `MLPInferenceWrapper` exposing sklearn-shaped
  `predict` / `predict_proba` so SHAP, the comparison notebook, and the dashboard treat it
  identically to LR / RF / XGB.
- **Results:** TBD.
- **Kept/Dropped:** Kept regardless (proves capability + checkpoint workflow). If F1 beats
  XGBoost, used for the headline prediction. Either way, used in the comparison table.

## Other modeling decisions

### Feature scaling

- **LR:** `StandardScaler(with_mean=False)` — preserves sparsity for the TF-IDF block.
- **RF, XGBoost:** no scaling — tree splits are scale-invariant.
- **MLP:** `StandardScaler` fit on the train split (mean + scale persisted into the checkpoint
  so dashboard inference scales identically to training).

### Class imbalance

- LR / RF: `class_weight ∈ {None, balanced}` swept by RandomizedSearchCV.
- XGBoost: `scale_pos_weight = neg / pos` from train labels.
- MLP: `BCEWithLogitsLoss(pos_weight = neg / pos)`.
- **No SMOTE / oversampling** — stays inside the original feature distribution; we let the loss
  function carry the imbalance correction.

### Train / test split

- 80/20 stratified holdout (`random_state=42`).
- **Same split for every model** — ensures comparison-table numbers are apples-to-apples.

### Threshold

- **0.5 by default** for predict() across all models. The dashboard's "high / medium / low"
  bucketing uses `[0.5, 0.3, 0.0]` cutoffs. Future work: per-domain calibrated thresholds.

### Reproducibility

- Single `random_state=42` everywhere (`StratifiedKFold`, `train_test_split`,
  `RandomizedSearchCV`, MLP seeds for `torch` and `numpy`).
- Pinned versions in `requirements.txt`.
