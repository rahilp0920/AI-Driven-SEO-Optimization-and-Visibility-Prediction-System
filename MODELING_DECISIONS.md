# Modeling Decisions

## Cut models (deadline-scoped)

Original plan: 8 models (LogReg, RF, GB, XGBoost, LightGBM, CatBoost, SVM-RBF, MLP). Final sweep: **4 models — LogReg, RF, XGBoost, MLP.**

**Cut: GBM, LightGBM, CatBoost, SVM-RBF.** Justification:
- **LightGBM, CatBoost** — empirically converge to within 1-3% F1 of XGBoost on tabular binary classification at this dataset size (~1500 rows). Differences vs XGBoost are well-documented in the literature; running all three would consume tuning budget without changing the winner-class story for the comparison table.
- **GBM (sklearn GradientBoostingClassifier)** — superseded by XGBoost; redundant with the boosting family already represented.
- **SVM-RBF** — RBF kernel scales poorly to TF-IDF feature dimensionality; would require dimensionality reduction (extra pipeline step) for marginal expected gain. Cut to keep evaluation pipeline uniform.

The retained 4 cover the four model families the rubric expects — linear (LogReg), bagging tree ensemble (RF), boosting (XGBoost), neural (MLP) — preserving the baseline-to-advanced progression story.

## Kept models — to be filled during Phase 1 build

For each: why-tried · search space · CV strategy · metrics on holdout (P, R, F1, ROC-AUC, PR-AUC) · kept/dropped · prior-result reasoning that led to the next model.

### LogReg (`src/models/baseline.py`)
TBD.

### Random Forest (`src/models/tree_models.py`)
TBD.

### XGBoost (`src/models/boosting.py`)
TBD.

### MLP (`src/models/neural.py`, Colab-trained, checkpoint loaded via rubric §VI)
TBD.

## Comparison table
TBD — `notebooks/03_model_comparison.ipynb`.

## Winner + defending paragraph
TBD.
