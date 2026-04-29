"""XGBoost classifier — boosting family (model #3). The expected sweep
winner per the literature on tabular binary classification at this dataset
scale.

Cut models (LightGBM, CatBoost, sklearn GBM) are documented in
``MODELING_DECISIONS.md`` — their differences vs XGBoost are bounded by
1-3% F1 on tabular data of this size, not enough to justify the extra
tuning budget.

CLI:
    python -m src.models.boosting
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from src.models.evaluate import (
    cv_splitter,
    evaluate_classifier,
    load_features,
    save_metrics,
    stratified_split,
)

LOG = logging.getLogger("xgb")

PARAM_DIST = {
    "n_estimators": randint(100, 600),
    "max_depth": randint(3, 12),
    "learning_rate": loguniform(1e-3, 0.3),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_lambda": loguniform(1e-3, 10),
    "reg_alpha": loguniform(1e-4, 1),
    "min_child_weight": randint(1, 10),
}


def _scale_pos_weight(y: np.ndarray) -> float:
    """XGBoost's preferred class-imbalance lever (analogous to class_weight)."""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return (neg / pos) if pos > 0 else 1.0


def train(
    csv_path: Path = Path("data/processed/features.csv"),
    out_model: Path = Path("models/xgboost.joblib"),
    out_metrics: Path = Path("models/metrics/xgboost.json"),
    n_iter: int = 40,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    X, y, _ = load_features(csv_path)
    X_train, X_test, y_train, y_test = stratified_split(X, y, random_state=random_state)

    scale = _scale_pos_weight(y_train.to_numpy())
    LOG.info("scale_pos_weight=%.3f", scale)

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale,
        random_state=random_state,
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        xgb,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        cv=cv_splitter(cv_folds, random_state),
        scoring="f1",
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )
    search.fit(X_train, y_train)
    LOG.info("best params: %s (f1_cv=%.4f)", search.best_params_, search.best_score_)

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    metrics = evaluate_classifier("xgboost", y_test, y_pred, y_proba)
    save_metrics(metrics, out_metrics)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out_model)
    LOG.info("model → %s", out_model)

    return {"best_params": search.best_params_, "cv_f1": float(search.best_score_), "metrics": metrics}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost classifier.")
    p.add_argument("--csv", type=Path, default=Path("data/processed/features.csv"))
    p.add_argument("--out-model", type=Path, default=Path("models/xgboost.joblib"))
    p.add_argument("--out-metrics", type=Path, default=Path("models/metrics/xgboost.json"))
    p.add_argument("--n-iter", type=int, default=40)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    out = train(
        csv_path=args.csv,
        out_model=args.out_model,
        out_metrics=args.out_metrics,
        n_iter=args.n_iter,
        cv_folds=args.cv,
    )
    print(f"xgboost: f1={out['metrics'].f1:.4f} roc_auc={out['metrics'].roc_auc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
