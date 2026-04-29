"""Random Forest — bagging tree ensemble (model family #2 in the sweep).

Tuned via RandomizedSearchCV on n_estimators / max_depth / min_samples_*.
Output mirrors ``baseline.py`` so the comparison notebook can concat metrics
JSONs uniformly.

CLI:
    python -m src.models.tree_models
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.models.evaluate import (
    cv_splitter,
    evaluate_classifier,
    load_features,
    save_metrics,
    stratified_split,
)

LOG = logging.getLogger("rf")

PARAM_DIST = {
    "n_estimators": randint(100, 600),
    "max_depth": [None, 4, 8, 16, 32],
    "min_samples_split": randint(2, 12),
    "min_samples_leaf": randint(1, 8),
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": [None, "balanced", "balanced_subsample"],
}


def train(
    csv_path: Path = Path("data/processed/features.csv"),
    out_model: Path = Path("models/random_forest.joblib"),
    out_metrics: Path = Path("models/metrics/random_forest.json"),
    n_iter: int = 30,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    X, y, _ = load_features(csv_path)
    X_train, X_test, y_train, y_test = stratified_split(X, y, random_state=random_state)

    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        rf,
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

    metrics = evaluate_classifier("random_forest", y_test, y_pred, y_proba)
    save_metrics(metrics, out_metrics)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out_model)
    LOG.info("model → %s", out_model)

    return {"best_params": search.best_params_, "cv_f1": float(search.best_score_), "metrics": metrics}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Random Forest classifier.")
    p.add_argument("--csv", type=Path, default=Path("data/processed/features.csv"))
    p.add_argument("--out-model", type=Path, default=Path("models/random_forest.joblib"))
    p.add_argument("--out-metrics", type=Path, default=Path("models/metrics/random_forest.json"))
    p.add_argument("--n-iter", type=int, default=30)
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
    print(f"random_forest: f1={out['metrics'].f1:.4f} roc_auc={out['metrics'].roc_auc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
