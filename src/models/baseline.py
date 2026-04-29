"""Logistic Regression baseline — the linear, easily-defended starting point.

Tuned via RandomizedSearchCV on the C / penalty / class_weight grid using the
shared StratifiedKFold from ``evaluate.cv_splitter``. Saves the fitted model
to ``models/baseline.joblib`` and the held-out metrics to
``models/metrics/baseline.json``.

CLI:
    python -m src.models.baseline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluate import (
    cv_splitter,
    evaluate_classifier,
    load_features,
    save_metrics,
    stratified_split,
)

LOG = logging.getLogger("baseline")

PARAM_DIST = {
    "clf__C": loguniform(1e-3, 1e2),
    "clf__penalty": ["l1", "l2"],
    "clf__class_weight": [None, "balanced"],
}


def train(
    csv_path: Path = Path("data/processed/features.csv"),
    out_model: Path = Path("models/baseline.joblib"),
    out_metrics: Path = Path("models/metrics/baseline.json"),
    n_iter: int = 30,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    X, y, _ = load_features(csv_path)
    X_train, X_test, y_train, y_test = stratified_split(X, y, random_state=random_state)
    LOG.info("train n=%d (pos=%d) test n=%d (pos=%d)", len(X_train), y_train.sum(), len(X_test), y_test.sum())

    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),  # sparse-friendly (TF-IDF cols)
            ("clf", LogisticRegression(solver="liblinear", max_iter=2000, random_state=random_state)),
        ]
    )
    search = RandomizedSearchCV(
        pipe,
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

    metrics = evaluate_classifier("logreg", y_test, y_pred, y_proba)
    save_metrics(metrics, out_metrics)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out_model)
    LOG.info("model → %s", out_model)

    return {"best_params": search.best_params_, "cv_f1": float(search.best_score_), "metrics": metrics}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Logistic Regression baseline.")
    p.add_argument("--csv", type=Path, default=Path("data/processed/features.csv"))
    p.add_argument("--out-model", type=Path, default=Path("models/baseline.joblib"))
    p.add_argument("--out-metrics", type=Path, default=Path("models/metrics/baseline.json"))
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
    print(f"logreg: f1={out['metrics'].f1:.4f} roc_auc={out['metrics'].roc_auc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
