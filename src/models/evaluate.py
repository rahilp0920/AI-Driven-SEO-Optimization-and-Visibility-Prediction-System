"""Single model evaluator + shared data loader for the model sweep.

All four classifiers (`baseline`, `tree_models`, `boosting`, `neural`) call
``evaluate_classifier`` with predictions on a held-out test set. Returns a
metrics dict suitable for the comparison table in
``notebooks/03_model_comparison.ipynb``.

Why these metrics: the dataset is class-imbalanced (most pages are NOT
top-10). Precision/recall/F1/ROC-AUC/PR-AUC together cover both ranking
quality and threshold-dependent quality; the confusion matrix discloses
which class the model is biased toward.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

LOG = logging.getLogger("evaluate")

NON_FEATURE_COLS = ("url", "domain", "query_id", "query")
TARGET_COL = "is_top_10"


@dataclass
class Metrics:
    """Per-model evaluation metrics in a flat shape that JSON-serializes
    cleanly and concatenates into a single comparison DataFrame."""

    model: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list[list[int]]
    n_test: int
    n_pos_test: int


def load_features(
    csv_path: Path = Path("data/processed/features.csv"),
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load the feature matrix and split into (X, y, identifiers).

    Returns:
        X: numeric feature DataFrame (TF-IDF + content + metadata + structural + graph).
        y: ``is_top_10`` Series.
        identifiers: ``url``/``domain``/``query_id``/``query`` columns kept
            aside for downstream attribution (recommendations + dashboard).
    """
    df = pd.read_csv(csv_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{csv_path} missing target column {TARGET_COL!r}")
    identifiers = df[[c for c in NON_FEATURE_COLS if c in df.columns]].copy()
    y = df[TARGET_COL].astype(int)
    drop_cols = list(NON_FEATURE_COLS) + [TARGET_COL]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    return X, y, identifiers


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Single stratified holdout split. CV is used for hyperparameter search;
    this split is the held-out evaluation set reported in the comparison table."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def cv_splitter(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """Single source of truth for CV folds — every model uses the same one."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def evaluate_classifier(
    name: str,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> Metrics:
    """Compute the standard metric panel from predictions + probabilities.

    Args:
        name: model name (goes into the comparison table).
        y_true: ground-truth labels.
        y_pred: thresholded predictions (0/1).
        y_proba: positive-class probabilities. Pass ``None`` and ROC-AUC /
            PR-AUC fall back to 0.0 (logged).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_proba is None:
        roc_auc = 0.0
        pr_auc = 0.0
        LOG.warning("%s: no probabilities provided — ROC/PR AUC reported as 0.0", name)
    else:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_proba))
            pr_auc = float(average_precision_score(y_true_arr, y_proba))
        except ValueError as exc:
            LOG.warning("%s: AUC fallback (%s)", name, exc)
            roc_auc = 0.0
            pr_auc = 0.0

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).tolist()
    return Metrics(
        model=name,
        precision=float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        recall=float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        f1=float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm,
        n_test=int(len(y_true_arr)),
        n_pos_test=int(y_true_arr.sum()),
    )


def save_metrics(metrics: Metrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    LOG.info("metrics → %s", path)


def load_all_metrics(metrics_dir: Path = Path("models/metrics")) -> pd.DataFrame:
    """Concatenate every ``*.json`` in the metrics directory into one row-per-model
    DataFrame. Used by ``notebooks/03_model_comparison.ipynb``."""
    rows: list[dict[str, Any]] = []
    for f in sorted(metrics_dir.glob("*.json")):
        rows.append(json.loads(f.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)
