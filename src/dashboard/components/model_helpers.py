"""Helpers for the Models tab — regenerates ROC / PR curves and feature
importances from saved sklearn models without retraining.

Why regenerate curves at runtime? Saving the held-out predictions for every
model would mean shipping more parquet files; instead we re-create the same
stratified split (random_state=42) and call ``predict_proba`` on the loaded
estimator. The cost is one prediction pass over 20 rows — well under a
second per model — and the dashboard always shows numbers consistent with
the saved metrics JSON.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger("model_helpers")


def stratified_test_split(features_csv: Path, random_state: int = 42, test_size: float = 0.2
                          ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame] | None:
    """Recreate the same train/test split used during model training so the
    dashboard's curves match the saved metrics. Returns (X_test, y_test, idents)
    or None if the CSV is missing.
    """
    if not features_csv.exists():
        return None
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(features_csv)
    if "is_top_10" not in df.columns:
        return None
    NON_FEATURE = ("url", "domain", "query_id", "query")
    idents = df[[c for c in NON_FEATURE if c in df.columns]].copy()
    y = df["is_top_10"].astype(int)
    drop_cols = list(NON_FEATURE) + ["is_top_10"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number]).fillna(0.0)

    _, X_te, _, y_te, _, idents_te = train_test_split(
        X, y, idents, test_size=test_size, random_state=random_state, stratify=y,
    )
    return X_te, y_te, idents_te


def model_curves(models: dict[str, Any], features_csv: Path
                 ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, float]],
                            dict[str, tuple[np.ndarray, np.ndarray, float]]]:
    """Return ({model: (fpr, tpr, auc)}, {model: (precision, recall, ap)})
    for every model that exposes ``predict_proba``. Skips silently on errors
    so a single broken model doesn't blank the whole tab."""
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
    )

    split = stratified_test_split(features_csv)
    if split is None:
        return {}, {}
    X_te, y_te, _ = split
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    pr_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for name, model in models.items():
        try:
            proba = model.predict_proba(X_te)[:, 1]
            fpr, tpr, _ = roc_curve(y_te, proba)
            auc = float(roc_auc_score(y_te, proba))
            roc_curves[name] = (fpr, tpr, auc)
            precision, recall, _ = precision_recall_curve(y_te, proba)
            ap = float(average_precision_score(y_te, proba))
            pr_curves[name] = (precision, recall, ap)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("curves failed for %s: %s", name, exc)
    return roc_curves, pr_curves


def model_feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    """Best-effort extraction of feature importance from any model.

    Tries (in order): ``feature_importances_`` (tree models) → first-step
    coefficient (sklearn Pipeline) → top-level ``coef_`` (linear models).
    Returns ``{}`` if nothing usable is found.
    """
    if model is None:
        return {}
    # Direct tree-model attribute.
    fi = getattr(model, "feature_importances_", None)
    if fi is not None and len(fi) == len(feature_names):
        return dict(zip(feature_names, [float(x) for x in fi]))

    # sklearn Pipeline → look at the final estimator.
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            final = model.steps[-1][1]
            return model_feature_importance(final, feature_names)
    except ImportError:
        pass

    coef = getattr(model, "coef_", None)
    if coef is not None:
        arr = np.asarray(coef).reshape(-1)
        if len(arr) == len(feature_names):
            return dict(zip(feature_names, [float(x) for x in arr]))
    return {}


def confusion_for_model(model: Any, features_csv: Path) -> list[list[int]] | None:
    """Recompute the confusion matrix on the saved test split."""
    split = stratified_test_split(features_csv)
    if split is None:
        return None
    X_te, y_te, _ = split
    try:
        from sklearn.metrics import confusion_matrix
        y_pred = model.predict(X_te)
        return confusion_matrix(y_te, y_pred).tolist()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("confusion_matrix failed: %s", exc)
        return None


def feature_columns_from_model(model: Any, fallback: list[str] | None = None) -> list[str]:
    """Best-effort recovery of the column order the model was trained on."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(n) for n in names]
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            for _, step in model.steps:
                names = getattr(step, "feature_names_in_", None)
                if names is not None:
                    return [str(n) for n in names]
    except ImportError:
        pass
    return list(fallback or [])
