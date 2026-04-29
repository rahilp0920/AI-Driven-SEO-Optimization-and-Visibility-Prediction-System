"""Smoke tests for src/models/evaluate.py + thin model imports.

We don't train full models in the test suite (the sweep is slow + stochastic).
We verify load/split/eval helpers handle synthetic data correctly and that
every model module imports cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (
    evaluate_classifier,
    load_all_metrics,
    load_features,
    save_metrics,
    stratified_split,
)


def _synthetic_features(tmp_path: Path, n: int = 80, n_pos: int = 20) -> Path:
    """Write a tiny features.csv that mirrors the real schema."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "url": [f"https://example.com/p{i}" for i in range(n)],
        "domain": ["example.com"] * n,
        "query_id": ["abc123"] * n,
        "query": ["topic"] * n,
        "title_length": rng.integers(20, 80, size=n),
        "h2_count": rng.integers(0, 10, size=n),
        "pagerank": rng.random(n),
        "tfidf_python": rng.random(n),
        "is_top_10": [1] * n_pos + [0] * (n - n_pos),
    })
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    p = tmp_path / "features.csv"
    df.to_csv(p, index=False)
    return p


def test_load_features_drops_identifiers(tmp_path: Path):
    csv = _synthetic_features(tmp_path)
    X, y, ids = load_features(csv)
    assert "url" not in X.columns
    assert "domain" not in X.columns
    assert "query_id" not in X.columns
    assert "query" not in X.columns
    assert "is_top_10" not in X.columns
    assert set(["url", "domain", "query_id", "query"]).issubset(set(ids.columns))
    assert len(X) == len(y) == 80


def test_load_features_raises_on_missing_target(tmp_path: Path):
    p = tmp_path / "no_target.csv"
    pd.DataFrame({"url": ["x"], "feature_a": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_features(p)


def test_stratified_split_preserves_class_ratio(tmp_path: Path):
    csv = _synthetic_features(tmp_path, n=80, n_pos=20)
    X, y, _ = load_features(csv)
    X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=0.25)
    assert len(X_tr) + len(X_te) == 80
    assert abs(y_tr.mean() - y_te.mean()) < 0.05


def test_evaluate_classifier_perfect_predictions():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.95])
    m = evaluate_classifier("perfect", y_true, y_pred, y_proba)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.roc_auc == 1.0
    assert m.n_test == 6
    assert m.n_pos_test == 3


def test_evaluate_classifier_all_negative_predictions():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    m = evaluate_classifier("all_neg", y_true, y_pred, y_proba=None)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.roc_auc == 0.0  # falls back when y_proba is None


def test_save_and_load_metrics_round_trip(tmp_path: Path):
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4])
    m = evaluate_classifier("toy", y_true, y_pred, y_proba)
    out = tmp_path / "metrics" / "toy.json"
    save_metrics(m, out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["model"] == "toy"
    assert loaded["n_test"] == 4
    df = load_all_metrics(out.parent)
    assert len(df) == 1
    assert df.iloc[0]["model"] == "toy"


def test_model_modules_import_cleanly():
    """Catch typos / circular imports without running training."""
    import src.models.baseline  # noqa: F401
    import src.models.boosting  # noqa: F401
    import src.models.neural    # noqa: F401
    import src.models.tree_models  # noqa: F401
