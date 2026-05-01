"""Class-imbalance handling for the SEO ranking dataset.

The committed feature matrix is class-imbalanced (≈ 26 % top-10 vs 74 % not),
which biases unweighted classifiers toward the majority class. Two response
strategies are exposed here:

* ``random_oversample(df, target_col)`` — duplicate minority-class rows with
  replacement until both classes are equally represented. Cheap, deterministic
  with a seed, and the strategy explicitly listed in the CIS 2450 §3
  "Imbalance data" rubric concept.
* ``bootstrap_augment(df, target_col, factor, noise)`` — bootstrap-sample with
  small Gaussian perturbation on numeric columns. Useful when the model
  benefits from variety (random oversampling alone gives the model exact
  duplicates, which doesn't add new signal).

Both helpers preserve identifier columns (``url``, ``domain``, ``query_id``,
``query``) on the duplicated rows; only the rebalanced feature matrix changes.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

LOG = logging.getLogger("balance")

DEFAULT_NUMERIC_NOISE = 0.02  # 2 % relative jitter on numeric features
IDENTIFIER_COLS = ("url", "domain", "query_id", "query")


def _classes(df: pd.DataFrame, target_col: str) -> dict[int, pd.DataFrame]:
    """Return ``{class_label: rows_in_that_class}``."""
    return {int(c): df[df[target_col] == c] for c in sorted(df[target_col].unique())}


def random_oversample(
    df: pd.DataFrame,
    target_col: str = "is_top_10",
    random_state: int = 42,
) -> pd.DataFrame:
    """Duplicate minority-class rows with replacement until classes match.

    Returns a new DataFrame; input is unchanged. Class proportions are equal
    after the call. Useful for the imbalance-aware modeling pass without
    touching the underlying scrape.
    """
    classes = _classes(df, target_col)
    target_n = max(len(g) for g in classes.values())
    parts: list[pd.DataFrame] = []
    rng = np.random.default_rng(random_state)
    for cls, group in classes.items():
        if len(group) == target_n:
            parts.append(group)
            continue
        idx = rng.integers(low=0, high=len(group), size=target_n - len(group))
        parts.append(pd.concat([group, group.iloc[idx]], ignore_index=True))
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    LOG.info("random_oversample: %d → %d rows (per-class %s)",
             len(df), len(out),
             {c: int((out[target_col] == c).sum()) for c in classes})
    return out


def bootstrap_augment(
    df: pd.DataFrame,
    target_col: str = "is_top_10",
    factor: int = 5,
    noise: float = DEFAULT_NUMERIC_NOISE,
    random_state: int = 42,
    skip_cols: Iterable[str] = IDENTIFIER_COLS,
) -> pd.DataFrame:
    """Bootstrap-sample the dataset to ``factor * len(df)`` rows, jittering
    numeric features by Gaussian noise with sigma ``noise * column_std``.

    Identifier columns (URL, domain, query, query_id) are copied untouched —
    only the numeric features see noise. The target column is also untouched.
    Class proportions are preserved (per-class bootstrap).

    The justification for jitter (vs pure duplication) is that exact copies
    add no new signal at training time — they just change the sample weight.
    Small Gaussian noise produces a smoother local distribution around each
    real point, which acts as a regularizer for the classifiers.
    """
    rng = np.random.default_rng(random_state)
    skip = set(skip_cols) | {target_col}
    numeric_cols = [c for c in df.columns
                    if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    stds = df[numeric_cols].std(ddof=0).fillna(0.0)
    parts: list[pd.DataFrame] = []
    for cls, group in _classes(df, target_col).items():
        n = len(group) * factor
        idx = rng.integers(low=0, high=len(group), size=n)
        sample = group.iloc[idx].reset_index(drop=True).copy()
        if noise > 0 and len(numeric_cols):
            jitter = rng.normal(loc=0.0, scale=noise, size=(n, len(numeric_cols)))
            jitter = jitter * stds.values  # column-wise std-scaled noise
            sample[numeric_cols] = sample[numeric_cols].values + jitter
            # Keep counts integer-valued and non-negative where the source was.
            for c in numeric_cols:
                col = sample[c]
                if (df[c] >= 0).all():
                    sample[c] = col.clip(lower=0)
                if (df[c].dropna() % 1 == 0).all():
                    sample[c] = col.round().astype(float)
        parts.append(sample)
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    LOG.info("bootstrap_augment: %d → %d rows (factor=%d, noise=%.3f)",
             len(df), len(out), factor, noise)
    return out
