"""Per-page content features — text length, Flesch reading ease, keyword
density, and a corpus-fitted TF-IDF projection (top-K terms).

Two API surfaces:

* ``extract_basic(text, query)`` — returns a dict of per-page numeric features
  computable without the full corpus.
* ``fit_tfidf(texts, max_features)`` / ``transform_tfidf(text, vec)`` — the
  TF-IDF step needs a corpus-level fit before per-page transform.

The orchestrator (``src.features.build_features``) calls ``extract_basic`` per
page, fits TF-IDF once on the corpus, then horizontally concats the TF-IDF
projection onto each page's row.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

_VOWEL_GROUP = re.compile(r"[aeiouy]+", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_WORD_SPLIT = re.compile(r"\b[a-zA-Z']+\b")


def _count_syllables(word: str) -> int:
    """Cheap heuristic syllable counter — good enough for a population-level
    Flesch score on ~1500 documents. Overestimates by ~5-10% on rare words;
    that bias is constant across the corpus, so it doesn't bias the model."""
    word = word.lower().rstrip("e")
    groups = _VOWEL_GROUP.findall(word)
    return max(1, len(groups))


def flesch_reading_ease(text: str) -> float:
    """Standard Flesch Reading Ease score: higher = easier to read.

    Returns 0.0 for texts too short to score reliably (≤1 sentence or ≤1 word).
    """
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    words = _WORD_SPLIT.findall(text)
    if len(sentences) < 1 or len(words) < 1:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / max(1, len(sentences))) - 84.6 * (syllables / max(1, len(words)))


def keyword_density(text: str, query: str) -> float:
    """Fraction of word tokens that match any query token (case-insensitive).
    Returns 0.0 for empty text or empty query."""
    words = _WORD_SPLIT.findall(text.lower())
    if not words:
        return 0.0
    query_terms = {t for t in _WORD_SPLIT.findall(query.lower()) if len(t) >= 2}
    if not query_terms:
        return 0.0
    hits = sum(1 for w in words if w in query_terms)
    return hits / len(words)


def extract_basic(text: str, query: str) -> dict[str, float]:
    """Per-page text features that do NOT require corpus context.

    Args:
        text: visible page text (extracted upstream from HTML).
        query: query derived for this page (used for keyword density).

    Returns:
        Dict with keys: ``text_length``, ``word_count``, ``sentence_count``,
        ``flesch_reading_ease``, ``keyword_density``.
    """
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    words = _WORD_SPLIT.findall(text)
    return {
        "text_length": float(len(text)),
        "word_count": float(len(words)),
        "sentence_count": float(len(sentences)),
        "flesch_reading_ease": flesch_reading_ease(text),
        "keyword_density": keyword_density(text, query),
    }


def fit_tfidf(
    texts: Iterable[str],
    max_features: int = 50,
    min_df: int = 2,
    max_df: float = 0.95,
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on the full corpus. Returns the fitted vectorizer.

    Args:
        texts: iterable of per-page text strings.
        max_features: top-K terms by document-frequency-weighted score.
        min_df: minimum document frequency for a term to be kept.
        max_df: maximum document frequency (drops near-stopword terms).
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
    )
    vec.fit(list(texts))
    return vec


def transform_tfidf(text: str, vec: TfidfVectorizer) -> dict[str, float]:
    """Project a single document into TF-IDF space, returning ``{tfidf_<term>: score}``."""
    if not text.strip():
        return {f"tfidf_{t}": 0.0 for t in vec.get_feature_names_out()}
    matrix: Any = vec.transform([text])
    row = matrix.toarray()[0]
    return {f"tfidf_{t}": float(v) for t, v in zip(vec.get_feature_names_out(), row)}
