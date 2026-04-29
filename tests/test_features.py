"""Smoke tests for the feature-extraction layer.

Synthetic HTML inputs — these run without any scraped data on disk, so they
work in a clean clone and on CI.
"""

from __future__ import annotations

import pandas as pd
from bs4 import BeautifulSoup

from src.features.build_features import url_key
from src.features.content_features import (
    extract_basic,
    fit_tfidf,
    flesch_reading_ease,
    keyword_density,
    transform_tfidf,
)
from src.features.metadata_features import extract_metadata
from src.features.structural_features import extract_structural
from src.recommendations.recommend import recommend
from src.scraping.serp_client import derive_query_from_title


SAMPLE_HTML = """
<html>
  <head>
    <title>asyncio — Asynchronous I/O — Python 3.12</title>
    <meta name="description" content="A concurrent Python framework using async/await syntax.">
  </head>
  <body>
    <h1>asyncio</h1>
    <h2>Tasks</h2><h2>Streams</h2>
    <h3>Subprocess</h3>
    <p>asyncio is a library to write concurrent code using the async/await syntax.</p>
    <p>It is used as a foundation for multiple Python asynchronous frameworks.</p>
    <a href="https://docs.python.org/3/library/asyncio-task.html">Tasks</a>
    <a href="https://example.com/external">External</a>
    <a href="#anchor">Anchor</a>
    <img src="x.png" alt="diagram">
    <img src="y.png">
  </body>
</html>
"""


def _soup() -> BeautifulSoup:
    return BeautifulSoup(SAMPLE_HTML, "lxml")


def test_url_key_normalizes_scheme_and_trailing_slash():
    assert url_key("https://Docs.Python.org/3/library/asyncio.html/") == "docs.python.org/3/library/asyncio.html"
    assert url_key("http://docs.python.org/3/library/asyncio.html") == url_key("https://docs.python.org/3/library/asyncio.html")


def test_derive_query_from_title_strips_site_name():
    assert derive_query_from_title("asyncio — Asynchronous I/O — Python 3.12") == "asyncio"
    assert derive_query_from_title("useEffect – React") == "useEffect"
    assert derive_query_from_title("Pods | Kubernetes") == "Pods"


def test_extract_basic_produces_expected_keys():
    text = _soup().get_text(" ", strip=True)
    feats = extract_basic(text, query="asyncio")
    assert set(feats) == {"text_length", "word_count", "sentence_count", "flesch_reading_ease", "keyword_density"}
    assert feats["word_count"] > 0
    assert feats["sentence_count"] >= 1
    assert 0 <= feats["keyword_density"] <= 1


def test_keyword_density_handles_empty_query():
    assert keyword_density("any text here", "") == 0.0
    assert keyword_density("", "asyncio") == 0.0


def test_flesch_returns_zero_for_empty_input():
    assert flesch_reading_ease("") == 0.0
    assert flesch_reading_ease(".") == 0.0


def test_tfidf_round_trip_on_tiny_corpus():
    corpus = [
        "asyncio coroutines event loop",
        "react hooks state effect",
        "kubernetes pods services deployment",
        "python asyncio tasks streams",
    ]
    vec = fit_tfidf(corpus, max_features=5)
    row = transform_tfidf("asyncio coroutines tasks", vec)
    assert all(k.startswith("tfidf_") for k in row)
    assert any(v > 0 for v in row.values())


def test_extract_metadata_flags_keyword_in_title():
    feats = extract_metadata(_soup(), query="asyncio")
    assert feats["title_length"] > 0
    assert feats["has_meta_description"] == 1.0
    assert feats["meta_description_length"] > 10
    assert feats["keyword_in_title"] == 1.0


def test_extract_metadata_no_meta_description():
    soup = BeautifulSoup("<html><head><title>Plain page</title></head><body>x</body></html>", "lxml")
    feats = extract_metadata(soup, query="plain")
    assert feats["has_meta_description"] == 0.0
    assert feats["meta_description_length"] == 0.0


def test_extract_structural_counts_match():
    feats = extract_structural(_soup(), page_url="https://docs.python.org/3/library/asyncio.html")
    assert feats["h1_count"] == 1.0
    assert feats["h2_count"] == 2.0
    assert feats["h3_count"] == 1.0
    assert feats["internal_link_count"] == 1.0
    assert feats["external_link_count"] == 1.0
    assert feats["image_count"] == 2.0
    assert feats["alt_text_coverage"] == 0.5


def test_recommend_pads_to_three_minimum():
    row = pd.Series({
        "title_length": 45.0, "has_meta_description": 1.0, "meta_description_length": 140.0,
        "keyword_in_title": 1.0, "h2_count": 5.0, "image_count": 2.0, "alt_text_coverage": 1.0,
        "keyword_density": 0.02,
    })
    out = recommend(row, query="asyncio", min_suggestions=3)
    assert len(out) >= 3
    for s in out:
        assert "action" in s and "target_feature" in s and "why" in s


def test_recommend_emits_concrete_alt_text_count():
    row = pd.Series({
        "title_length": 45.0, "has_meta_description": 1.0, "meta_description_length": 140.0,
        "keyword_in_title": 1.0, "h2_count": 5.0, "image_count": 7.0, "alt_text_coverage": 3 / 7,
        "keyword_density": 0.02,
    })
    out = recommend(row, query="asyncio")
    msgs = [s["action"] for s in out]
    assert any("alt text" in m for m in msgs)
    assert any("4" in m and "7" in m for m in msgs)
