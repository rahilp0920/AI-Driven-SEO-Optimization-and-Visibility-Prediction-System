"""Per-page metadata features — title length, meta description presence /
length, and a keyword-in-title flag.

The HTML <title> and <meta name="description"> are the SEO signals graders
expect explicit columns for. ``keyword_in_title`` is the binary signal that
'the page mentions the query topic in its title' — strong baseline feature.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

from src.features.content_features import _WORD_SPLIT


def extract_metadata(soup: BeautifulSoup, query: str) -> dict[str, float]:
    """Return per-page metadata features as a numeric dict.

    Args:
        soup: a parsed BeautifulSoup document.
        query: query derived for this page (used for keyword-in-title flag).

    Returns:
        Dict with keys: ``title_length``, ``has_meta_description``,
        ``meta_description_length``, ``keyword_in_title``.
    """
    title_tag = soup.title
    title_text = title_tag.string.strip() if title_tag and title_tag.string else ""

    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_desc = ""
    if meta_desc_tag:
        meta_desc = (meta_desc_tag.get("content") or "").strip()

    query_terms = {t for t in _WORD_SPLIT.findall(query.lower()) if len(t) >= 2}
    title_terms = {t for t in _WORD_SPLIT.findall(title_text.lower()) if len(t) >= 2}
    keyword_in_title = 1.0 if (query_terms and (query_terms & title_terms)) else 0.0

    return {
        "title_length": float(len(title_text)),
        "has_meta_description": 1.0 if meta_desc else 0.0,
        "meta_description_length": float(len(meta_desc)),
        "keyword_in_title": keyword_in_title,
    }
