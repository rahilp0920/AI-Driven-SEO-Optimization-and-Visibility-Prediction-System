"""Per-page structural features — heading counts, link counts (internal vs
external), image count, and alt-text coverage.

Internal-vs-external link distinction requires the page's own domain, so the
caller passes ``page_url`` and we compare against the URL host of each
``<a href>``.
"""

from __future__ import annotations

import urllib.parse

from bs4 import BeautifulSoup


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()


def extract_structural(soup: BeautifulSoup, page_url: str) -> dict[str, float]:
    """Return per-page structural features as a numeric dict.

    Args:
        soup: a parsed BeautifulSoup document.
        page_url: absolute URL of the page being scored — used to classify
            outbound links as internal vs external.

    Returns:
        Dict with keys: ``h1_count``, ``h2_count``, ``h3_count``,
        ``internal_link_count``, ``external_link_count``, ``image_count``,
        ``alt_text_coverage``.
    """
    h1 = len(soup.find_all("h1"))
    h2 = len(soup.find_all("h2"))
    h3 = len(soup.find_all("h3"))

    page_host = _host(page_url)
    internal = 0
    external = 0
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absolute = urllib.parse.urljoin(page_url, href)
        link_host = _host(absolute)
        if link_host == page_host or link_host.endswith("." + page_host) or page_host.endswith("." + link_host):
            internal += 1
        elif link_host:
            external += 1

    images = soup.find_all("img")
    image_count = len(images)
    if image_count == 0:
        alt_coverage = 0.0
    else:
        with_alt = sum(1 for img in images if (img.get("alt") or "").strip())
        alt_coverage = with_alt / image_count

    return {
        "h1_count": float(h1),
        "h2_count": float(h2),
        "h3_count": float(h3),
        "internal_link_count": float(internal),
        "external_link_count": float(external),
        "image_count": float(image_count),
        "alt_text_coverage": float(alt_coverage),
    }
