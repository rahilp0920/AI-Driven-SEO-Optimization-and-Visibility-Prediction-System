"""Asynchronous, robots.txt-aware crawler for developer documentation sites.

Crawls within a single domain starting from a seed URL via BFS, respecting
robots.txt and a configurable per-domain rate limit. Saves raw HTML to
`data/raw/<domain>/<sha1>.html` and a sidecar `<sha1>.json` with URL,
fetch timestamp, status code, and discovered outbound links (used later
by `src.graph.build_graph` to construct the link graph).

CLI:
    python -m src.scraping.doc_scraper --domain docs.python.org --limit 300

Designed for the ~1500-page deadline-scoped scope (CIS 2450 TA Ricky Gong's
2026-03-29 sanctioned pivot from the original 50K-row plan).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.robotparser
from collections import deque
from pathlib import Path
from typing import Iterable

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

LOG = logging.getLogger("doc_scraper")

DEFAULT_USER_AGENT = os.getenv(
    "SCRAPER_USER_AGENT",
    "AIDrivenSEOResearchBot/1.0 (+contact: rahilp07@seas.upenn.edu)",
)
DEFAULT_DELAY = float(os.getenv("SCRAPER_DELAY_SECONDS", "1.0"))
DEFAULT_TIMEOUT = 20.0

# File extensions we never fetch (binary/non-HTML).
SKIP_EXTENSIONS = (
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".png", ".jpg", ".jpeg",
    ".gif", ".svg", ".ico", ".css", ".js", ".woff", ".woff2", ".ttf",
    ".eot", ".mp4", ".webm", ".mp3", ".wav",
)


@dataclasses.dataclass
class FetchResult:
    """One fetched page's metadata, persisted alongside the raw HTML."""

    url: str
    status: int
    fetched_at: str
    content_length: int
    title: str
    outbound_links: list[str]


def _sanitize_id(url: str) -> str:
    """Stable filesystem-safe id for a URL — sha1 keeps filenames short."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _normalize_url(url: str) -> str:
    """Drop fragments and lowercase the scheme+host so we don't double-fetch."""
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, parsed.params, parsed.query, "")
    )


def _same_domain(url: str, domain: str) -> bool:
    netloc = urllib.parse.urlparse(url).netloc.lower()
    return netloc == domain.lower() or netloc.endswith("." + domain.lower())


def _looks_like_html(url: str) -> bool:
    path = urllib.parse.urlparse(url).path.lower()
    return not path.endswith(SKIP_EXTENSIONS)


async def _load_robots(client: httpx.AsyncClient, domain: str) -> urllib.robotparser.RobotFileParser:
    """Fetch /robots.txt for the domain. Returns an open parser if the file
    is missing or unreachable (fail-open is standard)."""
    rp = urllib.robotparser.RobotFileParser()
    robots_url = f"https://{domain}/robots.txt"
    try:
        resp = await client.get(robots_url, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
            LOG.info("robots.txt loaded for %s (%d bytes)", domain, len(resp.text))
        else:
            LOG.warning("robots.txt not found for %s (status %d) — defaulting to allow", domain, resp.status_code)
            rp.parse([])
    except (httpx.HTTPError, OSError) as exc:
        LOG.warning("robots.txt fetch failed for %s (%s) — defaulting to allow", domain, exc)
        rp.parse([])
    return rp


def _extract_links(html: str, base_url: str) -> list[str]:
    """Return absolute, normalized outbound links from <a href> tags."""
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        joined = urllib.parse.urljoin(base_url, a["href"])
        if joined.startswith(("http://", "https://")):
            links.append(_normalize_url(joined))
    return links


def _extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return soup.title.string.strip()[:300]
    return ""


async def crawl_domain(
    domain: str,
    seed_path: str = "/",
    limit: int = 300,
    delay: float = DEFAULT_DELAY,
    output_dir: Path = Path("data/raw"),
    user_agent: str = DEFAULT_USER_AGENT,
) -> int:
    """BFS-crawl one domain up to `limit` HTML pages, sleeping `delay`s between
    requests. Returns the number of pages successfully saved.

    Args:
        domain: bare hostname, e.g. ``docs.python.org`` (no scheme, no path).
        seed_path: starting path on the domain.
        limit: max pages to fetch.
        delay: seconds to sleep between requests (per-domain rate limit).
        output_dir: parent directory; per-domain subdirectory created inside.
        user_agent: User-Agent header sent to the server.

    Returns:
        Count of pages successfully saved (status 200 + non-empty body).
    """
    domain_dir = output_dir / domain.replace(":", "_")
    domain_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"}
    seed = _normalize_url(f"https://{domain}{seed_path}")

    saved = 0
    seen: set[str] = set()
    queue: deque[str] = deque([seed])

    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=DEFAULT_TIMEOUT) as client:
        rp = await _load_robots(client, domain)

        while queue and saved < limit:
            url = queue.popleft()
            if url in seen:
                continue
            seen.add(url)

            if not _same_domain(url, domain) or not _looks_like_html(url):
                continue
            if not rp.can_fetch(user_agent, url):
                LOG.info("robots.txt disallows %s — skipping", url)
                continue

            try:
                resp = await client.get(url)
            except httpx.HTTPError as exc:
                LOG.warning("fetch failed %s (%s)", url, exc)
                continue

            if resp.status_code != 200 or "html" not in resp.headers.get("content-type", "").lower():
                LOG.debug("skip %s (status=%d, ct=%s)", url, resp.status_code, resp.headers.get("content-type", ""))
                await asyncio.sleep(delay)
                continue

            html = resp.text
            if not html.strip():
                await asyncio.sleep(delay)
                continue

            page_id = _sanitize_id(url)
            html_path = domain_dir / f"{page_id}.html"
            meta_path = domain_dir / f"{page_id}.json"

            outbound = _extract_links(html, url)
            title = _extract_title(html)

            html_path.write_text(html, encoding="utf-8")
            meta = FetchResult(
                url=url,
                status=resp.status_code,
                fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                content_length=len(html),
                title=title,
                outbound_links=outbound,
            )
            meta_path.write_text(json.dumps(dataclasses.asdict(meta), indent=2), encoding="utf-8")

            saved += 1
            if saved % 25 == 0:
                LOG.info("[%s] %d/%d saved", domain, saved, limit)

            for link in outbound:
                if link not in seen and _same_domain(link, domain) and _looks_like_html(link):
                    queue.append(link)

            await asyncio.sleep(delay)

    LOG.info("[%s] done — %d pages saved (queue exhausted=%s)", domain, saved, not queue)
    return saved


async def crawl_many(domain_limits: Iterable[tuple[str, int]], **kwargs: object) -> dict[str, int]:
    """Crawl multiple domains sequentially (sequential keeps total request rate
    polite even though each domain has its own delay). Returns a dict of
    ``{domain: pages_saved}``."""
    results: dict[str, int] = {}
    for domain, limit in domain_limits:
        try:
            results[domain] = await crawl_domain(domain=domain, limit=limit, **kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 — keep going on per-domain failures
            LOG.exception("crawl_domain crashed for %s: %s", domain, exc)
            results[domain] = 0
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async robots-aware crawler for developer documentation sites.")
    p.add_argument("--domain", required=True, help="Bare hostname, e.g. docs.python.org")
    p.add_argument("--seed", default="/", help="Path to start crawling from (default: /).")
    p.add_argument("--limit", type=int, default=300, help="Max pages to fetch (default: 300).")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between requests (default: env or 1.0).")
    p.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Output directory (default: data/raw).")
    p.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="Override scraper User-Agent.")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    saved = asyncio.run(
        crawl_domain(
            domain=args.domain,
            seed_path=args.seed,
            limit=args.limit,
            delay=args.delay,
            output_dir=args.output_dir,
            user_agent=args.user_agent,
        )
    )
    print(f"saved {saved} pages from {args.domain} → {args.output_dir / args.domain}")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
