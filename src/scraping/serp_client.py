"""SERP rankings client — derives queries from scraped page titles, fetches
top-10 Google results per query (Brave Search API by default, SerpApi fallback),
and writes ``data/interim/serp.csv`` for downstream label join.

Two CLI subcommands:

    # Walk data/raw/<domain>/*.json sidecars, write data/interim/queries.csv
    python -m src.scraping.serp_client build-queries

    # Read data/interim/queries.csv, fetch top-10 per query, write data/interim/serp.csv
    python -m src.scraping.serp_client fetch

The page-level ``is_top_10`` label (joined in ``src.features.build_features``)
is true iff the page's own URL appears among the top-10 SERP rows for the
query that page generated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import httpx
from dotenv import load_dotenv

load_dotenv()

LOG = logging.getLogger("serp_client")

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
SERPAPI_ENDPOINT = "https://serpapi.com/search"
DEFAULT_TIMEOUT = 30.0
DEFAULT_DELAY = 1.1  # Brave free tier ~1 req/s.

_TITLE_TAIL_PATTERN = re.compile(r"\s*[—–|·:\-]\s*[^—–|·:\-]+$")


def _query_id(query: str) -> str:
    return hashlib.sha1(query.lower().encode("utf-8")).hexdigest()[:16]


def derive_query_from_title(title: str) -> str:
    """Strip trailing site-name boilerplate from an HTML ``<title>`` tag and
    return the remainder as the topic query. Examples::

        "asyncio — Asynchronous I/O — Python 3.12"  →  "asyncio"
        "useEffect – React"                          →  "useEffect"
        "Pods | Kubernetes"                          →  "Pods"
    """
    cleaned = (title or "").strip()
    # Strip the rightmost separator-bounded chunk twice so two-segment tails
    # ("foo — bar — Site Name") collapse to "foo".
    for _ in range(2):
        new = _TITLE_TAIL_PATTERN.sub("", cleaned).strip()
        if new and new != cleaned:
            cleaned = new
        else:
            break
    return cleaned


def build_queries_from_scrape(
    raw_dir: Path = Path("data/raw"),
    out_csv: Path = Path("data/interim/queries.csv"),
    min_query_len: int = 3,
    max_query_len: int = 80,
) -> int:
    """Walk ``data/raw/<domain>/*.json`` sidecars, derive one query per page
    from the page's ``<title>``, and write the queries CSV. Returns the
    number of (query, source_url) rows written. Pages with no usable title
    are skipped (logged at DEBUG).
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, str]] = []
    seen_urls: set[str] = set()

    for json_path in sorted(raw_dir.glob("*/*.json")):
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("skip %s (%s)", json_path, exc)
            continue
        url = meta.get("url", "")
        title = meta.get("title", "")
        query = derive_query_from_title(title)
        if not url or url in seen_urls:
            continue
        if not (min_query_len <= len(query) <= max_query_len):
            LOG.debug("skip %s (query=%r len=%d)", url, query, len(query))
            continue
        seen_urls.add(url)
        rows.append((_query_id(query), query, url))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query", "source_url"])
        writer.writerows(rows)

    LOG.info("wrote %d queries → %s", len(rows), out_csv)
    return len(rows)


def _brave_search(client: httpx.Client, query: str, key: str, count: int = 10) -> list[dict]:
    headers = {"X-Subscription-Token": key, "Accept": "application/json"}
    params = {"q": query, "count": count, "result_filter": "web"}
    resp = client.get(BRAVE_ENDPOINT, headers=headers, params=params, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    web = resp.json().get("web", {}).get("results", [])
    return [
        {"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("description", "")}
        for r in web[:count]
    ]


def _serpapi_search(client: httpx.Client, query: str, key: str, count: int = 10) -> list[dict]:
    params = {"engine": "google", "q": query, "num": count, "api_key": key}
    resp = client.get(SERPAPI_ENDPOINT, params=params, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    organic = resp.json().get("organic_results", [])
    return [
        {"url": r.get("link", ""), "title": r.get("title", ""), "snippet": r.get("snippet", "")}
        for r in organic[:count]
    ]


def _read_queries(queries_csv: Path) -> Iterator[tuple[str, str, str]]:
    with queries_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row["query_id"], row["query"], row["source_url"]


def fetch_serp(
    queries_csv: Path = Path("data/interim/queries.csv"),
    out_csv: Path = Path("data/interim/serp.csv"),
    delay: float = DEFAULT_DELAY,
    count: int = 10,
    resume: bool = True,
) -> int:
    """Fetch top-``count`` results for every query in ``queries_csv`` and
    write rows to ``out_csv``. Provider auto-selected from env vars
    (``BRAVE_SEARCH_KEY`` preferred; ``SERPAPI_KEY`` fallback). When
    ``resume=True`` (default), already-fetched query_ids in an existing
    output file are skipped — important for free-tier quota safety.

    Returns the number of (query, rank, url) rows written this run.
    """
    brave_key = os.getenv("BRAVE_SEARCH_KEY", "").strip()
    serpapi_key = os.getenv("SERPAPI_KEY", "").strip()
    if not brave_key and not serpapi_key:
        raise SystemExit(
            "no SERP API key found — set BRAVE_SEARCH_KEY or SERPAPI_KEY in .env"
        )
    provider = "brave" if brave_key else "serpapi"
    LOG.info("SERP provider: %s", provider)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    if resume and file_exists:
        with out_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_ids.add(row["query_id"])
        LOG.info("resume: skipping %d already-fetched query_ids", len(done_ids))

    write_header = not file_exists
    rows_written = 0

    with httpx.Client() as client, out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["query_id", "query", "source_url", "rank", "url", "title", "snippet"])

        for qid, query, source_url in _read_queries(queries_csv):
            if qid in done_ids:
                continue
            try:
                if provider == "brave":
                    results = _brave_search(client, query, brave_key, count=count)
                else:
                    results = _serpapi_search(client, query, serpapi_key, count=count)
            except httpx.HTTPError as exc:
                LOG.warning("SERP fetch failed (qid=%s, query=%r): %s", qid, query, exc)
                time.sleep(delay)
                continue

            for rank, r in enumerate(results, start=1):
                writer.writerow([qid, query, source_url, rank, r["url"], r["title"], r["snippet"]])
                rows_written += 1
            f.flush()
            if rows_written and rows_written % 100 == 0:
                LOG.info("rows written so far: %d", rows_written)
            time.sleep(delay)

    LOG.info("done — %d rows written → %s", rows_written, out_csv)
    return rows_written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SERP rankings client.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    bq = sub.add_parser("build-queries", help="Derive queries from scraped JSON sidecars.")
    bq.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    bq.add_argument("--out", type=Path, default=Path("data/interim/queries.csv"))

    fc = sub.add_parser("fetch", help="Fetch top-10 SERP results per query.")
    fc.add_argument("--queries", type=Path, default=Path("data/interim/queries.csv"))
    fc.add_argument("--out", type=Path, default=Path("data/interim/serp.csv"))
    fc.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    fc.add_argument("--count", type=int, default=10)
    fc.add_argument("--no-resume", action="store_true")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.cmd == "build-queries":
        n = build_queries_from_scrape(raw_dir=args.raw_dir, out_csv=args.out)
        print(f"wrote {n} queries → {args.out}")
        return 0 if n > 0 else 1
    if args.cmd == "fetch":
        n = fetch_serp(
            queries_csv=args.queries,
            out_csv=args.out,
            delay=args.delay,
            count=args.count,
            resume=not args.no_resume,
        )
        print(f"wrote {n} SERP rows → {args.out}")
        return 0 if n > 0 else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
