"""Pipeline orchestrator — produces ``data/processed/features.csv``.

Walks ``data/raw/<domain>/*.json`` sidecars + corresponding HTML files,
joins the per-page query from ``data/interim/queries.csv``, computes content
+ metadata + structural features, fits a corpus-wide TF-IDF projection,
joins the ``is_top_10`` label from ``data/interim/serp.csv``, and writes
the final flat feature matrix.

Note: graph features (PageRank, HITS, degrees, clustering) are merged in by
``src.graph.graph_features`` after this step, since they require building
the link graph from JSON sidecar ``outbound_links``.

CLI:
    python -m src.features.build_features
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.parse
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from src.features.content_features import extract_basic, fit_tfidf, transform_tfidf
from src.features.metadata_features import extract_metadata
from src.features.structural_features import extract_structural

LOG = logging.getLogger("build_features")


def url_key(url: str) -> str:
    """Match key for comparing scraped page URLs to SERP result URLs.
    Drops scheme, query, and fragment; lowercases host; strips trailing
    slashes. Tolerates http/https mismatches and trailing-query noise."""
    p = urllib.parse.urlparse(url)
    return (p.netloc.lower() + p.path.rstrip("/")).lower()


def _read_html(html_path: Path) -> str:
    try:
        return html_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        LOG.warning("read failed %s (%s)", html_path, exc)
        return ""


def build(
    raw_dir: Path = Path("data/raw"),
    queries_csv: Path = Path("data/interim/queries.csv"),
    serp_csv: Path = Path("data/interim/serp.csv"),
    out_csv: Path = Path("data/processed/features.csv"),
    tfidf_max_features: int = 50,
) -> int:
    """Build the per-page feature matrix and write it to ``out_csv``.
    Returns the number of rows written.
    """
    if not queries_csv.exists():
        raise SystemExit(f"missing {queries_csv} — run `python -m src.scraping.serp_client build-queries`")
    if not serp_csv.exists():
        raise SystemExit(f"missing {serp_csv} — run `python -m src.scraping.serp_client fetch`")

    queries_df = pd.read_csv(queries_csv)
    queries_by_url = {url_key(u): (qid, q) for qid, q, u in queries_df.itertuples(index=False)}

    serp_df = pd.read_csv(serp_csv)
    top10_by_query: dict[str, set[str]] = {}
    for qid, group in serp_df.groupby("query_id"):
        top10_by_query[qid] = {url_key(u) for u in group["url"].astype(str)}

    rows: list[dict] = []
    texts: list[str] = []
    skipped_no_query = skipped_no_html = 0

    for json_path in sorted(raw_dir.glob("*/*.json")):
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("skip %s (%s)", json_path, exc)
            continue
        page_url = meta.get("url", "")
        if not page_url:
            continue

        key = url_key(page_url)
        if key not in queries_by_url:
            skipped_no_query += 1
            continue
        qid, query = queries_by_url[key]

        html_path = json_path.with_suffix(".html")
        if not html_path.exists():
            skipped_no_html += 1
            continue
        html = _read_html(html_path)
        if not html:
            skipped_no_html += 1
            continue

        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(" ", strip=True)

        basic = extract_basic(text, query)
        meta_feats = extract_metadata(soup, query)
        struct_feats = extract_structural(soup, page_url)

        top10 = top10_by_query.get(qid, set())
        is_top_10 = 1 if key in top10 else 0

        row: dict = {
            "url": page_url,
            "domain": urllib.parse.urlparse(page_url).netloc,
            "query_id": qid,
            "query": query,
            **basic,
            **meta_feats,
            **struct_feats,
            "is_top_10": is_top_10,
        }
        rows.append(row)
        texts.append(text)

    if not rows:
        raise SystemExit("no feature rows produced — verify data/raw/, queries.csv, serp.csv all populated")

    LOG.info(
        "rows=%d (skipped: no_query=%d no_html=%d) — fitting TF-IDF (max_features=%d)",
        len(rows), skipped_no_query, skipped_no_html, tfidf_max_features,
    )
    vec = fit_tfidf(texts, max_features=tfidf_max_features)
    for row, text in zip(rows, texts):
        row.update(transform_tfidf(text, vec))

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    LOG.info("wrote %d rows × %d cols → %s", len(df), df.shape[1], out_csv)
    return len(df)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the per-page feature matrix.")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--queries", type=Path, default=Path("data/interim/queries.csv"))
    p.add_argument("--serp", type=Path, default=Path("data/interim/serp.csv"))
    p.add_argument("--out", type=Path, default=Path("data/processed/features.csv"))
    p.add_argument("--tfidf-max-features", type=int, default=50)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    n = build(
        raw_dir=args.raw_dir,
        queries_csv=args.queries,
        serp_csv=args.serp,
        out_csv=args.out,
        tfidf_max_features=args.tfidf_max_features,
    )
    print(f"wrote {n} rows → {args.out}")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
