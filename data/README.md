# Data — sourcing, pivot, and ethics

## Scope pivot (sanctioned)

> **Per CIS 2450 TA Ricky Gong's email of 2026-03-29, the project was narrowed from the original 50K-row scope to ~1500 developer documentation pages because full-scale free-tier scraping is rate-limited. This is sanctioned, not a deviation.**

This README documents that pivot for graders and reviewers. The narrowed scope is reflected in `data/raw/` (one subdirectory per domain) and the joined feature matrix in `data/processed/features.csv`.

## Sources (two distinct, per CIS 2450 §2)

**Source 1 — Developer documentation HTML.** Crawled with `src/scraping/doc_scraper.py` (async, robots.txt-aware, BFS within a single domain at a configurable per-domain rate limit). Each page is saved to `data/raw/<domain>/<sha1>.html` with a `<sha1>.json` sidecar containing URL, fetch timestamp, status, page title, and outbound links (the latter feeds the link graph in Phase 1 step 7).

**Source 2 — Google SERP rankings.** Top-10 organic results per page-derived query, fetched via Brave Search API (free tier; SerpApi supported as fallback). One query is generated per scraped page from its `<title>` (see `src/scraping/serp_client.py` `build_queries_from_scrape`). Output: `data/interim/serp.csv` with one row per (query, rank). The page-level `is_top_10` label is true if and only if the page's own URL appears in the top-10 for its own derived query.

## Domains targeted

The scraper is run once per domain. The domain set targets ≥5 distinct developer documentation sites to ensure feature distributions aren't a single-site artifact. Candidate domains:

| Domain | Limit | robots.txt status |
|--------|-------|-------------------|
| docs.python.org | 300 | (verified at scrape) |
| developer.mozilla.org | 300 | (verified at scrape) |
| react.dev | 250 | (verified at scrape) |
| nodejs.org | 200 | (verified at scrape) |
| kubernetes.io | 250 | (verified at scrape) |
| fastapi.tiangolo.com | 200 | (verified at scrape) |

Final domain list and per-domain page counts are captured at scrape time in each domain subdirectory. Robots.txt is fetched at the start of every crawl; any disallowed path is skipped (logged at INFO).

## Rate-limit policy

- Default delay: 1.0 second between requests **per domain** (configurable via `SCRAPER_DELAY_SECONDS` env var or `--delay` CLI flag).
- Domains are crawled **sequentially** so total outbound request rate stays at ~1 req/s overall, even when the per-domain delay is shorter.
- User-Agent identifies the project and a contact email (`SCRAPER_USER_AGENT` env var). Default is `AIDrivenSEOResearchBot/1.0 (+contact: rahilp07@seas.upenn.edu)`.
- 4xx/5xx responses are logged and skipped; the crawler does not retry aggressively.

## Robots.txt compliance

Every crawl starts by fetching `/robots.txt` for the target domain. The standard `urllib.robotparser` is used; any URL the parser rejects for our User-Agent is skipped (counted in INFO logs). If `/robots.txt` is unreachable, the crawler defaults to allow (standard fail-open behavior) and logs a WARNING.

## Layout

```
data/
├── README.md              # this file
├── raw/<domain>/          # per-domain HTML + JSON sidecars from doc_scraper.py
├── interim/
│   ├── queries.csv        # one query per scraped page (query_id, query, source_url)
│   └── serp.csv           # top-10 SERP rows per query (query_id, rank, url, title, snippet)
└── processed/
    └── features.csv       # final per-page feature matrix + is_top_10 target
```

## Reproducibility

- `requirements.txt` is pinned.
- Scrape date and per-page fetch timestamps are recorded in JSON sidecars.
- The full pipeline is `python -m src.scraping.doc_scraper --domain X --limit N` per domain → `python -m src.scraping.serp_client build-queries` → `python -m src.scraping.serp_client fetch` → `python -m src.features.build_features`.
