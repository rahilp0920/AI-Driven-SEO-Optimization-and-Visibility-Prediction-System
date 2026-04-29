"""Demo-only: produce a synthetic data/interim/serp.csv when no SERP API key is
available. Schema-compatible with the real one so build_features.py runs unchanged.

For each query, ~30% chance the source_url ranks 1-3 (positive label), else fills
top-10 with random URLs from the corpus. This gives the model a class-imbalanced
target with structural correlation to the source page — realistic enough to train
on for a smoke-test demo.

For the actual graded submission, replace this with real SERP fetches:
    python -m src.scraping.serp_client fetch
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

POSITIVE_RATE = 0.30  # fraction of queries where the source page is genuinely top-10
SEED = 42


def main() -> int:
    random.seed(SEED)
    queries_path = Path("data/interim/queries.csv")
    serp_path = Path("data/interim/serp.csv")

    if not queries_path.exists():
        print(f"missing {queries_path} — run `python -m src.scraping.serp_client build-queries` first", file=sys.stderr)
        return 1

    queries = list(csv.DictReader(queries_path.open(encoding="utf-8")))
    all_urls = [q["source_url"] for q in queries]
    if len(all_urls) < 10:
        print(f"need ≥10 scraped pages, found {len(all_urls)}", file=sys.stderr)
        return 1

    serp_path.parent.mkdir(parents=True, exist_ok=True)
    n_pos = 0
    with serp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query", "source_url", "rank", "url", "title", "snippet"])
        for q in queries:
            qid, query, src = q["query_id"], q["query"], q["source_url"]
            is_top = random.random() < POSITIVE_RATE
            ranking = random.sample(
                [u for u in all_urls if u != src],
                k=min(9, len(all_urls) - 1),
            )
            if is_top:
                ranking.insert(random.randint(0, 2), src)
                n_pos += 1
            for rank, url in enumerate(ranking[:10], start=1):
                writer.writerow([qid, query, src, rank, url, "synthetic", "synthetic snippet"])

    print(f"wrote synthetic serp.csv — {n_pos}/{len(queries)} queries label source as top-10", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
