"""Build the directed link graph across all scraped pages.

Reads ``data/raw/<domain>/*.json`` sidecars, where each sidecar carries the
page's ``outbound_links`` list. Builds an ``nx.DiGraph`` whose nodes are
*scraped* page URLs (matched by host+path key) — outbound links to URLs we
did not scrape are dropped, so the graph is closed under our corpus and
PageRank/HITS converge meaningfully.

The same host+path matching key is used by ``src.features.build_features``
to join SERP labels, so node identity here matches feature-row identity.

CLI:
    python -m src.graph.build_graph --out data/interim/graph.gpickle
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import urllib.parse
from pathlib import Path

import networkx as nx

LOG = logging.getLogger("build_graph")


def url_key(url: str) -> str:
    """Same key as ``src.features.build_features.url_key`` — kept duplicated to
    avoid a cross-package import cycle in this small project."""
    p = urllib.parse.urlparse(url)
    return (p.netloc.lower() + p.path.rstrip("/")).lower()


def build(raw_dir: Path = Path("data/raw")) -> nx.DiGraph:
    """Construct the link graph. Node id = ``url_key(page_url)``. Each node
    carries a ``url`` attribute (the canonical page URL) and a ``domain``
    attribute. Edges are page → page where the destination is also scraped."""
    g: nx.DiGraph = nx.DiGraph()

    # Pass 1 — register every scraped page as a node.
    sidecar_paths: list[Path] = []
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
        g.add_node(key, url=page_url, domain=urllib.parse.urlparse(page_url).netloc)
        sidecar_paths.append(json_path)

    # Pass 2 — add edges only where both endpoints are scraped pages.
    edge_count = 0
    for json_path in sidecar_paths:
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        src = url_key(meta.get("url", ""))
        if src not in g:
            continue
        for link in meta.get("outbound_links", []):
            dst = url_key(link)
            if dst != src and dst in g:
                g.add_edge(src, dst)
                edge_count += 1

    LOG.info("graph built — %d nodes, %d edges", g.number_of_nodes(), edge_count)
    return g


def save(g: nx.DiGraph, out_path: Path) -> None:
    """Persist via pickle (NetworkX's gpickle path is deprecated in newer versions)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)
    LOG.info("graph saved → %s", out_path)


def load(path: Path) -> nx.DiGraph:
    with path.open("rb") as f:
        return pickle.load(f)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the directed link graph from scraped JSON sidecars.")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/interim/graph.pkl"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    g = build(raw_dir=args.raw_dir)
    save(g, args.out)
    print(f"graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges → {args.out}")
    return 0 if g.number_of_nodes() > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
