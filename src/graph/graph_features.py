"""Compute graph-derived features (PageRank, HITS hub/authority, degrees,
clustering) and merge them into ``data/processed/features.csv`` keyed on the
host+path URL key shared with ``src.features.build_features``.

CLI:
    python -m src.graph.graph_features
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

from src.features.build_features import url_key
from src.graph.build_graph import build as build_graph_from_disk
from src.graph.build_graph import load as load_graph

LOG = logging.getLogger("graph_features")


def compute(g: nx.DiGraph) -> pd.DataFrame:
    """Compute per-node PageRank, HITS hub/authority, in-/out-degree, and
    clustering coefficient. Returns a DataFrame keyed on the ``url_key`` node id."""
    if g.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=["node", "pagerank", "hits_hub", "hits_authority", "in_degree", "out_degree", "clustering"]
        )

    pagerank = nx.pagerank(g, alpha=0.85, max_iter=100, tol=1e-6)

    try:
        hubs, authorities = nx.hits(g, max_iter=200, tol=1e-6, normalized=True)
    except (nx.PowerIterationFailedConvergence, ZeroDivisionError) as exc:
        LOG.warning("HITS did not converge (%s) — defaulting to 0.0", exc)
        hubs = {n: 0.0 for n in g.nodes}
        authorities = {n: 0.0 for n in g.nodes}

    in_deg = dict(g.in_degree())
    out_deg = dict(g.out_degree())
    # Clustering on the underlying undirected graph for stability/interpretability.
    clustering = nx.clustering(g.to_undirected())

    rows = []
    for node in g.nodes:
        rows.append({
            "node": node,
            "pagerank": float(pagerank.get(node, 0.0)),
            "hits_hub": float(hubs.get(node, 0.0)),
            "hits_authority": float(authorities.get(node, 0.0)),
            "in_degree": float(in_deg.get(node, 0)),
            "out_degree": float(out_deg.get(node, 0)),
            "clustering": float(clustering.get(node, 0.0)),
        })
    return pd.DataFrame(rows)


def merge_into_features(
    features_csv: Path = Path("data/processed/features.csv"),
    graph_path: Path | None = Path("data/interim/graph.pkl"),
    raw_dir: Path = Path("data/raw"),
    out_csv: Path | None = None,
) -> int:
    """Compute graph features and left-join them onto the existing features
    matrix on ``url_key(url)``. Pages absent from the graph (shouldn't happen
    if scraper output and feature pipeline are aligned) get 0.0 across all
    graph columns.

    Args:
        features_csv: existing per-page feature matrix.
        graph_path: pickled graph from ``src.graph.build_graph``. If missing,
            the graph is rebuilt from ``raw_dir`` on the fly.
        raw_dir: fallback source for graph construction.
        out_csv: write target. Defaults to ``features_csv`` (in-place merge).

    Returns:
        Number of feature rows in the merged file.
    """
    if not features_csv.exists():
        raise SystemExit(f"missing {features_csv} — run `python -m src.features.build_features` first")

    if graph_path and graph_path.exists():
        g = load_graph(graph_path)
        LOG.info("loaded graph from %s", graph_path)
    else:
        LOG.info("graph pickle missing — building from %s", raw_dir)
        g = build_graph_from_disk(raw_dir=raw_dir)

    gf_df = compute(g)
    LOG.info("graph features rows=%d cols=%d", len(gf_df), gf_df.shape[1])

    feats_df = pd.read_csv(features_csv)
    feats_df["_node"] = feats_df["url"].astype(str).map(url_key)
    merged = feats_df.merge(gf_df, how="left", left_on="_node", right_on="node")
    merged = merged.drop(columns=["_node", "node"])
    for col in ["pagerank", "hits_hub", "hits_authority", "in_degree", "out_degree", "clustering"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    target = out_csv or features_csv
    merged.to_csv(target, index=False)
    LOG.info("wrote %d rows × %d cols → %s", len(merged), merged.shape[1], target)
    return len(merged)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute graph features and merge into features.csv.")
    p.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    p.add_argument("--graph", type=Path, default=Path("data/interim/graph.pkl"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=None, help="Output CSV (default: in-place).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    n = merge_into_features(
        features_csv=args.features,
        graph_path=args.graph,
        raw_dir=args.raw_dir,
        out_csv=args.out,
    )
    print(f"merged graph features into {args.out or args.features} ({n} rows)")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
