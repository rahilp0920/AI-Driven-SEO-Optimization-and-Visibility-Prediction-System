"""Smoke tests for src/graph/.

We build a tiny DiGraph by hand (skipping the disk walk) and verify
``compute()`` returns the expected six columns with sensible values.
"""

from __future__ import annotations

import networkx as nx

from src.graph.build_graph import url_key
from src.graph.graph_features import compute


def _toy_graph() -> nx.DiGraph:
    g: nx.DiGraph = nx.DiGraph()
    nodes = [
        ("docs.python.org/a", "https://docs.python.org/a", "docs.python.org"),
        ("docs.python.org/b", "https://docs.python.org/b", "docs.python.org"),
        ("docs.python.org/c", "https://docs.python.org/c", "docs.python.org"),
        ("docs.python.org/d", "https://docs.python.org/d", "docs.python.org"),
    ]
    for node_id, url, domain in nodes:
        g.add_node(node_id, url=url, domain=domain)
    g.add_edges_from([
        ("docs.python.org/a", "docs.python.org/b"),
        ("docs.python.org/a", "docs.python.org/c"),
        ("docs.python.org/b", "docs.python.org/c"),
        ("docs.python.org/c", "docs.python.org/d"),
        ("docs.python.org/d", "docs.python.org/a"),
    ])
    return g


def test_url_key_normalization():
    assert url_key("HTTPS://Example.com/Path/") == "example.com/path"
    assert url_key("https://example.com/p?q=1") == "example.com/p"


def test_compute_returns_six_features_per_node():
    g = _toy_graph()
    df = compute(g)
    assert len(df) == 4
    assert set(df.columns) == {"node", "pagerank", "hits_hub", "hits_authority", "in_degree", "out_degree", "clustering"}


def test_pagerank_sums_to_about_one():
    g = _toy_graph()
    df = compute(g)
    assert abs(df["pagerank"].sum() - 1.0) < 1e-3


def test_in_out_degrees_match_graph():
    g = _toy_graph()
    df = compute(g).set_index("node")
    # node a → out 2 (b,c); in 1 (d)
    assert df.loc["docs.python.org/a", "out_degree"] == 2.0
    assert df.loc["docs.python.org/a", "in_degree"] == 1.0
    # node c → in 2 (a,b); out 1 (d)
    assert df.loc["docs.python.org/c", "in_degree"] == 2.0
    assert df.loc["docs.python.org/c", "out_degree"] == 1.0


def test_compute_handles_empty_graph():
    g: nx.DiGraph = nx.DiGraph()
    df = compute(g)
    assert df.empty
    assert "pagerank" in df.columns
