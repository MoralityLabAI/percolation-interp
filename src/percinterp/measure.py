"""Percolation measurements on a LayeredDAG.

Everything works in terms of an *entry key* per edge: the edge is present at
occupation p when entry <= p. Ranking weights within a layer (rank_entry with
groups) gives "keep the top fraction p of each layer"; ranking globally gives
"keep the top fraction p overall", which on one graph is the same sweep as an
absolute weight threshold. i.i.d. uniform keys give Bernoulli bond percolation.

Two dynamic-programming passes give the whole sweep at once, with no
per-threshold work:

    F(v) = min over input-to-v paths of the largest entry on the path
    B(v) = min over v-to-output paths of the largest entry on the path

Node v lies on a surviving input-to-output path at p exactly when
max(F(v), B(v)) <= p, and the graph spans exactly when p >= min over outputs
of F. In weight terms F is the widest-path (bottleneck) problem, solved here
in O(E) by one pass in topological order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .graph import LayeredDAG


def rank_entry(weight: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Entry key = fraction of edges in the same group whose weight is >= this one.

    Keeping edges with key <= p keeps the top fraction p of each group. Ties
    enter together. Pass magnitudes if sign should not matter.
    """
    weight = np.asarray(weight, dtype=np.float64)
    if groups is None:
        groups = np.zeros(len(weight), dtype=np.int64)
    out = np.empty(len(weight))
    for gid in np.unique(groups):
        idx = np.flatnonzero(groups == gid)
        w = weight[idx]
        asc = np.sort(w)
        n_ge = len(w) - np.searchsorted(asc, w, side="left")
        out[idx] = n_ge / len(w)
    return out


def _edge_blocks(g: LayeredDAG, by: str):
    """Edge permutation sorted by (dst or src) layer, plus per-layer slices. Cached on g."""
    key = ("blocks", by)
    cache = g.__dict__.setdefault("_cache", {})
    if key not in cache:
        layer = g.edge_layer(by)
        perm = np.argsort(layer, kind="stable")
        sorted_layer = layer[perm]
        bounds = np.searchsorted(sorted_layer, np.arange(g.n_layers + 1), side="left")
        cache[key] = (perm, bounds)
    return cache[key]


def forward_entry(g: LayeredDAG, entry: np.ndarray) -> np.ndarray:
    """F(v): smallest p at which v is reachable from some source. +inf if never."""
    F = np.full(g.n_nodes, np.inf)
    F[g.sources] = -np.inf
    perm, bounds = _edge_blocks(g, "dst")
    for layer in range(1, g.n_layers):
        e = perm[bounds[layer]:bounds[layer + 1]]
        if len(e):
            np.minimum.at(F, g.dst[e], np.maximum(F[g.src[e]], entry[e]))
    return F


def backward_entry(g: LayeredDAG, entry: np.ndarray) -> np.ndarray:
    """B(v): smallest p at which some sink is reachable from v. +inf if never."""
    B = np.full(g.n_nodes, np.inf)
    B[g.sinks] = -np.inf
    perm, bounds = _edge_blocks(g, "src")
    for layer in range(g.n_layers - 2, -1, -1):
        e = perm[bounds[layer]:bounds[layer + 1]]
        if len(e):
            np.minimum.at(B, g.src[e], np.maximum(B[g.dst[e]], entry[e]))
    return B


def bottleneck_weight(g: LayeredDAG, weight: np.ndarray | None = None) -> float:
    """Largest w such that a source-to-sink path survives keeping edges with weight >= w."""
    weight = g.weight if weight is None else weight
    return float(-forward_entry(g, -np.asarray(weight))[g.sinks].min())


def _frac_le(values: np.ndarray, ps: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(len(ps))
    return np.searchsorted(np.sort(values), ps, side="right") / len(values)


@dataclass
class Profile:
    """Entry values for one graph; evaluate any curve on any p grid without recomputing."""

    g: LayeredDAG
    entry: np.ndarray
    F: np.ndarray
    B: np.ndarray

    @property
    def node_entry(self) -> np.ndarray:
        """p at which the node joins the backbone (lies on a surviving input-output path)."""
        return np.maximum(self.F, self.B)

    @property
    def edge_backbone_entry(self) -> np.ndarray:
        return np.maximum(np.maximum(self.F[self.g.src], self.entry), self.B[self.g.dst])

    @property
    def p_span(self) -> float:
        """Smallest p at which some input-to-output path survives."""
        return float(self.F[self.g.sinks].min())

    def spanning(self, ps) -> np.ndarray:
        return (np.asarray(ps) >= self.p_span).astype(float)

    def kept_edge_fraction(self, ps) -> np.ndarray:
        return _frac_le(self.entry, np.asarray(ps))

    def backbone_node_fraction(self, ps, layers: list[int] | None = None) -> np.ndarray:
        """Fraction of nodes (optionally only in the given layers) on a surviving path."""
        ne = self.node_entry
        if layers is not None:
            ne = np.concatenate([ne[self.g.layer_nodes(l)] for l in layers])
        return _frac_le(ne, np.asarray(ps))

    def backbone_edge_fraction(self, ps) -> np.ndarray:
        """Fraction of all edges that are kept and lie on a surviving path."""
        return _frac_le(self.edge_backbone_entry, np.asarray(ps))

    def reach_fraction(self, ps, layer: int) -> np.ndarray:
        """Fraction of nodes in a layer reachable from the sources."""
        return _frac_le(self.F[self.g.layer_nodes(layer)], np.asarray(ps))

    def mean_kept_out_degree(self, p: float) -> float:
        """Kept edges per non-sink node at occupation p (the branching ratio c)."""
        return float((self.entry <= p).sum() / max(1, (~self.g.sinks).sum()))


def profile(g: LayeredDAG, entry: np.ndarray) -> Profile:
    entry = np.asarray(entry, dtype=np.float64)
    return Profile(g, entry, forward_entry(g, entry), backward_entry(g, entry))


def components(g: LayeredDAG, entry: np.ndarray, ps) -> dict[str, np.ndarray]:
    """Undirected cluster statistics at each p.

    largest: fraction of all nodes in the largest weakly connected cluster.
    susceptibility: mean size of the cluster containing a random node outside
    the largest one (sum of s^2 / sum of s over the other clusters).
    """
    ps = np.asarray(ps, dtype=float)
    largest = np.zeros(len(ps))
    chi = np.zeros(len(ps))
    for k, p in enumerate(ps):
        keep = entry <= p
        adj = coo_matrix(
            (np.ones(keep.sum()), (g.src[keep], g.dst[keep])), shape=(g.n_nodes, g.n_nodes)
        )
        _, labels = connected_components(adj, directed=False)
        sizes = np.sort(np.bincount(labels))[::-1]
        largest[k] = sizes[0] / g.n_nodes
        rest = sizes[1:].astype(float)
        chi[k] = (rest**2).sum() / rest.sum() if rest.sum() else 0.0
    return {"largest": largest, "susceptibility": chi}
