"""The percolation sieve: which nodes of an attribution graph are connected structure, not noise.

For a weighted LayeredDAG (edges from earlier to later layers, skip edges allowed):
  1. rank edges within each group (default: destination layer), in degree units:
     c = kept edges per live source node of that group;
  2. compute the backbone node fraction (nodes on a surviving source-to-sink path)
     across a grid of c, for the graph and for within-group shuffles of its weights;
  3. learned connectivity = graph minus the mean shuffle, and its z-score against the
     shuffles' spread at each c (floored at one node, 1/n);
  4. significance: the graph's peak z against each shuffle's own peak z measured against
     the remaining shuffles (raw differences are dominated by fluctuations near the
     percolation transition, where shuffles vary most);
  5. c* = the c of peak raw learned connectivity among points with z >= z_min
     (default 3); the sieved structure is the backbone at c*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controls import permute_within
from .graph import LayeredDAG
from .measure import profile, rank_entry


def degree_entry(g: LayeredDAG, weight: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-group rank keys in degree units: kept at c when key <= c.

    Zero-weight edges are not edges (+inf). Within a group with E edges and S distinct
    source nodes, keeping the top fraction p keeps p*E edges, i.e. c = p*E/S per source.
    """
    weight = np.asarray(weight, dtype=float)
    out = np.full(len(weight), np.inf)
    live = weight > 0
    q = rank_entry(weight[live], groups[live])
    scale = np.empty(live.sum())
    gl = groups[live]
    src = g.src[live]
    for gid in np.unique(gl):
        m = gl == gid
        scale[m] = m.sum() / len(np.unique(src[m]))
    out[live] = q * scale
    return out


@dataclass
class SieveResult:
    cs: np.ndarray
    real: np.ndarray  # backbone node fraction of the graph, per c
    shuffles: np.ndarray  # (n_shuffles, len(cs))
    learned: np.ndarray  # real - mean(shuffles)
    z: np.ndarray  # learned / sd(shuffles), sd floored at 1/n
    c_star: float
    peak: float  # raw learned connectivity at c*
    z_peak: float
    null_peaks: np.ndarray  # each shuffle's peak z against the other shuffles
    backbone: np.ndarray  # boolean node mask: on the backbone at c*
    node_entry: np.ndarray  # c at which each node joins the backbone (graph)
    null_node_entry: np.ndarray | None = None  # (n_shuffles, n_nodes), if keep_null_entries

    @property
    def p_value(self) -> float:
        """Fraction of shuffles whose own peak z is at least the graph's (with the +1 correction)."""
        return (1 + (self.null_peaks >= self.z_peak).sum()) / (1 + len(self.null_peaks))


def _z(x: np.ndarray, ref: np.ndarray, floor: float) -> np.ndarray:
    return (x - ref.mean(0)) / np.maximum(ref.std(0), floor)


def sieve(g: LayeredDAG, weight: np.ndarray, groups: np.ndarray | None = None, cs: np.ndarray | None = None,
          n_shuffles: int = 100, rng: np.random.Generator | None = None,
          nodes: np.ndarray | None = None, z_min: float = 3.0,
          null_groups: np.ndarray | None = None, keep_null_entries: bool = False) -> SieveResult:
    """Run the sieve. `nodes` restricts the backbone fraction to a node subset (default: all
    nodes that are neither sources nor sinks).

    null_groups: labels within which the null permutes weights (default: the ranking groups,
    i.e. within-layer shuffles). Pass g.src to keep every source node's out-strength
    distribution (permute its out-edges among its targets), or g.dst for in-strength.
    Ranking always uses `groups`.

    keep_null_entries: also return each shuffle's node entry c (null_node_entry), e.g. to take
    the backbones of shuffled graphs as a null of connected sets."""
    rng = rng or np.random.default_rng(0)
    groups = g.edge_layer("dst") if groups is None else groups
    if nodes is None:
        nodes = ~(g.sources | g.sinks)
    E = degree_entry(g, weight, groups)
    finite = E[np.isfinite(E)]
    cs = np.geomspace(max(finite.min(), 1e-3), finite.max(), 48) if cs is None else cs
    prof = profile(g, E)
    ne = prof.node_entry[nodes]
    real = (ne[None, :] <= cs[:, None]).mean(1)
    shuffles, null_entries = [], []
    live = np.asarray(weight) > 0
    null_groups = groups if null_groups is None else np.asarray(null_groups)
    for _ in range(n_shuffles):
        w = np.zeros_like(np.asarray(weight, dtype=float))
        w[live] = permute_within(np.asarray(weight, dtype=float)[live], null_groups[live], rng)
        ps = profile(g, degree_entry(g, w, groups))
        shuffles.append((ps.node_entry[nodes][None, :] <= cs[:, None]).mean(1))
        if keep_null_entries:
            null_entries.append(ps.node_entry.astype(np.float32))
    shuffles = np.array(shuffles)
    floor = 1.0 / max(1, nodes.sum())
    learned = real - shuffles.mean(0)
    z = _z(real, shuffles, floor)
    null_peaks = np.array([
        _z(shuffles[i], np.delete(shuffles, i, axis=0), floor).max() for i in range(n_shuffles)
    ]) if n_shuffles > 2 else np.array([])
    ok = z >= z_min
    k = int(np.argmax(np.where(ok, learned, -np.inf))) if ok.any() else int(np.argmax(learned))
    return SieveResult(cs=cs, real=real, shuffles=shuffles, learned=learned, z=z, c_star=float(cs[k]),
                       peak=float(learned[k]), z_peak=float(z.max()), null_peaks=null_peaks,
                       backbone=prof.node_entry <= cs[k], node_entry=prof.node_entry,
                       null_node_entry=np.array(null_entries) if keep_null_entries else None)
