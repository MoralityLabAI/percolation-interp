"""Diagnostics for the sieve's z-curve: why the backbone is larger or smaller than its null.

Exploratory tools (e06). None of them changes `sieve()`; they reuse its ranking
(`degree_entry`) and its shuffles (`permute_within`).

1. Coverage/alignment decomposition. With A_v(c) = 1[F(v) <= c] (reached from the
   sources) and D_v(c) = 1[B(v) <= c] (reaches the sink), the backbone fraction of a
   layer is, exactly,

       b_l(c) = mean_l(A D) = r^F_l(c) r^B_l(c) + Cov_l(A, D)

   with r^F = mean_l(A), r^B = mean_l(D) and the population covariance over the layer's
   nodes. r^F r^B is what independent placement would give ("coverage"); Cov is how much
   the nodes reached from the sources are the same nodes that reach the sink
   ("alignment"). The interior backbone fraction is the node-weighted sum over layers.
2. Node strength: in- and out-strength per node and their rank correlation by layer.
3. Kept-degree coverage: share of each layer with at least one kept in-edge (out-edge)
   at c, against N[1 - (1 - 1/N)^m] for m kept edges landing uniformly.
4. Backbone anatomy at c: nodes per layer, edge layer spans, and the hop-length
   distribution of source-to-sink paths inside the backbone.
5. Joint-degree null at c: binarise the kept edges and rewire them by double-edge swaps
   within (source layer, destination layer) blocks, keeping every node's kept in- and
   out-degree per block. Optionally only onto edges of the original skeleton.
6. Group labels for weight-level nulls: layer pairs, and node-score bins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controls import permute_within
from .graph import LayeredDAG
from .measure import backward_entry, forward_entry, profile
from .sieve import degree_entry


# --- 1. coverage / alignment ---------------------------------------------------------------


def default_grid(entry: np.ndarray, n: int = 48) -> np.ndarray:
    """The sieve's default c grid for an entry array (degree units)."""
    finite = entry[np.isfinite(entry)]
    return np.geomspace(max(finite.min(), 1e-3), finite.max(), n)


@dataclass
class Decomposition:
    """Per-layer terms on a c grid. Arrays are (n_layers_used, len(cs))."""

    layers: np.ndarray  # graph layer index of each row
    n: np.ndarray  # nodes counted per layer
    cs: np.ndarray
    rF: np.ndarray
    rB: np.ndarray
    cov: np.ndarray
    b: np.ndarray

    @property
    def weights(self) -> np.ndarray:
        return self.n / self.n.sum()

    @property
    def coverage(self) -> np.ndarray:
        return self.rF * self.rB

    def total(self, term: str = "b") -> np.ndarray:
        """Node-weighted sum over layers of b, coverage or cov: one curve over c."""
        x = {"b": self.b, "coverage": self.coverage, "cov": self.cov}[term]
        return (self.weights[:, None] * x).sum(0)


def decompose_entries(g: LayeredDAG, F: np.ndarray, B: np.ndarray, cs: np.ndarray,
                      nodes: np.ndarray | None = None) -> Decomposition:
    """b_l = r^F r^B + Cov per layer, from F and B (one graph)."""
    if nodes is None:
        nodes = ~(g.sources | g.sinks)
    cs = np.asarray(cs, dtype=float)
    layers = [l for l in range(g.n_layers) if nodes[g.layer_nodes(l)].any()]
    rows = {k: [] for k in ("rF", "rB", "cov", "b")}
    ns = []
    for l in layers:
        sl = g.layer_nodes(l)
        m = nodes[sl]
        A = (F[sl][m][None, :] <= cs[:, None]).astype(float)
        D = (B[sl][m][None, :] <= cs[:, None]).astype(float)
        rF, rB, b = A.mean(1), D.mean(1), (A * D).mean(1)
        rows["rF"].append(rF)
        rows["rB"].append(rB)
        rows["b"].append(b)
        rows["cov"].append(((A - rF[:, None]) * (D - rB[:, None])).mean(1))
        ns.append(int(m.sum()))
    return Decomposition(np.array(layers), np.array(ns), cs, *(np.array(rows[k]) for k in ("rF", "rB", "cov", "b")))


@dataclass
class DecompositionNull:
    real: Decomposition
    shuffles: list[Decomposition]

    def stack(self, term: str) -> np.ndarray:
        """(n_shuffles, n_layers_used, len(cs)) for rF, rB, cov, b or coverage."""
        return np.array([getattr(d, term) for d in self.shuffles])

    def excess(self, term: str, total: bool = True) -> np.ndarray:
        """Real minus mean shuffle, per layer or (total=True) node-weighted over layers.

        excess('b') = excess('coverage') + excess('cov') exactly, per layer and in total."""
        if total:
            return self.real.total(term) - np.mean([d.total(term) for d in self.shuffles], axis=0)
        return getattr(self.real, term) - self.stack(term).mean(0)


def decompose(g: LayeredDAG, weight: np.ndarray, groups: np.ndarray | None = None, cs: np.ndarray | None = None,
              n_shuffles: int = 100, rng: np.random.Generator | None = None, nodes: np.ndarray | None = None,
              null_groups: np.ndarray | None = None) -> DecompositionNull:
    """Coverage/alignment decomposition of a graph and of its shuffles.

    Uses sieve()'s ranking, grid and shuffle procedure: with the same arguments and rng
    seed, real.total('b') equals sieve().real and the shuffles' totals equal its shuffles."""
    rng = rng or np.random.default_rng(0)
    groups = g.edge_layer("dst") if groups is None else groups
    weight = np.asarray(weight, dtype=float)
    E = degree_entry(g, weight, groups)
    cs = default_grid(E) if cs is None else np.asarray(cs, dtype=float)
    p = profile(g, E)
    real = decompose_entries(g, p.F, p.B, cs, nodes)
    live = weight > 0
    null_groups = groups if null_groups is None else np.asarray(null_groups)
    out = []
    for _ in range(n_shuffles):
        w = np.zeros_like(weight)
        w[live] = permute_within(weight[live], null_groups[live], rng)
        ps = profile(g, degree_entry(g, w, groups))
        out.append(decompose_entries(g, ps.F, ps.B, cs, nodes))
    return DecompositionNull(real, out)


# --- 2. node strength ----------------------------------------------------------------------


def node_strength(g: LayeredDAG, weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(in-strength, out-strength) per node: summed incoming and outgoing edge weight."""
    w = g.weight if weight is None else np.asarray(weight, dtype=float)
    s_in = np.bincount(g.dst, weights=w, minlength=g.n_nodes)
    s_out = np.bincount(g.src, weights=w, minlength=g.n_nodes)
    return s_in, s_out


def _rank(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(x)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def strength_correlation(g: LayeredDAG, weight: np.ndarray | None = None, nodes: np.ndarray | None = None,
                         node_groups: np.ndarray | None = None) -> dict:
    """Spearman rank correlation of in- and out-strength, per node group (default: layer).

    Only nodes with both strengths > 0 among `nodes` (default: interior) are used.
    Also returns the pooled correlation of within-group ranks (each group's ranks scaled to
    (0, 1)), and per group the share of total out- and in-strength held by its top 1% and
    top 10% of nodes (hub concentration)."""
    s_in, s_out = node_strength(g, weight)
    if nodes is None:
        nodes = ~(g.sources | g.sinks)
    node_groups = g.node_layer if node_groups is None else np.asarray(node_groups)
    ok = nodes & (s_in > 0) & (s_out > 0)
    per, pooled_x, pooled_y = {}, [], []
    for gid in np.unique(node_groups[ok]):
        m = ok & (node_groups == gid)
        n = int(m.sum())
        rec = {"n": n, "rho": spearman(s_in[m], s_out[m])}
        for name, s in (("out", s_out[m]), ("in", s_in[m])):
            srt = np.sort(s)[::-1]
            rec[f"top1_{name}"] = float(srt[:max(1, int(np.ceil(0.01 * n)))].sum() / srt.sum())
            rec[f"top10_{name}"] = float(srt[:max(1, int(np.ceil(0.10 * n)))].sum() / srt.sum())
        per[gid.item() if hasattr(gid, "item") else gid] = rec
        if n >= 3:
            pooled_x.append((_rank(s_in[m]) - 0.5) / n)
            pooled_y.append((_rank(s_out[m]) - 0.5) / n)
    pooled = (float(np.corrcoef(np.concatenate(pooled_x), np.concatenate(pooled_y))[0, 1])
              if pooled_x else float("nan"))
    return {"per_group": per, "pooled_within_group_rho": pooled, "n": int(ok.sum())}


# --- 3. kept-degree coverage -----------------------------------------------------------------


def uniform_coverage(m: float, N: int) -> float:
    """Expected share of N slots hit at least once by m uniform draws: 1 - (1 - 1/N)^m."""
    return 1.0 - (1.0 - 1.0 / N) ** m if N > 0 else float("nan")


def kept_coverage(g: LayeredDAG, entry: np.ndarray, c: float) -> dict:
    """Per layer: share of nodes with >= 1 kept in-edge (out-edge) at c, the kept edge counts
    into (out of) the layer, and the uniform-placement expectation for those counts."""
    keep = entry <= c
    has_in = np.zeros(g.n_nodes, dtype=bool)
    has_out = np.zeros(g.n_nodes, dtype=bool)
    has_in[g.dst[keep]] = True
    has_out[g.src[keep]] = True
    m_in = np.bincount(g.node_layer[g.dst[keep]], minlength=g.n_layers)
    m_out = np.bincount(g.node_layer[g.src[keep]], minlength=g.n_layers)
    rows = []
    for l in range(g.n_layers):
        sl = g.layer_nodes(l)
        N = g.layer_sizes[l]
        rows.append({"layer": l, "N": N, "m_in": int(m_in[l]), "m_out": int(m_out[l]),
                     "in_cov": float(has_in[sl].mean()), "out_cov": float(has_out[sl].mean()),
                     "in_cov_uniform": uniform_coverage(m_in[l], N),
                     "out_cov_uniform": uniform_coverage(m_out[l], N)})
    return {"c": float(c), "layers": rows}


# --- 4. backbone anatomy ---------------------------------------------------------------------


def backbone_anatomy(g: LayeredDAG, entry: np.ndarray, c: float) -> dict:
    """Structure of the backbone at c: nodes per layer, kept backbone edges by layer span,
    and source-to-sink path counts by hop length within the backbone subgraph."""
    p = profile(g, entry)
    on = p.node_entry <= c
    e_on = p.edge_backbone_entry <= c
    span = g.node_layer[g.dst[e_on]] - g.node_layer[g.src[e_on]]
    pair = {}
    for s, d in zip(g.node_layer[g.src[e_on]], g.node_layer[g.dst[e_on]]):
        pair[(int(s), int(d))] = pair.get((int(s), int(d)), 0) + 1
    # path counts by hop length: P[v, k] = # backbone paths from a source to v with k edges
    L = g.n_layers
    P = np.zeros((g.n_nodes, L))
    P[g.sources & on, 0] = 1.0
    src, dst = g.src[e_on], g.dst[e_on]
    order = np.argsort(g.node_layer[dst], kind="stable")
    src, dst = src[order], dst[order]
    dl = g.node_layer[dst]
    for layer in range(1, L):
        m = dl == layer
        if m.any():
            np.add.at(P[:, 1:], dst[m], P[src[m], :-1])
    paths = P[g.sinks].sum(0)
    hops = np.flatnonzero(paths > 0)
    return {
        "c": float(c),
        "n_nodes": int(on.sum()),
        "nodes_per_layer": [int(on[g.layer_nodes(l)].sum()) for l in range(L)],
        "n_edges": int(e_on.sum()),
        "edge_span_counts": {int(k): int(v) for k, v in zip(*np.unique(span, return_counts=True))},
        "edge_layer_pairs": {f"{s}->{d}": v for (s, d), v in sorted(pair.items())},
        "path_counts_by_hops": {int(k): float(paths[k]) for k in hops},
        "min_hops": int(hops.min()) if len(hops) else None,
        "max_hops": int(hops.max()) if len(hops) else None,
    }


# --- 5. joint-degree null --------------------------------------------------------------------


def binary_backbone_fraction(g: LayeredDAG, src: np.ndarray, dst: np.ndarray,
                             nodes: np.ndarray | None = None) -> float:
    """Backbone fraction of `nodes` in the unweighted graph made of the given edges only."""
    if nodes is None:
        nodes = ~(g.sources | g.sinks)
    h = LayeredDAG(g.layer_sizes, src, dst, np.ones(len(src)), sources=g.sources, sinks=g.sinks)
    z = np.zeros(len(src))
    ne = np.maximum(forward_entry(h, z), backward_entry(h, z))
    return float((ne[nodes] <= 0).mean())


def double_edge_swaps(g: LayeredDAG, src: np.ndarray, dst: np.ndarray, rng: np.random.Generator,
                      n_rounds: int = 30, allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Rewire a simple directed edge set by double-edge swaps within (src layer, dst layer) blocks.

    A swap takes a->b, c->d from the same block and makes a->d, c->b. Every node keeps its
    in- and out-degree within each block; no multi-edges are created. Each round pairs the
    block's edges at random and proposes all pairs' swaps at once; a proposal is rejected
    if a new edge already exists, is proposed twice, or (if `allowed` keys are given) is
    not in `allowed` (edge keys src * n_nodes + dst). Returns (src, dst, acceptance rate).
    """
    src, dst = np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)
    N = g.n_nodes
    block = g.node_layer[src] * g.n_layers + g.node_layer[dst]
    allowed = None if allowed is None else np.unique(np.asarray(allowed, dtype=np.int64))
    tried = accepted = 0
    for _ in range(n_rounds):
        perm = rng.permutation(len(src))
        perm = perm[np.argsort(block[perm], kind="stable")]  # random order within each block
        bl = block[perm]
        # pair consecutive edges within a block: positions 0-1, 2-3, ... of each block run
        starts = np.flatnonzero(np.r_[True, bl[1:] != bl[:-1]])
        pos = np.arange(len(perm)) - np.repeat(starts, np.diff(np.r_[starts, len(perm)]))
        first = np.flatnonzero((pos % 2 == 0) & (np.r_[bl[1:] == bl[:-1], False]))
        i, j = perm[first], perm[first + 1]
        ok = (src[i] != src[j]) & (dst[i] != dst[j])
        k1 = src[i] * N + dst[j]
        k2 = src[j] * N + dst[i]
        keys = src * N + dst
        ok &= ~np.isin(k1, keys) & ~np.isin(k2, keys)
        if allowed is not None:
            ok &= np.isin(k1, allowed) & np.isin(k2, allowed)
        new = np.concatenate([k1[ok], k2[ok]])
        u, cnt = np.unique(new, return_counts=True)
        dup = u[cnt > 1]
        if len(dup):
            ok &= ~(np.isin(k1, dup) | np.isin(k2, dup))
        tried += len(i)
        accepted += int(ok.sum())
        di, dj = dst[i[ok]].copy(), dst[j[ok]].copy()
        dst[i[ok]], dst[j[ok]] = dj, di
    return src, dst, accepted / max(1, tried)


@dataclass
class JointDegreeNull:
    c: float
    real: float  # backbone fraction of the kept graph at c
    null: np.ndarray  # backbone fraction of each rewired graph
    n_edges: int
    acceptance: float

    @property
    def z(self) -> float:
        sd = self.null.std()
        return float((self.real - self.null.mean()) / sd) if sd > 0 else float("nan")

    @property
    def p_high(self) -> float:
        return float((1 + (self.null >= self.real).sum()) / (1 + len(self.null)))

    @property
    def p_low(self) -> float:
        return float((1 + (self.null <= self.real).sum()) / (1 + len(self.null)))


def joint_degree_null(g: LayeredDAG, entry: np.ndarray, c: float, n_null: int = 100,
                      rng: np.random.Generator | None = None, n_rounds: int = 30, skeleton: bool = True,
                      nodes: np.ndarray | None = None) -> JointDegreeNull:
    """Backbone fraction at c against kept graphs rewired with every node's kept degrees fixed.

    entry: the sieve's degree-unit keys (degree_entry). skeleton=True only allows swaps onto
    edges that exist (finite entry) in the original graph, like the weight shuffles."""
    rng = rng or np.random.default_rng(0)
    keep = entry <= c
    s, d = g.src[keep], g.dst[keep]
    real = binary_backbone_fraction(g, s, d, nodes)
    allowed = (g.src[np.isfinite(entry)] * g.n_nodes + g.dst[np.isfinite(entry)]) if skeleton else None
    vals, acc = [], []
    for _ in range(n_null):
        s2, d2, a = double_edge_swaps(g, s, d, rng, n_rounds=n_rounds, allowed=allowed)
        vals.append(binary_backbone_fraction(g, s2, d2, nodes))
        acc.append(a)
    return JointDegreeNull(float(c), real, np.array(vals), int(keep.sum()), float(np.mean(acc)))


# --- 6. group labels for weight-level nulls -------------------------------------------------


def pair_groups(g: LayeredDAG) -> np.ndarray:
    """One null group per (source layer, destination layer): keeps the layer-pair weight mix."""
    return g.edge_layer("pair")


def binned_groups(g: LayeredDAG, base: np.ndarray, src_score: np.ndarray | None = None,
                  dst_score: np.ndarray | None = None, n_bins: int = 5) -> np.ndarray:
    """Refine edge groups `base` by quantile bins of a node score of the source and/or target,
    with bins computed within each node layer. E.g. firing rate gives an activity-conditioned
    null: weights move only between edges whose endpoints have similar activity."""
    def node_bins(score):
        score = np.asarray(score, dtype=float)
        out = np.zeros(g.n_nodes, dtype=np.int64)
        for l in range(g.n_layers):
            sl = g.layer_nodes(l)
            x = score[sl]
            if len(x) > 1:
                r = _rank(x)
                out[sl] = np.minimum((r - 1) * n_bins // len(x), n_bins - 1).astype(np.int64)
        return out

    lab = np.asarray(base, dtype=np.int64) * n_bins * n_bins
    if src_score is not None:
        lab = lab + node_bins(src_score)[g.src] * n_bins
    if dst_score is not None:
        lab = lab + node_bins(dst_score)[g.dst]
    return lab
