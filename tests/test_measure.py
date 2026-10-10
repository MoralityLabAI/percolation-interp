from collections import deque

import numpy as np
import pytest

from percinterp import LayeredDAG, bottleneck_weight, components, profile, rank_entry
from percinterp.controls import complete_layered, mlp_graph, init_mlp, permute_within


def random_dag_with_skips(rng, sizes=(3, 4, 4, 3, 2), density=0.5):
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    layer = np.repeat(np.arange(len(sizes)), sizes)
    pairs = [(u, v) for u in range(offsets[-1]) for v in range(offsets[-1]) if layer[u] < layer[v]]
    keep = [pv for pv in pairs if rng.random() < density]
    src, dst = np.array(keep).T
    return LayeredDAG(list(sizes), src, dst, rng.random(len(src)))


def brute_reach(g, kept, start, forward=True):
    adj = {}
    for s, d in zip(g.src[kept], g.dst[kept]):
        a, b = (s, d) if forward else (d, s)
        adj.setdefault(a, []).append(b)
    seen = set(np.flatnonzero(start))
    queue = deque(seen)
    while queue:
        u = queue.popleft()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return seen


@pytest.mark.parametrize("seed", range(20))
def test_profile_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    g = random_dag_with_skips(rng)
    entry = rng.random(g.n_edges)
    prof = profile(g, entry)
    for p in np.concatenate([[0.0], np.sort(entry), [1.0]]):
        kept = entry <= p
        fwd = brute_reach(g, kept, g.sources, forward=True)
        bwd = brute_reach(g, kept, g.sinks, forward=False)
        for v in range(g.n_nodes):
            assert (prof.F[v] <= p) == (v in fwd)
            assert (prof.node_entry[v] <= p) == (v in fwd and v in bwd)
        spans = bool(fwd & set(np.flatnonzero(g.sinks)))
        assert spans == bool(prof.spanning([p])[0])
        bb_edges = [k for k in np.flatnonzero(kept) if g.src[k] in fwd and g.dst[k] in bwd]
        assert np.isclose(prof.backbone_edge_fraction([p])[0], len(bb_edges) / g.n_edges)


def test_bottleneck_is_widest_path():
    rng = np.random.default_rng(0)
    g = random_dag_with_skips(rng, sizes=(2, 3, 3, 2), density=0.6)
    out = {}
    for s, d, w in zip(g.src, g.dst, g.weight):
        out.setdefault(s, []).append((d, w))

    def widest(u):
        if g.sinks[u]:
            return np.inf
        return max((min(w, widest(d)) for d, w in out.get(u, [])), default=-np.inf)

    expected = max(widest(s) for s in np.flatnonzero(g.sources))
    assert np.isclose(bottleneck_weight(g), expected)


def test_rank_entry_keeps_top_fraction_per_group():
    rng = np.random.default_rng(1)
    w = rng.random(1000)
    groups = rng.integers(0, 4, size=1000)
    q = rank_entry(w, groups)
    for gid in range(4):
        m = groups == gid
        top = np.argsort(w[m])[::-1][: int(0.3 * m.sum())]
        kept = np.flatnonzero(q[m] <= 0.3)
        assert set(top) == set(kept)


def test_rank_entry_ties_enter_together():
    q = rank_entry(np.array([1.0, 2.0, 2.0, 3.0]))
    assert list(q) == [1.0, 0.75, 0.75, 0.25]


def test_components_on_known_graph():
    # two disjoint paths 0-2-4 and 1-3-5 plus an isolated node 6 in a 3-layer graph
    g = LayeredDAG([2, 2, 3], [0, 2, 1, 3], [2, 4, 3, 5], [1.0, 1.0, 1.0, 1.0])
    c = components(g, np.array([0.1, 0.1, 0.5, 0.5]), [0.2, 1.0])
    # p=0.2: clusters {0,2,4}, {1}, {3}, {5}, {6}
    assert np.isclose(c["largest"][0], 3 / 7)
    assert np.isclose(c["susceptibility"][0], 1.0)
    # p=1: clusters {0,2,4}, {1,3,5}, {6}; one 3-cluster is the largest, rest = {3, 1}
    assert np.isclose(c["largest"][1], 3 / 7)
    assert np.isclose(c["susceptibility"][1], (9 + 1) / 4)


def test_permute_within_preserves_group_multisets():
    rng = np.random.default_rng(2)
    w = rng.random(500)
    groups = rng.integers(0, 5, size=500)
    s = permute_within(w, groups, rng)
    for gid in range(5):
        assert np.allclose(np.sort(w[groups == gid]), np.sort(s[groups == gid]))
    assert not np.allclose(w, s)


def test_from_dense_layout_and_zero_dropping():
    W0 = np.array([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]])  # 2 -> 3
    g = LayeredDAG.from_dense([W0])
    assert g.layer_sizes == [2, 3]
    edges = {(int(s), int(d)): w for s, d, w in zip(g.src, g.dst, g.weight)}
    assert edges == {(0, 2): 1.0, (0, 3): 2.0, (1, 3): 3.0, (1, 4): 4.0}


def test_mlp_eap_matches_finite_difference_of_edge():
    rng = np.random.default_rng(3)
    widths = [4, 6, 5, 3]
    Ws = init_mlp(widths, rng)
    x = rng.normal(size=4)
    readout = rng.normal(size=3)
    g = mlp_graph(Ws, x, kind="eap", readout=readout)
    # first-order effect of deleting edge (j -> i) in layer 1 equals a_j W_ij g_i
    from percinterp.controls import mlp_forward

    acts, _ = mlp_forward(Ws, x)
    base = readout @ acts[-1]
    eps = 1e-6
    for k in range(g.n_edges):  # every surviving edge, all layers
        s, d = g.src[k], g.dst[k]
        l = g.node_layer[s]
        j, i = s - g.offsets[l], d - g.offsets[l + 1]
        Wp = [W.copy() for W in Ws]
        Wp[l][i, j] *= 1 - eps
        acts_p, _ = mlp_forward(Wp, x)
        fd = (base - readout @ acts_p[-1]) / eps
        assert np.isclose(g.weight[k], abs(fd), rtol=1e-4, atol=1e-9)
    assert g.n_edges > 0
