import itertools

import numpy as np
from scipy.stats import chisquare

from percinterp import LayeredDAG
from percinterp.diagnostics import (binned_groups, decompose, decompose_fast, joint_degree_null,
                                    joint_degree_null_sequential, node_strength, pair_groups, sequential_edge_swaps,
                                    skeleton_keys)
from percinterp.sieve import degree_entry

from test_diagnostics import _degrees, skip_graph


def test_decompose_fast_is_bit_identical_to_decompose():
    g = skip_graph(7)
    w = g.weight.copy()
    w[::7] = 0.0  # some non-edges
    s_in, s_out = node_strength(g)
    for ng in (None, g.src, g.dst, pair_groups(g), binned_groups(g, g.edge_layer("dst"), s_out, s_in, n_bins=3)):
        a = decompose(g, w, n_shuffles=6, rng=np.random.default_rng(11), null_groups=ng)
        b = decompose_fast(g, w, n_shuffles=6, rng=np.random.default_rng(11), null_groups=ng)
        assert np.array_equal(a.real.cs, b.real.cs)
        for x, y in zip([a.real] + a.shuffles, [b.real] + b.shuffles):
            for t in ("rF", "rB", "cov", "b"):
                assert np.array_equal(getattr(x, t), getattr(y, t))


def test_sequential_swaps_preserve_degrees_blocks_and_simplicity():
    g = skip_graph(1)
    E = degree_entry(g, g.weight, g.edge_layer("dst"))
    keep = E <= 2.0
    s, d = g.src[keep], g.dst[keep]
    allowed = g.src * g.n_nodes + g.dst
    for skel in (None, allowed, skeleton_keys(g, E)):
        s2, d2, acc = sequential_edge_swaps(g, s, d, np.random.default_rng(0), n_rounds=20, allowed=skel)
        assert acc > 0
        assert np.array_equal(s2, s)  # sources stay put; targets move
        assert not np.array_equal(d2, d)
        ko, ki, _ = _degrees(g, s, d)
        ko2, ki2, _ = _degrees(g, s2, d2)
        for a, b in ((ko, ko2), (ki, ki2)):
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        keys = s2 * g.n_nodes + d2
        assert len(np.unique(keys)) == len(keys)
        assert np.all(g.node_layer[s2] < g.node_layer[d2])
        if skel is not None:
            assert np.isin(keys, allowed).all()
    for blocks in ("dst", "src", "none"):
        s2, d2, acc = sequential_edge_swaps(g, s, d, np.random.default_rng(1), n_rounds=20, blocks=blocks)
        assert np.array_equal(np.bincount(d2, minlength=g.n_nodes), np.bincount(d, minlength=g.n_nodes))
        assert np.all(g.node_layer[s2] < g.node_layer[d2])
        assert len(np.unique(s2 * g.n_nodes + d2)) == len(s2)


def test_sequential_swaps_sample_uniformly():
    # 4 sources -> 4 targets with out-degrees (2, 2, 1, 1) and in-degrees (2, 1, 2, 1). Enumerate
    # every simple bipartite graph with these degrees; the sampler should hit each equally often.
    out_deg, in_deg = (2, 2, 1, 1), (2, 1, 2, 1)
    states = []
    for bits in itertools.product((0, 1), repeat=16):
        M = np.array(bits).reshape(4, 4)
        if tuple(M.sum(1)) == out_deg and tuple(M.sum(0)) == in_deg:
            states.append(M)
    index = {M.tobytes(): k for k, M in enumerate(states)}
    assert len(states) > 5
    r, c = np.nonzero(states[0])
    g = LayeredDAG([4, 4], r, c + 4, np.ones(len(r)))
    rng = np.random.default_rng(3)
    counts = np.zeros(len(states))
    n = 300 * len(states)
    for _ in range(n):
        s2, d2, _ = sequential_edge_swaps(g, r, c + 4, rng, n_swaps=40)
        M = np.zeros((4, 4), dtype=int)
        M[s2, d2 - 4] = 1
        counts[index[M.tobytes()]] += 1
    assert chisquare(counts).pvalue > 1e-3


def test_sequential_joint_degree_null_sees_wiring_and_agrees_with_parallel():
    # the tiny case of test_joint_degree_null_sees_wiring_not_degrees: the one swap kills the backbone
    # (only block (1, 2) has two edges, and its swap is always valid, so an odd number of
    # proposals lands on the rewired graph and an even number returns to the real one)
    g = LayeredDAG([2, 2, 2, 1], [0, 2, 3, 4], [2, 4, 5, 6], np.ones(4))
    s1, d1, acc = sequential_edge_swaps(g, g.src, g.dst, np.random.default_rng(0), n_swaps=1)
    assert acc == 1.0 and set(zip(s1.tolist(), d1.tolist())) == {(0, 2), (2, 5), (3, 4), (4, 6)}
    s2, d2, _ = sequential_edge_swaps(g, g.src, g.dst, np.random.default_rng(0), n_swaps=2)
    assert np.array_equal(d2, g.dst)
    r = joint_degree_null_sequential(g, np.zeros(4), 0.0, n_null=3, n_rounds=1, skeleton=False)
    assert r.real == 0.5 and np.all(r.null == 0.5)  # two proposals: back to the real graph
    # on a random skip graph the two samplers give similar backbone-fraction distributions
    g = skip_graph(2, widths=(10, 30, 30, 30, 1), p=0.3)
    E = degree_entry(g, g.weight, g.edge_layer("dst"))
    a = joint_degree_null(g, E, 0.8, n_null=60, rng=np.random.default_rng(0))
    b = joint_degree_null_sequential(g, E, 0.8, n_null=60, rng=np.random.default_rng(0))
    assert a.real == b.real and a.n_edges == b.n_edges
    sd = max(a.null.std(), b.null.std(), 1e-3)
    assert abs(a.null.mean() - b.null.mean()) < 4 * sd / np.sqrt(30)
