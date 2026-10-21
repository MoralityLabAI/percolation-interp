import numpy as np

from percinterp import LayeredDAG
from percinterp.controls import complete_layered
from percinterp.diagnostics import (backbone_anatomy, binned_groups, decompose, decouple_in_out, degree_determined_share,
                                    double_edge_swaps,
                                    joint_degree_null, kept_coverage, node_strength, pair_groups,
                                    strength_correlation, uniform_coverage)
from percinterp.sieve import degree_entry, sieve

from test_sieve import planted_graph


def skip_graph(seed=0, widths=(6, 10, 10, 10, 1), p=0.5):
    """Random layered graph with skip edges and a single sink."""
    rng = np.random.default_rng(seed)
    off = np.concatenate([[0], np.cumsum(widths)])
    src, dst = [], []
    for a in range(len(widths)):
        for b in range(a + 1, len(widths)):
            i, j = np.indices((widths[a], widths[b])).reshape(2, -1)
            keep = rng.random(len(i)) < (p if b == a + 1 else p / 3)
            src.append(i[keep] + off[a])
            dst.append(j[keep] + off[b])
    src, dst = np.concatenate(src), np.concatenate(dst)
    return LayeredDAG(list(widths), src, dst, rng.lognormal(0, 1.5, len(src)))


def test_decomposition_identity_and_matches_sieve():
    for g, w in [(planted_graph()[0], planted_graph()[1]), (skip_graph(), skip_graph().weight)]:
        res = sieve(g, w, n_shuffles=8, rng=np.random.default_rng(7))
        dn = decompose(g, w, cs=res.cs, n_shuffles=8, rng=np.random.default_rng(7))
        for d in [dn.real] + dn.shuffles:
            assert np.allclose(d.b, d.rF * d.rB + d.cov, atol=1e-12)
        assert np.allclose(dn.real.total("b"), res.real, atol=1e-12)
        assert np.allclose([d.total("b") for d in dn.shuffles], res.shuffles, atol=1e-12)
        assert np.allclose(dn.excess("b"), dn.excess("coverage") + dn.excess("cov"), atol=1e-12)
        assert np.allclose(dn.excess("b"), res.learned, atol=1e-12)


def _degrees(g, s, d):
    blk_s = g.node_layer[s] * g.n_layers + g.node_layer[d]
    out = {(int(a), int(b)) for a, b in zip(s, blk_s)}
    ko = np.unique(np.stack([s, blk_s]), axis=1, return_counts=True)
    ki = np.unique(np.stack([d, blk_s]), axis=1, return_counts=True)
    return ko, ki, out


def test_double_edge_swaps_preserve_degrees_blocks_and_simplicity():
    g = skip_graph(1)
    E = degree_entry(g, g.weight, g.edge_layer("dst"))
    keep = E <= 2.0
    s, d = g.src[keep], g.dst[keep]
    allowed = g.src * g.n_nodes + g.dst
    for skel in (None, allowed):
        s2, d2, acc = double_edge_swaps(g, s, d, np.random.default_rng(0), n_rounds=20, allowed=skel)
        assert acc > 0
        assert not (np.array_equal(s2, s) and np.array_equal(d2, d))
        ko, ki, _ = _degrees(g, s, d)
        ko2, ki2, _ = _degrees(g, s2, d2)
        for a, b in ((ko, ko2), (ki, ki2)):
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        keys = s2 * g.n_nodes + d2
        assert len(np.unique(keys)) == len(keys)  # no multi-edges
        assert np.all(g.node_layer[s2] < g.node_layer[d2])
        if skel is not None:
            assert np.isin(keys, allowed).all()


def test_joint_degree_null_runs_and_is_flat_on_random_graph():
    g = complete_layered([30] * 5, np.random.default_rng(2))
    E = degree_entry(g, g.weight, g.edge_layer("dst"))
    r = joint_degree_null(g, E, 1.5, n_null=30, rng=np.random.default_rng(1))
    assert r.n_edges == (E <= 1.5).sum()
    assert 0 <= r.real <= 1 and r.null.shape == (30,)
    assert abs(r.z) < 4


def test_joint_degree_null_sees_wiring_not_degrees():
    # sources a0 a1 -> b0 b1 -> c0 c1 -> sink. Kept: a0->b0, b0->c0, b1->c1, c0->s.
    # Real interior backbone: b0, c0 (1/2). The one swap in block (1, 2) gives b0->c1, b1->c0,
    # and no backbone at all.
    g = LayeredDAG([2, 2, 2, 1], [0, 2, 3, 4], [2, 4, 5, 6], np.ones(4))
    r = joint_degree_null(g, np.zeros(4), 0.0, n_null=5, n_rounds=1, skeleton=False)
    assert r.real == 0.5 and np.all(r.null == 0.0)
    # planted chains among planted nodes only: the backbone is fixed by the kept degrees, so the
    # rewired graphs keep it
    g, w, _ = planted_graph(width=60, n_planted=10)
    E = degree_entry(g, w, g.edge_layer("dst"))
    r = joint_degree_null(g, E, 1.5, n_null=10, rng=np.random.default_rng(1))
    assert np.allclose(r.null, r.real)


def test_strength_coverage_anatomy():
    g = skip_graph(3)
    s_in, s_out = node_strength(g)
    assert np.isclose(s_in.sum(), g.weight.sum()) and np.isclose(s_out.sum(), g.weight.sum())
    sc = strength_correlation(g)
    assert -1 <= sc["pooled_within_group_rho"] <= 1
    assert set(sc["per_group"]) <= {1, 2, 3}
    E = degree_entry(g, g.weight, g.edge_layer("dst"))
    kc = kept_coverage(g, E, 1.0)
    assert all(0 <= r["in_cov"] <= 1 for r in kc["layers"])
    assert np.isclose(uniform_coverage(1, 10), 0.1)
    an = backbone_anatomy(g, E, 3.0)
    assert sum(an["nodes_per_layer"]) == an["n_nodes"]
    assert an["min_hops"] is None or an["min_hops"] >= 1


def test_path_counts_on_tiny_graph():
    # 0 -> 1 -> 3 (sink), 0 -> 3 (skip), 0 -> 2 -> 3 ; layers [1, 2, 1]
    g = LayeredDAG([1, 2, 1], [0, 1, 0, 0, 2], [1, 3, 3, 2, 3], [1.0, 1.0, 1.0, 1.0, 1.0])
    E = np.zeros(g.n_edges)
    an = backbone_anatomy(g, E, 0.0)
    assert an["path_counts_by_hops"] == {1: 1.0, 2: 2.0}
    assert an["edge_span_counts"] == {1: 4, 2: 1}


def test_group_labels():
    g = skip_graph(4)
    pg = pair_groups(g)
    assert len(np.unique(pg)) == len({(a, b) for a, b in zip(g.node_layer[g.src], g.node_layer[g.dst])})
    s_in, s_out = node_strength(g)
    bg = binned_groups(g, g.edge_layer("dst"), s_out, s_in, n_bins=3)
    # refinement: every new group lies within one base group
    base = g.edge_layer("dst")
    for lab in np.unique(bg):
        assert len(np.unique(base[bg == lab])) == 1


def test_coarser_swap_blocks_keep_total_degrees_and_layer_order():
    g = skip_graph(5)
    E = degree_entry(g, g.weight, g.edge_layer("dst"))
    keep = E <= 2.0
    s, d = g.src[keep], g.dst[keep]
    for blocks in ("dst", "src", "none"):
        s2, d2, acc = double_edge_swaps(g, s, d, np.random.default_rng(1), n_rounds=20, blocks=blocks)
        assert acc > 0
        assert np.array_equal(np.bincount(s2, minlength=g.n_nodes), np.bincount(s, minlength=g.n_nodes))
        assert np.array_equal(np.bincount(d2, minlength=g.n_nodes), np.bincount(d, minlength=g.n_nodes))
        assert np.all(g.node_layer[s2] < g.node_layer[d2])
        keys = s2 * g.n_nodes + d2
        assert len(np.unique(keys)) == len(keys)


def test_decouple_keeps_in_edges_and_strength_sequences():
    g = skip_graph(6)
    gn = decouple_in_out(g, np.random.default_rng(0))
    assert np.array_equal(gn.dst, g.dst)
    s_in, s_out = node_strength(g)
    t_in, t_out = node_strength(gn)
    assert np.allclose(s_in, t_in)
    for l in range(g.n_layers):
        sl = g.layer_nodes(l)
        assert np.allclose(np.sort(s_out[sl]), np.sort(t_out[sl]))
    # same ranks: degree_entry is unchanged by the relabelling
    E = degree_entry(g, g.weight, g.edge_layer("dst"))
    assert np.array_equal(E, degree_entry(gn, g.weight, gn.edge_layer("dst")))
    dn = decompose(g, g.weight, n_shuffles=4, null="decouple")
    for d in dn.shuffles:
        assert np.allclose(d.b, d.rF * d.rB + d.cov)


def test_degree_determined_share():
    # one interior layer: membership is exactly "kept in-edge and kept out-edge"
    g = complete_layered([20, 20, 20], np.random.default_rng(0))
    E = degree_entry(g, g.weight, g.edge_layer("dst"))
    for c in (0.3, 1.0, 3.0):
        d = degree_determined_share(g, E, c)
        assert d["share_equal"] == 1.0 and d["backbone"] == d["local"]
    # the rewired graph of the joint-degree test: a0->b0, b0->c1, b1->c0, c0->s. b0 and c0 have
    # kept in- and out-edges but no source-to-sink path
    g = LayeredDAG([2, 2, 2, 1], [0, 2, 3, 4], [2, 5, 4, 6], np.ones(4))
    d = degree_determined_share(g, np.zeros(4), 0.0)
    assert d["backbone"] == 0.0 and d["local"] == 0.5 and d["share_equal"] == 0.5
