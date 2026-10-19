import numpy as np

from percinterp import LayeredDAG
from percinterp.controls import complete_layered
from percinterp.sieve import degree_entry, sieve


def planted_graph(seed=0, width=40, depth=4, n_planted=6):
    """Complete layered graph with i.i.d. weights plus a planted set of strong chains."""
    rng = np.random.default_rng(seed)
    g = complete_layered([width] * (depth + 1), rng)
    w = rng.random(g.n_edges)
    planted = rng.choice(width, size=n_planted, replace=False)
    nodes = [g.offsets[l] + planted for l in range(depth + 1)]
    strong = np.zeros(g.n_edges, dtype=bool)
    for l in range(depth):
        src_ok = np.isin(g.src, nodes[l])
        dst_ok = np.isin(g.dst, nodes[l + 1])
        strong |= src_ok & dst_ok
    w[strong] += 5.0
    planted_mask = np.zeros(g.n_nodes, dtype=bool)
    for l in range(1, depth):
        planted_mask[nodes[l]] = True
    return g, w, planted_mask


def test_degree_entry_units():
    rng = np.random.default_rng(1)
    g = complete_layered([5, 8, 8], rng)
    E = degree_entry(g, g.weight, g.edge_layer("dst"))
    for layer in (1, 2):
        m = g.edge_layer("dst") == layer
        n_src = len(np.unique(g.src[m]))
        for c in (0.5, 1.0, 3.0):
            assert (E[m] <= c).sum() == int(np.floor(c * n_src + 1e-9))


def test_sieve_finds_planted_structure_and_null_is_flat():
    g, w, planted = planted_graph()
    res = sieve(g, w, n_shuffles=10)
    assert res.peak > 0.05
    assert res.p_value <= 1 / 11 + 1e-9
    # every planted interior node is on the sieved backbone
    assert res.backbone[planted].all()
    # an unstructured graph shows no learned connectivity beyond its null
    rng = np.random.default_rng(3)
    g2 = complete_layered([40] * 5, rng)
    res2 = sieve(g2, rng.random(g2.n_edges), n_shuffles=10)
    assert res2.p_value > 0.05


def test_sieve_keeps_null_entries_matching_its_shuffles():
    g, w, _ = planted_graph()
    res = sieve(g, w, n_shuffles=6, keep_null_entries=True, rng=np.random.default_rng(5))
    assert res.null_node_entry.shape == (6, g.n_nodes)
    nodes = ~(g.sources | g.sinks)
    again = (res.null_node_entry[:, nodes][:, None, :] <= res.cs[None, :, None]).mean(2)
    assert np.allclose(again, res.shuffles)
    assert sieve(g, w, n_shuffles=2).null_node_entry is None
