import numpy as np
import torch

from percinterp import LayeredDAG, profile, rank_entry
from percinterp import dense
from percinterp.controls import init_mlp, mlp_edge_matrices
from percinterp.modadd import (
    MLP,
    all_pairs,
    edge_scores,
    masked_logits,
    mean_activations,
    metric_direction,
    one_hot_pairs,
)


def test_dense_profile_matches_layered_dag():
    rng = np.random.default_rng(0)
    widths = [7, 9, 9, 6, 4]
    for trial in range(10):
        Ws = init_mlp(widths, rng)
        mats = mlp_edge_matrices(Ws, rng.normal(size=widths[0]), kind="eap", readout=rng.normal(size=4))
        g = LayeredDAG.from_dense(mats)
        prof = profile(g, rank_entry(g.weight, g.edge_layer()))
        E = dense.rank_entry([torch.tensor(m, dtype=torch.float32)[None] for m in mats])
        dp = dense.DenseProfile(E)
        assert np.isclose(dp.p_span.item(), prof.p_span, atol=1e-6)
        ps = torch.linspace(0.01, 1.0, 25)
        hidden = list(range(1, len(widths) - 1))
        assert np.allclose(
            dp.backbone_node_fraction(ps, hidden)[0].numpy(),
            prof.backbone_node_fraction(ps.numpy(), hidden),
            atol=1e-6,
        )
        # node-level entries agree, layer by layer
        for l in range(len(widths)):
            assert np.allclose(dp.node_entry(l)[0].numpy(), prof.node_entry[g.layer_nodes(l)], atol=1e-6)
        assert np.allclose(
            dp.backbone_edge_fraction(ps)[0].numpy(), prof.backbone_edge_fraction(ps.numpy()), atol=1e-6
        )


def test_shuffled_entry_is_a_ranking_on_the_same_skeleton():
    S = [torch.rand(3, 5, 4) * (torch.rand(3, 5, 4) > 0.3)]
    q = dense.shuffled_entry(S)[0]
    assert torch.equal(torch.isfinite(q), S[0] > 0)
    for b in range(3):
        vals = torch.sort(q[b][torch.isfinite(q[b])])[0]
        n = len(vals)
        assert torch.allclose(vals, torch.arange(1, n + 1, dtype=torch.float32) / n)


def test_reference_entry_is_monotone_and_matches_counts():
    ref = [torch.tensor([1.0, 2.0, 3.0, 4.0])]
    S = [torch.tensor([[[0.0, 1.0, 2.5, 5.0]]])]
    q = dense.reference_entry(S, ref)[0][0, 0]
    assert torch.isinf(q[0])
    assert torch.allclose(q[1:], torch.tensor([1.0, 0.5, 0.0]))


def _model(P=11, depth=2, width=16, seed=0):
    torch.manual_seed(seed)
    return MLP(P, depth, width).double()


def test_masked_forward_endpoints():
    P = 11
    m = _model(P)
    a, b, c = all_pairs(P)
    x = one_hot_pairs(a, b, P).double()
    means = [t.double() for t in mean_activations(m, P)]
    ones = [torch.ones(len(x), lin.weight.shape[0] if l < 2 else 1, lin.weight.shape[1], dtype=torch.float64)
            for l, lin in enumerate(m.layers)]
    assert torch.allclose(masked_logits(m, x, ones, means), m(x))
    zeros = [torch.zeros_like(t) for t in ones]
    out = masked_logits(m, x, zeros, means)
    assert torch.allclose(out, out[:1].expand_as(out))  # fully ablated: same output for every input


def test_eap_scores_match_finite_differences():
    P = 7
    m = _model(P, depth=2, width=6, seed=1)
    a, b, c = all_pairs(P)
    idx = torch.tensor([3, 17, 40])
    x = one_hot_pairs(a[idx], b[idx], P).double()
    c = c[idx]
    means = [t.double() for t in mean_activations(m, P)]
    scores = edge_scores(m, x, c, means, steps=1)
    shapes = [(lin.weight.shape[0] if l < 2 else 1, lin.weight.shape[1]) for l, lin in enumerate(m.layers)]
    ones = [torch.ones(len(x), *s, dtype=torch.float64) for s in shapes]
    mdir = metric_direction(m, c).double()

    def metric(masks):
        return (masked_logits(m, x, masks, means) * mdir).sum(1)

    base = metric(ones)
    eps = 1e-6
    rng = np.random.default_rng(0)
    for l, (n_out, n_in) in enumerate(shapes):
        for _ in range(5):
            i, j = rng.integers(n_out), rng.integers(n_in)
            masks = [t.clone() for t in ones]
            masks[l][:, i, j] = 1 - eps
            fd = ((base - metric(masks)) / eps).abs()
            assert torch.allclose(scores[l][:, i, j].double(), fd, rtol=1e-4, atol=1e-8), (l, i, j)
