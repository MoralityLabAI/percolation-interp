import numpy as np
import pytest

from percinterp import profile, rank_entry
from percinterp.controls import complete_layered
from percinterp.theory import giant_fraction, mean_field_reach, path_count_threshold, reach_chain


def test_chain_trivial_cases():
    assert reach_chain([5, 5, 5], 0.0)["span_prob"] == 0.0
    assert np.isclose(reach_chain([5, 5, 5], 1.0)["span_prob"], 1.0)
    # single edge layer, one source, one sink: spanning iff the edge is kept
    assert np.isclose(reach_chain([1, 1], 0.3)["span_prob"], 0.3)
    # two sources, one sink: 1 - (1-p)^2
    assert np.isclose(reach_chain([2, 1], 0.3)["span_prob"], 1 - 0.7**2)


@pytest.mark.parametrize("widths", [[12, 12, 12, 12], [20, 8, 20, 8, 20]])
def test_monte_carlo_matches_exact_chain(widths):
    rng = np.random.default_rng(0)
    g = complete_layered(widths, rng)
    n = 3000
    p_span = np.array([profile(g, rng.random(g.n_edges)).p_span for _ in range(n)])
    last = len(widths) - 1
    for p in [0.03, 0.06, 0.1, 0.15]:
        exact = reach_chain(widths, p)["span_prob"]
        mc = (p_span <= p).mean()
        se = np.sqrt(max(exact * (1 - exact), 1e-4) / n)
        assert abs(mc - exact) < 4.5 * se, (p, mc, exact)


def test_mean_reach_matches_chain():
    rng = np.random.default_rng(1)
    widths = [30] * 6
    g = complete_layered(widths, rng)
    p = 0.05
    fr = [profile(g, rng.random(g.n_edges)).reach_fraction([p], 5)[0] for _ in range(400)]
    exact = reach_chain(widths, p)["mean_reach_frac"][5]
    assert abs(np.mean(fr) - exact) < 4.5 * np.std(fr) / np.sqrt(400)


def test_rank_and_bernoulli_agree_for_iid_weights():
    rng = np.random.default_rng(2)
    widths = [40] * 5
    g = complete_layered(widths, rng)
    bern, rank = [], []
    for _ in range(400):
        w = rng.random(g.n_edges)
        bern.append(profile(g, w).p_span)
        rank.append(profile(g, rank_entry(w)).p_span)
    # fixed-count and Bernoulli ensembles have nearly the same spanning point
    assert abs(np.median(bern) - np.median(rank)) < 0.002


def test_chain_approaches_mean_field_at_large_width():
    N, depth = 1500, 25
    for c in [0.7, 1.5, 3.0]:
        exact = reach_chain([N] * (depth + 1), c / N)["mean_reach_frac"]
        mf = mean_field_reach(c, depth)
        assert np.max(np.abs(exact - mf)) < 0.01


def test_mean_field_limits():
    assert np.allclose(giant_fraction([0.5, 1.0]), 0.0)
    rho = giant_fraction([2.0])[0]
    assert np.isclose(rho, 1 - np.exp(-2 * rho))
    assert np.isclose(mean_field_reach(2.0, 200)[-1], rho)
    # critical decay ~ 2/l
    r = mean_field_reach(1.0, 2000)
    assert abs(r[-1] * 2000 / 2 - 1) < 0.02


def test_path_count_threshold():
    assert np.isclose(path_count_threshold([10, 10, 10]), 10 ** (-1.5))
