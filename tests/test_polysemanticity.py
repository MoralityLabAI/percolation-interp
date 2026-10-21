import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from percinterp.polysemanticity import (bond_curve, configuration_giant,
    degree_summary, direction_energy, mix_basis, support_edges)


def test_gauge_invariance_and_zero_entries_never_become_edges():
    a = np.array([[1., 0, 2, 0], [0, 3, 0, 4]])
    assert np.allclose(direction_energy(a), direction_energy(a * [[-1e150], [1e-150]]))
    left, right = support_edges(direction_energy(a), 4)
    assert len(left) == 4 and np.all(a[left, right] != 0)
    with pytest.raises(ValueError):
        direction_energy([[0., 0]])


def test_bond_curve_matches_independent_sparse_components_and_susceptibility():
    rng = np.random.default_rng(7)
    nl, nr = 17, 23
    left = rng.integers(nl, size=130)
    right = rng.integers(nr, size=130)  # includes parallel edges
    priority = rng.random(len(left))
    ps = np.linspace(0, 1, 21)
    got = bond_curve(left, right, nl, nr, ps, priority)
    for i, p in enumerate(ps):
        keep = priority < p
        s, d = left[keep], right[keep] + nl
        matrix = coo_matrix((np.ones(2 * len(s)), (np.r_[s, d], np.r_[d, s])), shape=(nl + nr,) * 2)
        _, labels = connected_components(matrix, directed=False)
        sizes = np.bincount(labels[:nl])
        sizes = sizes[sizes > 0]
        largest = sizes.max()
        remaining = nl - largest
        chi = (np.sum(sizes * sizes) - largest * largest) / remaining if remaining else 0
        assert got["giant_left_fraction"][i] == largest / nl
        assert got["susceptibility_left"][i] == chi
        assert got["left_cluster_count"][i] == len(sizes)
    assert np.all(np.diff(got["giant_left_fraction"]) >= 0)


def test_monosemantic_and_shared_coordinate_controls():
    n = 64
    left = np.arange(n)
    ps = [0, 1]
    mono = bond_curve(left, np.arange(n), n, n, ps, np.full(n, .5))
    shared = bond_curve(left, np.zeros(n, dtype=int), n, n, ps, np.full(n, .5))
    assert mono["giant_left_fraction"][-1] == 1 / n
    assert shared["giant_left_fraction"][-1] == 1
    # A single macroscopic star violates the locally-tree-like finite-moment
    # ensemble assumptions: degree criterion alone cannot certify mixing.
    assert degree_summary(left, np.zeros(n, dtype=int), n, n)["branching_product"] == 0


def test_degree_preservation_under_configuration_rewiring():
    rng = np.random.default_rng(91)
    left = np.repeat(np.arange(20), 4)
    right = rng.integers(50, size=len(left))
    a = degree_summary(left, right, 20, 50)
    b = degree_summary(left, rng.permutation(right), 20, 50)
    assert np.array_equal(a["left"], b["left"])
    assert np.array_equal(a["right"], b["right"])
    assert a["configuration_bond_pc"] == b["configuration_bond_pc"]


def test_orthogonal_basis_control_preserves_gram_but_changes_coordinate_sharing():
    a = np.eye(16)
    rotated = mix_basis(a, np.random.default_rng(3))
    assert np.allclose(a @ a.T, rotated @ rotated.T, atol=1e-12)
    assert not np.allclose(direction_energy(a), direction_energy(rotated))
    assert np.allclose(direction_energy(rotated).sum(1), 1)


def test_pgf_regular_bipartite_threshold_and_exact_fixed_point():
    # 3-regular on both sides: kappa_L*kappa_R=4 => p_c=1/2.
    ps = np.array([0., .25, .49, .75, 1.])
    s = configuration_giant(np.full(100, 3), np.full(100, 3), ps)
    assert np.allclose(s[:3], 0)
    # Symmetric message x=1-p+p*x^2; smallest root=(1-p)/p above pc.
    expected = 1 - ((1 - ps[3:]) / ps[3:]) ** 3
    assert np.allclose(s[3:], expected, atol=1e-9)
    with pytest.raises(ValueError):
        configuration_giant(np.array([1]), np.array([2]), ps)
