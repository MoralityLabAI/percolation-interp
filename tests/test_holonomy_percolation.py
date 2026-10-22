"""E08 mathematical calibration, independent of network checkpoint and GPU."""
import numpy as np
import pytest

from percinterp.cycle_filtration import cycle_filtration
from percinterp.holonomy import (analyze_connection, frame_edge_transports,
                                 loop_matrix, signature)


SRC = np.array([0, 1, 1, 0, 2, 2, 0])
DST = np.array([3, 3, 4, 4, 4, 5, 5])
ENTRY = np.array([.08, .15, .24, .37, .43, .55, .68])


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def q_planted():
    q = np.repeat(np.eye(2)[None], 7, axis=0)
    q[3] = rot(np.pi / 4)
    q[6] = rot(np.pi / 2)
    return q


def test_cycle_births_and_component_denominator():
    ps = [0, .15, .36, .37, .55, .68, 1]
    f = cycle_filtration(6, SRC, DST, ENTRY, ps, n_component_vertices=3)
    assert f.betti1.tolist() == [0, 0, 0, 1, 1, 2, 2]
    assert f.kept_edges.tolist() == [0, 2, 3, 4, 6, 7, 7]
    assert f.largest[0] == pytest.approx(1 / 3)
    assert f.largest[1] == pytest.approx(2 / 3)
    assert f.largest[-1] == pytest.approx(1)
    assert [c.birth for c in f.cycles] == pytest.approx([.37, .68])
    for c in f.cycles:
        assert c.vertices[0] == c.vertices[-1]
        assert len(c.steps) == len(c.vertices) - 1
        for (edge, sign), a, b in zip(c.steps, c.vertices[:-1], c.vertices[1:]):
            expected = (a, b) if sign == 1 else (b, a)
            assert (SRC[edge], DST[edge]) == expected


def test_known_planted_angles_and_onset():
    d = analyze_connection(6, SRC, DST, ENTRY, q_planted(), n_component_vertices=3)
    assert d["p_h_known"] and d["p_h"] == pytest.approx(.37)
    assert [q["angle_degrees"] for q in d["loops"]] == pytest.approx([45, 90])
    assert d["total_certified_cycles"] == d["cycles_recorded"] == 2
    assert d["betti1"][-1] == 2
    assert all(q["determinant"] == pytest.approx(1) for q in d["loops"])


def test_flat_connection_even_with_rotated_local_gauges():
    gauge = np.stack([rot(i * 0.33) for i in range(6)])
    q = np.stack([gauge[v].T @ gauge[u] for u, v in zip(SRC, DST)])
    d = analyze_connection(6, SRC, DST, ENTRY, q, n_component_vertices=3)
    assert d["p_h"] is None and d["p_h_known"]
    assert max(c["frobenius_distance"] for c in d["loops"]) < 1e-12


def test_gauge_invariance_of_curved_connection():
    q = q_planted()
    G = np.stack([rot(i * .27) for i in range(6)])
    transformed = np.stack([G[v].T @ q[e] @ G[u] for e, (u, v) in enumerate(zip(SRC, DST))])
    a = analyze_connection(6, SRC, DST, ENTRY, q)
    b = analyze_connection(6, SRC, DST, ENTRY, transformed)
    assert a["p_h"] == b["p_h"]
    for x, y in zip(a["loops"], b["loops"]):
        assert x["angle_degrees"] == pytest.approx(y["angle_degrees"])
        assert x["frobenius_distance"] == pytest.approx(y["frobenius_distance"])


def test_uncertified_edge_removes_false_holonomy_loops():
    reliable = np.ones(7, dtype=bool)
    reliable[3] = False
    d = analyze_connection(6, SRC, DST, ENTRY, q_planted(), reliable=reliable)
    assert d["betti1"][-1] == 2
    assert d["certified_betti1"][-1] == 1
    assert d["cycles_recorded"] == 1
    assert d["loops"][0]["birth"] == pytest.approx(.68)


def test_truncation_reports_uncertainty_without_changing_betti():
    q = q_planted()
    q[3] = np.eye(2)  # first cycle flat, second remains curved
    partial = analyze_connection(6, SRC, DST, ENTRY, q, max_recorded_cycles=1)
    assert partial["betti1"][-1] == 2
    assert partial["p_h"] is None and not partial["p_h_known"]
    complete = analyze_connection(6, SRC, DST, ENTRY, q, max_recorded_cycles=2)
    assert complete["p_h"] == pytest.approx(.68) and complete["p_h_known"]


def test_rank_one_discrete_holonomy():
    q = np.ones((7, 1, 1))
    q[3] = -1
    d = analyze_connection(6, SRC, DST, ENTRY, q)
    h = d["loops"][0]
    assert h["angle_degrees"] is None
    assert h["frobenius_distance"] == pytest.approx(2)
    assert h["determinant"] == pytest.approx(-1)


def test_frames_flat_and_validation():
    rng = np.random.default_rng(42)
    common, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    frames = np.stack([common @ rot(i * .21) for i in range(6)])
    q, reliable, overlap = frame_edge_transports(frames, SRC, DST)
    assert reliable.all() and np.allclose(overlap, 1)
    d = analyze_connection(6, SRC, DST, ENTRY, q)
    assert max(c["frobenius_distance"] for c in d["loops"]) < 1e-12
    frames[0, 0, 0] += 0.2
    with pytest.raises(ValueError, match="orthonormal"):
        frame_edge_transports(frames, SRC, DST)


def test_spherical_tangent_frames_have_curvature():
    # A nontrivial, geometry-derived holonomy (not the manually planted Q test).
    n = np.array([[1., 0., 0.], [.7, .7, 0.], [.7, 0, .7]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    east = np.cross(np.array([0., 0., 1.]), n)
    # At the pole use an independent orthogonal seed instead.
    east /= np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(n, east)
    frames = np.stack([east, north], axis=2)
    s, d, p = np.array([0, 1, 2]), np.array([1, 2, 0]), np.array([.2, .3, .4])
    q, valid, _ = frame_edge_transports(frames, s, d, min_overlap=.1)
    assert valid.all()
    result = analyze_connection(3, s, d, p, q)
    assert result["p_h_known"]
    assert result["p_h"] == pytest.approx(.4)
    assert 1 < result["loops"][0]["angle_degrees"] < 90


def test_parallel_edges_and_empty_graph():
    f = cycle_filtration(3, [0, 0], [1, 1], [.1, .2], [.1, .2])
    assert f.betti1.tolist() == [0, 1]
    assert len(f.cycles[0].steps) == 2
    e = cycle_filtration(3, [], [], [], [0, 1])
    assert e.betti1.tolist() == [0, 0]
    assert e.largest.tolist() == pytest.approx([1 / 3, 1 / 3])


@pytest.mark.parametrize("seed", range(8))
def test_cycle_rank_matches_brute_random_graphs(seed):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, 9, 50)
    dst = rng.integers(0, 9, 50)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    entry = rng.random(len(src))
    ps = np.linspace(0, 1, 11)
    f = cycle_filtration(9, src, dst, entry, ps, max_recorded_cycles=1)
    for i, p in enumerate(ps):
        parent = list(range(9))
        def root(x):
            while x != parent[x]:
                x = parent[x]
            return x
        for u, v in zip(src[entry <= p], dst[entry <= p]):
            parent[root(int(u))] = root(int(v))
        classes = {root(i) for i in range(9)}
        expected = int((entry <= p).sum()) - 9 + len(classes)
        assert f.betti1[i] == expected


def test_reflection_gauge_preserves_rotation_magnitude():
    q = q_planted()
    G = np.stack([rot(i * .11) for i in range(6)])
    G[0] = G[0] @ np.diag([1, -1])
    transformed = np.stack([G[v].T @ q[e] @ G[u] for e, (u, v) in enumerate(zip(SRC, DST))])
    a = analyze_connection(6, SRC, DST, ENTRY, q)
    b = analyze_connection(6, SRC, DST, ENTRY, transformed)
    assert a["loops"][0]["angle_degrees"] == pytest.approx(b["loops"][0]["angle_degrees"])
    assert a["loops"][0]["signed_angle_degrees"] == pytest.approx(-b["loops"][0]["signed_angle_degrees"])


def test_onset_certified_even_with_later_loops_capped():
    d = analyze_connection(6, SRC, DST, ENTRY, q_planted(), max_recorded_cycles=1)
    assert d["truncated"]
    assert d["p_h_known"] and d["p_h"] == pytest.approx(.37)


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError, match="entry keys"):
        cycle_filtration(2, [0], [1], [1.1])
    with pytest.raises(ValueError, match="self-edges"):
        cycle_filtration(1, [0], [0], [.3])
    with pytest.raises(ValueError, match="rank"):
        analyze_connection(2, [0], [1], [.3], np.ones((1, 0, 0)))
    with pytest.raises(ValueError, match="orthogonal"):
        analyze_connection(2, [0], [1], [.3], np.ones((1, 2, 2)))


def test_nonabelian_rank_three_gauge_conjugacy():
    # Two loops in SO(3), carrying rotations around different axes.
    def rx(t):
        return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)],
                         [0, np.sin(t), np.cos(t)]])
    def ry(t):
        return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0],
                         [-np.sin(t), 0, np.cos(t)]])
    q = np.repeat(np.eye(3)[None], 7, axis=0)
    q[3] = rx(.51)
    q[6] = ry(.67)
    baseline = analyze_connection(6, SRC, DST, ENTRY, q)
    G = np.stack([rx(i * .2) @ ry(i * .17) for i in range(6)])
    shifted = np.stack([G[v].T @ q[e] @ G[u] for e, (u, v) in enumerate(zip(SRC, DST))])
    changed = analyze_connection(6, SRC, DST, ENTRY, shifted)
    assert baseline["p_h"] == changed["p_h"] == pytest.approx(.37)
    for a, b in zip(baseline["loops"], changed["loops"]):
        assert a["angle_degrees"] == pytest.approx(b["angle_degrees"])
        assert a["frobenius_distance"] == pytest.approx(b["frobenius_distance"])


def test_overlap_threshold_rejects_ill_conditioned_link():
    frames = np.array([
        [[1., 0.], [0., 1.], [0., 0.], [0., 0.]],
        [[1., 0.], [0., 0.], [0., 1.], [0., 0.]],
    ])
    q, reliable, singular = frame_edge_transports(frames, [0], [1], min_overlap=.2)
    assert q.shape == (1, 2, 2)
    assert singular[0] == pytest.approx(0)
    assert not reliable[0]
