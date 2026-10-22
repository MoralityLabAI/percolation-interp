"""E09 audit regression tests and topology/fidelity counterexamples."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from percinterp import LayeredDAG, components, profile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "percinterp_e09_audit", ROOT / "experiments" / "e09_utility_audit.py")
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_template_interpolation_and_validation():
    assert audit.linear_interpolate([0, 1, 2], [0, 10, 20], 0.75) == 7.5
    assert audit.linear_interpolate([0, 1], [3, 9], -1) == 3
    assert audit.linear_interpolate([0, 1], [3, 9], 4) == 9
    with pytest.raises(ValueError):
        audit.linear_interpolate([0], [0, 1], 0)


def test_real_e02_comparison_is_reproducible():
    report = audit.build_report(ROOT)
    x = report["fidelity_transfer"]
    assert x["n_runs"] == 13
    assert x["n_groups"] == 5
    assert all(g["n_transition_points"] > 0 for g in x["groups"])
    m = x["macro_mae"]
    assert m["retained_fraction"] < m["backbone_fraction"]
    assert m["retained_fraction"] < m["kept_outdegree"]
    assert x["backbone_addon_worse_in_groups"] == x["n_groups"]
    assert math.isclose(m["retained_fraction"], 0.020768537964245472,
                        rel_tol=0, abs_tol=1e-10)
    assert math.isclose(m["backbone_fraction"], 0.05236142093833392,
                        rel_tol=0, abs_tol=1e-10)
    assert math.isclose(m["retained_plus_backbone"], 0.021989176522445142,
                        rel_tol=0, abs_tol=1e-10)


def test_real_selection_and_degree_null_results():
    report = audit.build_report(ROOT)
    cases = report["causal_cases"]
    assert len(cases) == 5
    assert all(row["benefit_vs_importance_nats"] > 0 for row in cases)
    toy = [r for r in cases if r["case"].startswith("toy_")]
    qwen = [r for r in cases if r["case"].startswith("qwen_")]
    assert len(toy) == 3 and len(qwen) == 2
    assert min(x["additional_gap_closed_fraction"] for x in toy) > 0.05
    assert max(x["additional_gap_closed_fraction"] for x in qwen) < 0.02
    assert all(x["degree_determined_share"] > 0.995
               for x in report["degree_null_cases"])
    assert all(not x["wiring_excess_by_registered_threshold"]
               for x in report["degree_null_cases"])


def test_identical_degrees_can_hide_different_directed_reachability():
    # One designated source (0) and sink (6); nodes 1 and 7 are not task endpoints.
    # Two middle-layer edge switches preserve *each node's* in/out degrees.
    # Both graphs also have identical undirected largest-cluster fractions.
    src = np.array([0, 1, 2, 3, 4, 5])
    dst_live = np.array([2, 3, 4, 5, 6, 7])
    dst_dead = np.array([2, 3, 5, 4, 6, 7])
    sources = np.zeros(8, dtype=bool)
    sinks = np.zeros(8, dtype=bool)
    sources[0] = True
    sinks[6] = True
    graphs = [
        LayeredDAG([2, 2, 2, 2], src, dst, np.ones(6),
                   sources=sources, sinks=sinks)
        for dst in (dst_live, dst_dead)
    ]
    for u in (src,):
        assert np.array_equal(np.bincount(u, minlength=8),
                              np.bincount(graphs[1].src, minlength=8))
    assert np.array_equal(np.bincount(dst_live, minlength=8),
                          np.bincount(dst_dead, minlength=8))
    entry = np.full(6, 0.5)
    assert profile(graphs[0], entry).p_span == 0.5
    assert math.isinf(profile(graphs[1], entry).p_span)
    assert components(graphs[0], entry, [1])["largest"][0] == 0.5
    assert components(graphs[1], entry, [1])["largest"][0] == 0.5


def test_connectivity_is_insufficient_under_signed_path_cancellation():
    src = [0, 0, 1, 2]
    dst = [1, 2, 3, 3]
    positive = LayeredDAG([1, 2, 1], src, dst, [1, 1, 1, 1])
    cancel = LayeredDAG([1, 2, 1], src, dst, [1, 1, 1, -1])
    entry = np.ones(4)
    assert profile(positive, entry).p_span == profile(cancel, entry).p_span == 1
    assert np.allclose(abs(positive.weight), abs(cancel.weight))

    def output(g):
        act = np.zeros(g.n_nodes)
        act[0] = 1
        for l in range(g.n_layers - 1):
            for i in range(g.n_edges):
                if g.node_layer[g.src[i]] == l:
                    act[g.dst[i]] += act[g.src[i]] * g.weight[i]
        return act[-1]

    assert output(positive) == 2
    assert output(cancel) == 0
