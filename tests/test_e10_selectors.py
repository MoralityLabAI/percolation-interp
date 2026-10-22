"""Zero-GPU tests for E10 pre-score selector freezing and paired causal scoring."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from percinterp.causal_gate import combine_seed_results, score_sequential_kl
from percinterp.selector_gate import DETERMINISTIC, contact_scores, rank_top, select_sets

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e10_prepare", ROOT / "experiments" / "e10_prepare.py")
assert SPEC and SPEC.loader
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


def test_fixture_has_equal_module_budget_and_degree_alias():
    plan = prep.fixture(n_random=10, seed=13)
    assert plan["source_type"] == "SYNTHETIC_SELECTION_FIXTURE"
    assert plan["module_quotas"] == {"block0": 1, "block1": 1}
    assert len(plan["selection"]) == len(DETERMINISTIC) + 10
    for groups in plan["selection"].values():
        assert all(len(v) == 1 for v in groups.values())
        assert all(len(v) == len(set(v)) for v in groups.values())
    assert plan["overlap"]["local_degree"]["same_selected_components"]
    assert not plan["overlap"]["top_ci"]["same_selected_components"]
    assert prep.fixture(10, 13)["selection"] == plan["selection"]
    assert prep.fixture(10, 13)["graph_sha256"] == plan["graph_sha256"]


def test_local_degree_and_strength_counting():
    score = contact_scores(4, [0, 0, 1], [1, 2, 3], [10., -5., 2.],
                           [.1, .3, .2], .21)
    assert score["in_degree"].tolist() == [0, 1, 0, 1]
    assert score["out_degree"].tolist() == [1, 1, 0, 0]
    assert score["two_sided"].tolist() == [0, 1, 0, 0]
    assert score["total_absolute_attribution"].tolist() == [15, 12, 5, 2]
    with pytest.raises(ValueError):
        contact_scores(4, [0], [7], [1], [.2], .5)
    with pytest.raises(ValueError):
        contact_scores(4, [0], [1], [np.nan], [.2], .5)


def test_ties_are_deterministic_and_no_pool_replacement():
    assert rank_top([8, 4, 7], 2, lambda i: (0,)) == [4, 7]
    with pytest.raises(ValueError):
        rank_top([1, 1], 1, lambda i: (1,))
    with pytest.raises(ValueError):
        rank_top([0, 1], 3, lambda i: (1,))


def test_manifest_validates_deduplication(tmp_path):
    m = prep.fixture(0, 5)
    out = tmp_path / "manifest.json"
    prep.write_manifest(out, m)
    assert json.loads(out.read_text())["graph_sha256"] == m["graph_sha256"]
    with pytest.raises(FileExistsError):
        prep.write_manifest(out, m)
    prep.write_manifest(out, m, overwrite=True)
    with pytest.raises(ValueError):
        select_sets(n_nodes=2, src=[0], dst=[1], weight=[1], entry=[.1],
                    occupation=.2, node_ids=[("m", 0), ("m", 0)],
                    pools={"m": [0]}, backbone={"m": [0]},
                    ci_mean={"m": [1]}, firing_rate={"m": [.2]})


def sample_kl(delta=0.5):
    data = {name: [1 + delta] * 16 for name in DETERMINISTIC}
    data["sieve"] = [1.] * 16
    data["all"] = [.1] * 16
    data["none"] = [5.] * 16
    data["random_000"] = [2.] * 16
    return data


def test_paired_bootstrap_passes_clear_incremental_gain():
    report = score_sequential_kl(sample_kl(), n_bootstrap=500)
    assert report["status"] == "ALL_BASELINES_PASS"
    assert report["quality_gate_pass"]
    assert all(x["passes_practical_gate"] for x in report["comparators"].values())
    assert all(x["extra_gap_ci_95"][0] > 0 for x in report["comparators"].values())


def test_paired_bootstrap_rejects_equivalent_degree_selector():
    d = sample_kl()
    d["local_degree"] = d["sieve"][:]
    report = score_sequential_kl(d, n_bootstrap=500)
    assert report["status"] == "NO_INCREMENTAL_PERCO_ADVANTAGE"
    assert report["comparators"]["local_degree"]["extra_gap_closed"] == 0
    assert report["comparators"]["local_degree"]["exact_same_kl_vectors"]
    assert report["comparators"]["local_degree"]["extra_gap_ci_95"] == [0., 0.]


def test_bad_gap_and_unpaired_data_are_rejected():
    bad = sample_kl()
    bad["all"] = [6] * 16
    assert score_sequential_kl(bad, n_bootstrap=100)["status"] == "INVALID_ABLATION_GAP"
    bad = sample_kl()
    bad["top_ci"] = [1.5]
    with pytest.raises(ValueError):
        score_sequential_kl(bad, n_bootstrap=100)


def test_seed_aggregation_requires_distinct_frozen_sources():
    s = score_sequential_kl(sample_kl(), n_bootstrap=200)
    good = combine_seed_results([
        {"manifest_sha256": "seedA", "score": s},
        {"manifest_sha256": "seedB", "score": s},
    ])
    assert good["status"] == "WITHIN_TARGET_SEED_GATE_PASS"
    with pytest.raises(ValueError):
        combine_seed_results([{"manifest_sha256": "X", "score": s},
                              {"manifest_sha256": "X", "score": s}])


def test_historical_degree_preflight_counts_without_retraining():
    # Assert the E06 saved decimal fractions imply integer counts, including
    # zero local false negatives under a directed source-to-sink definition.
    spec = importlib.util.spec_from_file_location("e10_preflight", ROOT / "experiments" / "e10_preflight.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def doc(n, b, local, c1):
        return {"n_interior": n, "joint_degree": {"cells": {
            "c_s": {"c": .1, "degree_determined": {"backbone": b/n, "local": local/n},
                    "testable": True, "p_high": .07},
            "c_1": {"c": 1., "degree_determined": {"backbone": c1/n, "local": c1/n},
                    "testable": False, "p_high": 1},
        }}}
    x = module.audit_one(doc(1574, 96, 103, 702), "seed1")
    y = module.audit_one(doc(1582, 106, 109, 691), "seed2")
    assert (x["c_s"]["local_false_positives"], y["c_s"]["local_false_positives"]) == (7, 3)
    assert x["c_1"]["local_false_positives"] == y["c_1"]["local_false_positives"] == 0


def test_e10_gate_disambiguates_identical_selector_and_weak_power():
    # Equality of two scored vectors is not, on its own, proof of equal masks.
    d = sample_kl()
    d["local_degree"] = d["sieve"][:]
    equiv = score_sequential_kl(d, n_bootstrap=200,
                                same_selection={"local_degree"})
    assert equiv["decision_class"] == "EXACT_DEGREE_OR_OTHER_SELECTOR_EQUIVALENCE"
    assert equiv["comparators"]["local_degree"]["certified_identical_selector"]
    assert "local_degree" in equiv["certified_equivalent_selectors"]
    with pytest.raises(ValueError, match="different KL"):
        score_sequential_kl(sample_kl(), n_bootstrap=100,
                            same_selection={"local_degree"})

    # Positive mean exceeding the 2% practical threshold with wide uncertainty
    # must remain inconclusive, rather than being claimed as a refutation.
    wiggle = sample_kl()
    wiggle["local_degree"] = [1.15 + (.7 if i % 2 else -.7) for i in range(16)]
    uncertain = score_sequential_kl(wiggle, n_bootstrap=1000)
    assert uncertain["status"] == "NO_INCREMENTAL_PERCO_ADVANTAGE"
    assert uncertain["decision_class"] == "INCONCLUSIVE_GATE_NOT_MET"
    assert uncertain["comparators"]["local_degree"]["extra_gap_ci_95"][0] < 0

    dominated = sample_kl()
    dominated["local_degree"] = [1.0] * 16
    no_effect = score_sequential_kl(dominated, n_bootstrap=200)
    assert no_effect["decision_class"] == "PRACTICAL_ADVANTAGE_EXCLUDED_BY_BOOTSTRAP"
