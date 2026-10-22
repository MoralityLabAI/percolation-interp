"""Test model-agnostic attribution graph interchange and provenance."""

import numpy as np
import pytest

from percinterp.interop import from_attribution_edges
from percinterp.measure import profile, rank_entry


def test_goodfire_style_component_identifiers_preserve_signs_and_mapping():
    result = from_attribution_edges(
        node_ids=["output:logit", "layer0.gate_proj:12", "token:0", "layer0.up_proj:8"],
        layers=[2, 1, 0, 1],
        edges=[
            ("token:0", "layer0.gate_proj:12", 2.0),
            ("token:0", "layer0.up_proj:8", -1.0),
            ("layer0.gate_proj:12", "output:logit", -0.75),
            ("layer0.up_proj:8", "output:logit", 0.25),
        ],
    )
    assert result.node_ids == ("token:0", "layer0.gate_proj:12", "layer0.up_proj:8", "output:logit")
    assert result.graph.layer_sizes == [1, 2, 1]
    assert np.allclose(result.graph.weight, [2.0, 1.0, 0.75, 0.25])
    assert np.allclose(result.signed_weight, [2.0, -1.0, -0.75, 0.25])
    assert result.signed_edges()[1] == ("token:0", "layer0.up_proj:8", -1.0)
    entry = rank_entry(result.graph.weight, result.graph.edge_layer("dst"))
    path = profile(result.graph, entry)
    assert 0 < path.p_span <= 1
    assert result.selected_ids(path.node_entry <= 1) == result.node_ids


def test_designated_input_output_and_residual_skip():
    result = from_attribution_edges(
        ["input", "unused", "hidden", "sink", "other"],
        [0, 0, 1, 2, 2],
        [("input", "hidden", 0.5), ("hidden", "sink", 1.0),
         ("input", "sink", -0.125), ("unused", "other", 0)],
        sources=["input"], sinks=["sink"],
    )
    assert result.dropped_zero_edges == 1
    assert result.graph.n_edges == 3
    assert result.graph.sources.sum() == 1
    assert result.graph.sinks.sum() == 1
    assert np.array_equal(result.graph.node_layer[result.graph.dst], [1, 2, 2])
    assert result.selected_ids(np.array([True, False, False, False, True])) == ("input", "other")


@pytest.mark.parametrize("ids,layers,edges", [
    ([], [], []),
    (["a", "a"], [0, 1], []),
    (["a", "b"], [0], []),
    (["a", "b"], [0, 2], []),
    (["a", "b"], [0, True], []),
    (["a", "b"], [0, 1], [("b", "a", 1)]),
    (["a", "b"], [0, 1], [("a", "b", 1), ("a", "b", 2)]),
    (["a", "b"], [0, 1], [("a", "b", float("nan"))]),
    (["a", "b"], [0, 1], [("a", "missing", 1)]),
    (["a", "b"], [0, 1], [("a", "b")]),
])
def test_invalid_edge_contract_rejected(ids, layers, edges):
    with pytest.raises(ValueError):
        from_attribution_edges(ids, layers, edges)


def test_reject_duplicate_boundaries_and_bad_masks():
    with pytest.raises(ValueError, match="duplicate"):
        from_attribution_edges(["a", "b"], [0, 1], [("a", "b", 1)], sources=["a", "a"])
    with pytest.raises(ValueError, match="unknown"):
        from_attribution_edges(["a", "b"], [0, 1], [("a", "b", 1)], sinks=["oops"])
    data = from_attribution_edges(["a", "b"], [0, 1], [("a", "b", 1)])
    with pytest.raises(ValueError, match="boolean"):
        data.selected_ids([1, 0])
    with pytest.raises(ValueError, match="boolean"):
        data.selected_ids([True])


def test_all_zero_edges_survive_import_as_empty_graph():
    data = from_attribution_edges(["a", "b"], [0, 1], [("a", "b", -0.0)])
    assert data.graph.n_edges == 0
    assert data.dropped_zero_edges == 1
    assert data.signed_edges() == ()
