"""Contract check against real Goodfire param_decomp_formats (optional import)."""
import numpy as np
import pytest

from percinterp import from_attribution_arrays, profile, rank_entry


def test_goodfire_component_key_encoding_and_import():
    module = pytest.importorskip("param_decomp_formats.component_key")
    ComponentKey = module.ComponentKey
    a = ComponentKey(site="layers.0.gate_proj", component_idx=12)
    b = ComponentKey(site="layers.0.up_proj", component_idx=8)
    assert a.encoded() == "layers.0.gate_proj:12"
    assert ComponentKey.decode(a.encoded()) == a
    graph = from_attribution_arrays(
        ["metric:output", b.encoded(), "token:input", a.encoded()],
        [2, 1, 0, 1],
        np.asarray([2, 2, 3, 1], dtype=np.int64),
        np.asarray([3, 1, 0, 0], dtype=np.int64),
        np.asarray([.9, -.8, .7, -.25], dtype=np.float64),
    )
    assert a.encoded() in graph.node_ids and b.encoded() in graph.node_ids
    entry = rank_entry(graph.graph.weight, graph.graph.edge_layer("dst"))
    assert 0 <= profile(graph.graph, entry).p_span <= 1
