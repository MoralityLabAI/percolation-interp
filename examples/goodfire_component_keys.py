"""Real Goodfire component identifiers with synthetic attribution values.

Requires the separate param-decomp-formats package (no JAX trainer needed).
See docs/ADOPTION.md. The edges here are illustrative, not model measured.
"""
import numpy as np
from pathlib import Path
from param_decomp_formats.component_key import ComponentKey
from percinterp import from_attribution_arrays, rank_entry, profile
from percinterp.archive import save_graph_npz, load_graph_npz

a = ComponentKey(site="layers.0.gate_proj", component_idx=12)
b = ComponentKey(site="layers.0.up_proj", component_idx=8)
assert ComponentKey.decode(a.encoded()) == a
data = from_attribution_arrays(
    ["metric:output", b.encoded(), "token:input", a.encoded()],
    [2, 1, 0, 1],
    np.asarray([2, 2, 3, 1], dtype=np.int64),
    np.asarray([3, 1, 0, 0], dtype=np.int64),
    np.asarray([.9, -.8, .7, -.25], dtype=float),
)
entry = rank_entry(data.graph.weight, data.graph.edge_layer("dst"))
print("Goodfire ComponentKey names:", data.node_ids)
print("Synthetic earliest surviving path:", profile(data.graph, entry).p_span)
path = Path("goodfire_key_demo.npz")
try:
    save_graph_npz(path, data, provenance={
        "producer": "goodfire native component identifiers and synthetic edge fixture",
        "attribution_method": "SYNTHETIC_NO_MODEL",
    })
    restored, provenance = load_graph_npz(path)
    assert restored.signed_edges() == data.signed_edges()
    assert provenance["attribution_method"] == "SYNTHETIC_NO_MODEL"
finally:
    path.unlink(missing_ok=True)
