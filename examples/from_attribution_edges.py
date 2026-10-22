"""Minimal CPU-only example of PercInterp using imported attribution edges.

Run: python examples/from_attribution_edges.py

This is a synthetic example of the edge interchange contract. For actual
Goodfire VPD results, export validated *direct* attribution edges with stable
ComponentKey.encoded() identifiers, and construct stages from causal order.
"""

from percinterp.interop import from_attribution_edges
from percinterp.measure import profile, rank_entry


imported = from_attribution_edges(
    node_ids=["token:0", "gate.0:12", "up.0:8", "output:logit"],
    layers=[0, 1, 1, 2],
    edges=[
        ("token:0", "gate.0:12", 0.90),
        ("token:0", "up.0:8", -0.80),
        ("gate.0:12", "output:logit", 0.70),
        ("up.0:8", "output:logit", -0.25),
    ],
)

graph = imported.graph
occupation = rank_entry(graph.weight, graph.edge_layer("dst"))
result = profile(graph, occupation)
print("Earliest input-output path at retained fraction:", result.p_span)
print("Backbone node IDs at p=1:", imported.selected_ids(result.node_entry <= 1.0))
print("Original signed weights:", imported.signed_weight.tolist())
