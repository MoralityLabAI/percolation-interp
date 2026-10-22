# Using PercInterp with Goodfire parameter decomposition

## Scope

PercInterp accepts a **directed weighted influence graph** and computes graph
percolation, source-to-output backbone profiles and null comparisons. It is a
separate, NumPy/SciPy-only consumer of analysis data. It neither trains a VPD
decomposition nor validates its mechanistic faithfulness.

Goodfire's [`param-decomp`](https://github.com/goodfire-ai/param-decomp)
currently uses a JAX-based training environment; its
[`param_decomp_formats`](https://github.com/goodfire-ai/param-decomp/tree/main/param_decomp_formats)
package provides stable component identifiers and lightweight data readers.
PercInterp does **not** depend on either package and does not assume their
private checkpoint layouts. No direct Goodfire graph-harvesting adapter is
claimed here. The bridge begins after a producer has exported edges.

## Input contract

A producer provides:

- A unique string for each graph vertex, such as
  `ComponentKey(site="some_site", component_idx=12).encoded()` for a
  parameter subcomponent, or a namespaced token / output-metric identifier.
- An integer *topological stage* per vertex (0, 1, 2, ...). Distinguish
  attention and MLP operations within a transformer layer as needed; edge
  direction must respect computational order, including any skip edges.
- Exactly one signed **direct attribution** per ordered source-target pair.
  Aggregate per-token observations upstream using a stated statistic and
  recorded distribution. Duplicate rows are rejected rather than silently
  summed. Use positions and input contexts consistently.
- Optionally, a designated input or metric-output subset.

```python
from percinterp.interop import from_attribution_edges
from percinterp.measure import rank_entry, profile

imported = from_attribution_edges(
    node_ids=["token:0", "block0.gate:12", "output:logit"],
    layers=[0, 1, 2],
    edges=[("token:0", "block0.gate:12", 0.7),
           ("block0.gate:12", "output:logit", -0.4)],
)
entry = rank_entry(imported.graph.weight, imported.graph.edge_layer("dst"))
p = profile(imported.graph, entry)
print(p.p_span, imported.selected_ids(p.node_entry <= 1))
```

The signed edge values remain available as `imported.signed_weight`; the
`LayeredDAG` graph's weights are magnitudes for percolation. This explicit
split prevents negative direct attribution from being silently discarded by
sieve routines that expect nonnegative edge strengths.

## Evidence boundary

1. Validate the originating decomposition's parameter faithfulness,
   importance minimality and adversarially tested mechanistic faithfulness
   using its native Goodfire evaluation machinery before making claims.
2. Export *direct* rather than indirect influence estimates. Retain the edge
   statistic, sampling distribution, source run, target, checkpoint and
   masking policy in surrounding metadata. Graph edges are not functional
   equations and neither connectivity nor percolation alone proves circuit
   necessity or sufficiency.
3. Compare percolation profiles against within-layer shuffles and degree-aware
   controls. The E06/E07 studies found that much early connectivity is
   explained by local contact degrees; see [E09](e09_utility_decision.md).
4. Treat percolation-selected component masks as hypotheses to evaluate with
   held-out interventions; when available use adversarial masking as Goodfire's
   handbook requires for faithful post-hoc pruning.

Goodfire's detailed [evidence handbook](https://github.com/goodfire-ai/param-decomp/blob/main/docs/handbook.md)
is the standard for the causal claims; the PercInterp API reports graph
quantities and their null statistics.

## Why this boundary helps adoption

The library can consume graph data from Goodfire's JAX code, older PyTorch
parameter decompositions, SAEs, EAP-IG or handcrafted causal DAGs. The
import path runs on CPU and requires only NumPy and SciPy, so it can sit in
a lightweight analysis process or a CI job instead of acquiring the model's
full training dependencies. A native exporter for a documented Goodfire
attribution artifact can be added once that artifact contract is stable.
