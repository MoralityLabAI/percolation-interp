# Goodfire: optional attribution-graph percolation utility

PercInterp is a small NumPy/SciPy package that consumes a validated *direct*
attribution edge graph downstream of parameter decomposition. It computes
directed source-to-output bottleneck/backbone profiles across many thresholds
and compares them against graph nulls.

The integration boundary is after Goodfire generates edge values, leaving
causal importance training, post-hoc adversarial masking, and faithfulness
evaluation in Goodfire's native code. PercInterp does not assume private
Goodfire checkpoint layouts or require its JAX training dependencies.

Goodfire's public `param-decomp-formats` distribution defines a canonical
`ComponentKey.encoded()` identity. PercInterp accepts these labels together
with causal-order stage numbers and directed attribution edges. Its CI test
uses the actual upstream class, pinned to commit 177d0cc, on synthetic edges.

On a fixed-seed synthetic 40,708-edge graph, one-pass profiles match SciPy
BFS at 41 thresholds and run faster than rebuilding adjacency per threshold.
This is an algorithmic reference rather than a claim to faster Goodfire
attribution generation or improved circuit-finding quality.

The first useful collaboration would be a single shareable VPD
direct-attribution edge export, with the source run/checkpoint ID, metric,
prompt set, stage order, aggregation rule and decomposition quality record.
That would enable a real end-to-end integration and matched benchmark, followed
by a small downstream example. Goodfire does not need to adopt PercInterp as
a dependency before the demonstration has been reviewed.

See [Goodfire's parameter-decomposition handbook](https://github.com/goodfire-ai/param-decomp/blob/main/docs/handbook.md)
and [PercInterp's integration guide](GOODFIRE_INTEGRATION.md).


## Goodfire's June 2026 architecture decision

Goodfire's completed transition issues
[#841](https://github.com/goodfire-ai/param-decomp/issues/841) and
[#842](https://github.com/goodfire-ai/param-decomp/issues/842)
explicitly removed the live attribution/intervention backend and prompt-attribution
frontend while migrating the application toward read-only inspection of JAX runs.
Issue #841 distinguished these removed live-compute endpoints from precomputed
attribution and investigation readers. The current public source has since been
refactored further.

That is an important integration constraint. A PercInterp proposal should
**not** ask Goodfire to reinstate those live endpoints or take on another
graph-visualization dependency. The small seam is an optional downstream export
of direct-attribution edges in Goodfire's stable component identity scheme.
PercInterp can then compute graph profiles in a separate analysis process.
This remains compatible with Goodfire's documented evidence standard that
attribution graphs are not themselves faithful computational circuits.
