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
