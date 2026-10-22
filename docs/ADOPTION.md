# PercInterp adoption and integration

PercInterp is a NumPy/SciPy consumer of directed neural influence graphs.
It receives already-computed attribution edges, produces many-threshold
source-to-output connectivity profiles, and provides controls for evaluating
candidate circuit/component sets. It does not train VPD, harvest gradients,
load model weights, or validate mechanistic faithfulness.

## Public contracts

- `from_attribution_edges` accepts signed (source, target, value) rows.
- `from_attribution_arrays` accepts vectorized NumPy or JAX-exported indexed arrays.
- `save_graph_npz` / `load_graph_npz` serialize identity and signed values
  using NumPy with pickle disabled and a string-valued provenance record.
- `profile` computes forward/backward min-max entry keys in O(edges)
  and can evaluate a grid of occupation points from those entries.
- `sieve` selects candidate connected structure relative to shuffle nulls.

The input stages must follow computational order and all edges point forward.
External node IDs can be Goodfire `ComponentKey.encoded()` strings. A
producer should record the model, checkpoint, attribution method, prompt set,
edge aggregation, and original decomposition quality independently.

## Goodfire compatibility

The pinned public Goodfire source at commit
[177d0cc](https://github.com/goodfire-ai/param-decomp/tree/177d0cce809896d9d84b2bd04bbc2e319c64e9d0)
provides a JAX-free `param-decomp-formats` distribution with canonical
`ComponentKey(site, component_idx).encoded()` identifiers. A dedicated
GitHub Actions job installs that package from source on Python 3.12 and verifies
the actual encoding/decoding behavior. Edge weights in this test are
**synthetic**, so this is an identity-format integration check.

The current public repository does not expose a ready-to-import measured
direct-attribution edge graph. A complete measured integration will require
Goodfire-produced edge arrays, source and destination ComponentKeys, stages,
and a record of the attribution conditions. No native VPD exporter is claimed.

## Reference CPU benchmark

`python benchmarks/profile_vs_resweep.py --width 192 --points 41 --repeats 5`

A fixed-seed synthetic five-layer DAG is swept at 41 thresholds. The
PercInterp profile is compared to an independent SciPy reference that creates
CSR adjacency and runs forward/reverse BFS at each threshold. Exact numerical
parity is a prerequisite for reporting times.

On one local Python 3.13.5 CPU run with 578 vertices and 40,708 edges:
profile median 4.46 ms, repeated SciPy BFS median 103.85 ms (23.3x).
This compares one sweep against repeated graph rebuilding, not against an
optimized competing all-threshold solver. It measures runtime, not circuit
quality or Goodfire inference. GitHub CI retains a machine-readable benchmark.

## Research and adoption boundary

Retrospective E02/E09 fidelity tests favor simple retained-edge fraction over
backbone fraction for predicting model behavior. E03 identifies a promising
toy-model component-selection advantage, E04 fails to confirm Qwen transfer,
and E06/E07 expose strong degree effects. Causal E10 evaluation still needs
original saved model/decomposition artifacts. This library reports graph
measurements and generates interventions to validate elsewhere.

The [proposed Goodfire interface](GOODFIRE_PITCH.md) requests one measured
exported attribution graph, rather than making param-decomp depend on
PercInterp before interoperability is demonstrated.
