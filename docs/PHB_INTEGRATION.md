# PercInterp × Persona Holonomy Bench

**Cross-repository integration:** [PHB v0.12](https://github.com/patrickdugan/Persona-Holonmy-Bench/blob/main/docs/v012-direct-attribution.md) adds a training-only signed single-feature intervention exporter to the v0.8–v0.11 bridge (2026-10-22).

PHB now calls this library directly through three commands. These are two distinct scientific interfaces:

| PHB command | PercInterp surface | Input and interpretation |
| --- | --- | --- |
| `phb perc-select` | `from_attribution_edges`, `profile`, `rank_entry`, `sieve` | Genuine, signed, **direct** influence edges become a directed, topological attribution DAG. The sieve selects interior components; local-degree and budget-matched random selectors provide graph-only controls. |
| `phb perc-audit` | Frozen selection manifest and PHB held-out causal sweep | The bridge audits **decoder-block-level** effects of matched candidate-vs-orthogonal rank-one interventions, with seed/probe pairing and sign-flip tests. It is a coarse screening gate, not causal evidence for individual graph components. |
| `phb perc-holonomy` | `frame_edge_transports`, `analyze_connection` | Matched last-prefix-token activation vectors define PCA frames at (conversation condition, phase) states. Polar Procrustes links define an explicit discrete connection. PercInterp computes cycle transport, certifies minimum singular overlap, and reports gauge-invariant loop spectra. |

## Graph-side cross-implementation

Install the upstream library using Python 3.11+:

```bash
pip install 'git+https://github.com/MoralityLabAI/percolation-interp'
```

Within [PHB](https://github.com/patrickdugan/Persona-Holonmy-Bench):

```bash
phb perc-select --graph examples/percinterp/direct-attribution-SYNTHETIC.json \
  --shuffles 32 --random-controls 12 --output runs/perc-selection.json
```

That example is **synthetic** and verifies the interchange contract. For a cached Hugging Face checkpoint with trained, SHA-256-pinned TopK SAE dictionaries, PHB v0.12 can now estimate a bounded training-only influence graph directly:

```bash
phb sae-attribution-plan --input runs/seeded-cases.jsonl \
  --registry runs/qwen-saes/sae-registry.json \
  --model /path/to/local/HF-Qwen --top-features 4
phb sae-attribution --input runs/seeded-cases.jsonl \
  --registry runs/qwen-saes/sae-registry.json \
  --model /path/to/local/HF-Qwen --top-features 4 \
  --device cuda --dtype float16 --output runs/attribution-graph.json
phb perc-select --graph runs/attribution-graph.json \
  --output runs/perc-selection.json
```

The v0.12 estimator removes the donor-minus-reference contribution of one TopK SAE feature at the final prompt token and records the signed response in later sampled SAE features or first-token style log-odds. This is a **contextual finite-difference intervention** over sampled blocks. It is neither gradient EAP nor an additive parameter-level causal decomposition. Training seeds and probe questions are disjoint from PHB's subsequent held-out evaluations. Source-stage SAE features form the graph boundary; the sieve selects interior features. Graphs with no measured source-to-output path are reported as null/disconnected, without fabricating edges.

For independent VPD/Goodfire use, the caller must export valid direct signed influence edges with stable identifiers such as `ComponentKey.encoded()`, consecutive topological stages, and optionally explicit decoder-block mappings. Cosine distances between PHB hidden states are **not** direct attribution edges.

The selection manifest freezes graph fingerprint, selected IDs, source-to-sink connectivity threshold, null distribution and matched comparator budgets *before* seeing held-out PHB causal scores. When the sieve's nominal connectivity peak contains no interior backbone, the adapter records its fallback to the earliest spanning threshold.

## Geometric crossover

For exploratory frame holonomy from a real cached chat model:

```bash
phb generate --variation seeded_v1 --repetitions 12 --seed 100 \
  --conditions ordered,tail_matched --phases peak,exit,washout_2 \
  --probes flaky_ci,parser_validator --output runs/perc-frame-cases.jsonl
phb capture --model /path/to/local/checkpoint \
  --input runs/perc-frame-cases.jsonl --layers 8 \
  --output runs/perc-frame-activations.jsonl
phb perc-holonomy --activations runs/perc-frame-activations.jsonl \
  --layer hs_8 --phases peak,exit,washout_2 --rank 2 --null-shuffles 99 \
  --output runs/perc-holonomy.json
```

The adapter defines its state graph, PCA-frame construction, polar transport, overlap certification and permutation null explicitly. It checks loop spectra after independently rotating every local frame (gauge invariance). The statistic measures this **defined empirical frame connection**; it is not evidence by itself that a model maintains a persistent internal persona or that its computation implements that connection.

The `seeded_v1` baseline is fixed across seeds for a given probe, making baseline PCA potentially rank deficient; use `peak` and later seeded phases for the initial run.

## Scientific gate and open work

- First test whether the percolation sieve makes candidate selections *different from, and predictively superior to*, local-degree and total-attribution selectors under identical budgets. PercInterp E09/E10 already highlights degree and coverage confounds.
- Freeze all graph-only selections on training data and evaluate on disjoint PHB seed/probe partitions. The block audit is a filter for where to run more expensive **per-component** causal ablations.
- Compare frame-holonomy statistics to within-phase condition-label permutation nulls, then replicate on independent seeds and another model.
- Report null results and identity-equivalent selectors. A convincing cross-implementation requires both reliable direct-attribution edges and held-out, component-level causal effects.

**Status (2026-10-22):** PHB v0.12 is committed; [CI run 39810439178](https://github.com/patrickdugan/Persona-Holonmy-Bench/actions/runs/39810439178) passed all four jobs, including the pinned real-PercInterp adapter, signed-export smoke test, and CPU tensor hooks. No pretrained Qwen causal attribution, robust percolation advantage, or model-internal holonomy has been established.
