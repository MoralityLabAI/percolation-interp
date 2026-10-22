# percolation-interp

Percolation theory applied to mechanistic-interpretability influence graphs.
Threshold a network's causal-influence graph, sweep the kept fraction p, and ask
three questions:
- Is circuit fidelity gated by connectivity?
- How do trained graphs differ from shuffled and at-init controls?
- Does circuit formation show up as a connectivity event?

- `docs/HANDOFF.md`: live processes and state as of 2026-10-21 (read first when resuming)
- `docs/plan.md`: the research plan, with predictions written before running
- `docs/literature.md`: prior-art check (October 2026)
- `docs/e01_controls.md`: step 1 results (control pipeline)
- `docs/e02_deep_mlp.md`: step 2 results (grokking in deep MLPs, depths 2, 4 and 8)
- `docs/e03_design.md`: e03 design, the percolation sieve over a full VPD and an SAE tower
- `docs/e03_results.md`: e03 results (P1 passes for VPD and fails for the SAE graph; P2 fails; P3 passes)
- `docs/e04_design.md`: e04 design, the sieve on a VPD of Qwen3-0.6B layers 2-3
- `docs/e04_results.md`: e04 results, two seeds (Q1 fails in both; the zero-ablation baseline is ~94% the position-0 massive activation)
- `docs/e05_design.md`: e05 design, the sieve on Qwen3-0.6B with position-aware mean-ablation baselines
- `docs/e05b_design.md`: e05b design, e05 with a 16x larger decomposition batch on a rented GPU
- `docs/e06_exploratory.md`: e06 discovery analysis (coverage/alignment decomposition, nulls) on e03/e04 graphs
- `docs/e06_design.md`: e06 design, pre-registered null-model study of the sieve z-curve

## Install

The package is `percinterp`. The core (`graph`, `measure`, `theory`, `controls`, `sieve`)
needs only numpy and scipy:

```bash
pip install "git+https://github.com/MoralityLabAI/percolation-interp"
```

Optional extras:

- `[torch]`: `dense`, `modadd` and `sae`
- `[plot]`: figure style for the experiment scripts
- `[dev]`: tests

```bash
pip install "percinterp[torch,plot] @ git+https://github.com/MoralityLabAI/percolation-interp"
```

Minimal use: rank a weighted layered graph's edges within layers, then sieve for
connected structure against within-layer shuffles:

```python
from percinterp import LayeredDAG
from percinterp.sieve import sieve

g = LayeredDAG.from_dense([W0, W1])  # (out, in) weight or attribution matrices per layer
res = sieve(g, g.weight)
res.z_peak, res.p_value, res.backbone  # peak z, p against shuffles, sieved nodes
```

## Development setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
.venv/Scripts/python experiments/e01_controls.py
```

`e01_controls.py --quick` is a two-second smoke run. The full run takes about a
minute on CPU.

## Code

| Module | Contents |
|---|---|
| `graph.py` | `LayeredDAG`: nodes numbered by layer, edges from earlier to later layers (skip edges allowed) |
| `measure.py` | entry keys (`rank_entry`), forward and backward min-max passes, `Profile` (spanning point, backbone, reachability at any p), undirected `components` |
| `theory.py` | exact Markov chain for Bernoulli percolation on complete layered graphs, mean field, path-count threshold |
| `controls.py` | complete layered graphs, within-group shuffles, MLP influence graphs (\|W\|, direct effect, EAP) |
| `sieve.py` | the percolation sieve: degree-unit ranks, learned connectivity z-scored against shuffles (within-layer or strength-preserving), backbone at c* |
| `dense.py` | batched torch versions of the measurements for adjacent-layer graphs (GPU) |
| `modadd.py` | step 2: modular-addition MLPs, mean-ablation edges with EAP-IG scores, masked subgraph fidelity |
| `sae.py` | TopK sparse autoencoders (with AuxK) for an SAE tower |
| `plotstyle.py` | shared figure style |

How the measurement works:
- Each edge gets an entry key and is present at occupation p when key ≤ p.
- One forward pass computes, for every node, the smallest p at which an input
  reaches it. One backward pass computes the smallest p at which it reaches an
  output.
- Together these give the spanning point (the widest-path bottleneck) and the
  backbone membership of every node and edge, for every p at once, in O(E).
- Tests check all of this against brute-force search and against the exact chain.

## E08 globe / browser staging

A CPU-only holonomy-percolation calibration with a static interactive Gödel Globe.
The three six-node fixtures are **synthetic** (planted, flat, tangent-frame).
The site's separate Qwen E07 table reports observed contact-degree crossovers;
the globe's frames and loops are not measured from Qwen.

- [Interactive site (after GitHub Pages enablement)](https://moralitylabai.github.io/percolation-interp/)
- [Experiment and provenance](docs/e08_site.md)
- [Build and publishing workflow](.github/workflows/percinterp-pages.yml)

To enable public hosting once: **Settings → Pages → Source → GitHub Actions**.
Run locally with `python experiments/e08_build_site.py --output out/e08_site`.
