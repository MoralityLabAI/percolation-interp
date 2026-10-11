# percolation-interp

Percolation theory applied to mechanistic-interpretability influence graphs.
Threshold a network's causal-influence graph, sweep the kept fraction p, and ask
three questions:
- Is circuit fidelity gated by connectivity?
- How do trained graphs differ from shuffled and at-init controls?
- Does circuit formation show up as a connectivity event?

- `docs/plan.md`: the research plan, with predictions written before running
- `docs/literature.md`: prior-art check (October 2026)
- `docs/e01_controls.md`: step 1 results (control pipeline)
- `docs/e02_deep_mlp.md`: step 2 results so far (grokking in deep MLPs; depth 2 done)
- `docs/e03_design.md`: e03, the percolation sieve over a full VPD and an SAE tower (in progress)

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
