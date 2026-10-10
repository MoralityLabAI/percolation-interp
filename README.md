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

## Setup

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
| `plotstyle.py` | shared figure style |

How the measurement works:
- Each edge gets an entry key and is present at occupation p when key ≤ p.
- One forward pass computes, for every node, the smallest p at which an input
  reaches it. One backward pass computes the smallest p at which it reaches an
  output.
- Together these give the spanning point (the widest-path bottleneck) and the
  backbone membership of every node and edge, for every p at once, in O(E).
- Tests check all of this against brute-force search and against the exact chain.
