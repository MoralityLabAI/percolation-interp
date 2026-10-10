# Research plan

## Question

Circuit discovery thresholds a graph of causal influences and keeps what
survives. That is a percolation process, but nobody has analysed it as one (see
`literature.md`). The project asks:
- Is circuit fidelity gated by connectivity?
- How does a trained network's influence graph differ from plain percolation on
  the same skeleton?
- Does circuit formation during training show up as a connectivity event?

## The object

A weighted directed acyclic graph for one task and one input (prompt):

- **Nodes:** units at a chosen grain (neurons, attention heads, dictionary
  features), plus input and output nodes.
- **Edges:** causal influence on the task metric, from attribution (EAP or
  EAP-IG) or path patching. Use the influence graph, not the wiring: residual
  connections make the wiring connected at every p.
- **Control parameter:** keep the top fraction p of edges within each layer,
  then sweep p. Rank within layers, because a global threshold is biased by
  per-layer scale (e01 B3: 1.76×). As a second sweep, use an absolute threshold
  τ across training checkpoints.

## Measurements

All come from `percinterp.measure`. Two min-max passes give every directed
curve at every p.

| Observable | What it is | Note |
|---|---|---|
| spanning point p* | smallest p with an input-output path | Equals the bottleneck (widest path). One number per graph; a probability needs an ensemble, i.e. prompts. |
| backbone fraction | nodes (edges) on a surviving input-output path | The circuit analogue. Edge version = effective sparsity. |
| reachable fraction | nodes reachable from inputs, per layer | Has a real transition at affordable depth (e01 A2). |
| largest cluster, susceptibility | undirected cluster statistics | Separates from spanning on attribution graphs (e01 B4). |
| fidelity | task metric with only the kept subgraph | Report accuracy and loss/logit difference separately (H1). Fix the ablation type ([Miller et al.]). |

## Controls

1. **Shuffled skeleton.** Permute weights within each layer on the same edge
   set. Ranking a shuffled graph is exactly bond percolation on that skeleton,
   so this is the null.
2. **Same network at initialisation.** At init, EAP graphs already span 18-51%
   later than their shuffle (e01 B1). Training is credited only with the change
   relative to init.
3. **Exact theory** for complete layered graphs (`theory.reach_chain`).
4. **Per-prompt graphs** form the ensemble. The input-averaged graph is a
   different object: it looks about twice as well connected (e01 B2).

## Hypotheses and predictions (written before running)

**H1: fidelity is gated by connectivity.**
- *Modular addition prediction:* the circuit is a few parallel
  frequency modules, and since p is prime, one frequency alone already puts the
  argmax on the right answer.
- So accuracy should hold until the last module leaves the backbone, while loss
  worsens a step at a time as modules drop out.
- **Fails if:** accuracy falls while a module is still on the backbone, or
  survives after the backbone is empty.

**H2: training moves the transition beyond what init explains.**
- Measure Δ = log(p*_trained / p*_shuffled) − log(p*_init / p*_shuffled-init),
  and the same for the backbone half-point.
- Expect both trained and shuffled graphs to show mean-field exponents (dense
  layered graphs are effectively infinite-dimensional). So the claim is a shift
  in threshold and curve shape, not a new universality class.
- **Fails if:** Δ is within seed-to-seed spread.

**H3: circuit formation is a connectivity event.**
- Across checkpoints of a grokking run, compare two thresholds:
  - *absolute τ:* the task component should grow during circuit formation,
    before the test-loss drop. Weight-norm shrinkage in cleanup (Omnigrok) works
    against it there.
  - *rank p:* removing memorising edges in cleanup pushes circuit edges up the
    ranking, so the component may jump at cleanup from ranking alone.
- **Reading the result:**
  - Only the rank version lines up with the test-loss drop: the "event" is
    cleanup seen through the ranking.
  - The absolute version rises during formation: connectivity tracks
    formation, as Nanda et al.'s progress measures do.

## Order of work

1. **Control pipeline.** Done: e01, `docs/e01_controls.md`. The code recovers
   exact answers, and init effects are measured.
2. **Modular addition with grokking** (next).
   - Setup: Nanda et al.'s one-layer setup, p = 113.
   - Grain: inputs → attention heads → MLP neurons → logits.
   - Edges: EAP-IG per input pair; all p² pairs form the ensemble.
   - Checkpoints across memorisation, formation and cleanup.
   - Seeds and several widths of d_mlp. Run H1-H3 against the predictions
     above.
   - Depth is about 3 node layers, so use undirected and backbone observables
     there, not directed-percolation scaling.
   - Needs torch: the RTX 5080 laptop GPU is enough.
3. **Depth sweep (optional).** Only if a directed-percolation claim is wanted:
   deep MLPs or a multi-layer transformer on a compositional task. Here A1's
   ln N / L control applies.
4. **Language model.** GPT-2 small on IOI, Greater-Than and Docstring with
   EAP-IG (github.com/hannamw/EAP-IG). Compare against MIB's CPR/CMD curves,
   which are the overlap risk.

## Pitfalls

- **Sign.** Percolation ignores it, but circuits use inhibition and
  cancellation. Run on |w|, and also on the positive and negative parts
  separately.
- **Selection rule.** Ranking by weight is correlated with topology. That
  correlation is the signal, so always compare against the shuffled skeleton.
- **Finite size.** A sharp curve at one size proves nothing. Scale width at
  several depths, or use reachable/backbone fraction.
- **Weight norm across checkpoints.** Absolute thresholds track the norm. Run
  both sweeps (H3).
- **Ablation choice.** Fidelity curves move with resample versus mean ablation
  and with node versus edge ablation. Fix one and report sensitivity.
