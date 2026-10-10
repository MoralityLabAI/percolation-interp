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
2. **Modular addition with grokking, in deep MLPs** (in progress; design
   below). This merges the old depth-sweep step into step 2.
3. **Nanda's one-layer transformer.** A check of H1's frequency-module
   prediction only (see the step 2 design for why it is not the main testbed).
4. **Language model.** GPT-2 small on IOI, Greater-Than and Docstring with
   EAP-IG (github.com/hannamw/EAP-IG). Compare against MIB's CPR/CMD curves,
   which are the overlap risk.

## Step 2 design

Written 2026-10-10, before any step 2 training results.

### Why not Nanda's transformer as the main testbed

- At head and neuron grain its influence graph is: 2 token inputs → 4 heads →
  d_mlp neurons → 1 output.
- Under the null, the fraction of neurons on the backbone is roughly
  p(1 − (1 − p)^4). That is a smooth polynomial with no transition to find.
- The spanning point is set by a single 2-3 edge path.
- Deep MLPs grok the same task and give layered graphs where e01's machinery
  and theory apply. Depth becomes a sweep variable.

### Revisions before any analysis of trained models (2026-10-10)

1. **Graph.** The first version had 2P one-hot input units and P logit nodes
   (with the answer's logit as the sink). A smoke run at initialisation showed
   the ranked graph spanning with 1/23 of the edges its shuffle needed. The
   cause is trivial structure, not anything the network computes:
   - the 2 active input units score about P times the 2P − 2 inactive ones;
   - edges into the answer logit score about P times the others.
   
   A within-layer shuffle destroys both. The graph is now: 2 token nodes (a and
   b, each mean-ablated as a block) → D layers of N → one sink (the metric).
2. **Occupation axis.** With a width-1 sink layer, ranking at the same kept
   fraction p in every layer pinned the spanning point at p = 1/N for both graph
   and shuffle. Occupation is now c, the mean kept out-degree per live source
   node, per layer. This is e01's axis (c = pN for complete N → N layers), so the
   mean-field threshold is c = 1. The reference occupation is c_ref = 2, which
   replaces p_ref = 2/N below. "Nominal p" in H1b now means nominal c.
3. **Steps.** 40k instead of 20k. On CPU, depth 2 width 128 reached 95% test
   accuracy at step 18.3k of 20k. That is too close to the end, and deeper
   networks are likely slower. The decision used the training curve only.

### Status log

- **2026-10-11, depth 2 complete.** All 9 depth-2 runs grokked (widths 128, 256
  and 512; 3 seeds each). Steps to 95% test accuracy ranged from 10.4k to 36.4k.
- **Depth 4 stopped.** Depth 4, width 128, seed 0 did not grok in 40k steps:
  - final test accuracy 0.008;
  - test loss rose to about 87;
  - weight norm flat at about 43, with recurring loss spikes.
  
  I stopped the grid there rather than spend about 2 GPU-hours on a shared
  card on depth-4 and depth-8 runs likely to be excluded.
- **Probes queued.** Depth 4 and depth 8 at width 512, seed 0, with the same
  hyperparameters. Width 512 grokked fastest at depth 2.
- **If the probes fail:** the depth arm needs different hyperparameters. Any
  change will be chosen on accuracy curves only, before analysing those graphs,
  and recorded here.

### Setup

- **Task and training.** P = 113, one-hot (a, b) input, D hidden ReLU layers of
  width N, P logits. AdamW (lr 1e-3, weight decay 1.0, β2 0.98), full batch, 30%
  of pairs for training, 40k steps (see revision 3).
- **Grid.** D ∈ {2, 4, 8}, N ∈ {128, 256, 512}, seeds 0-2. Configurations that
  do not grok are reported as such and excluded from H1-H3.
- **Graph per input.** 2P inputs → D layers of N → one output node.
  - Edges: mean-ablation edges with EAP-IG scores (8 steps; `percinterp.modadd`).
  - Ranking: within each layer; zero scores are not edges.
- **Ensemble.** A fixed sample of 256 test-split inputs, at every checkpoint.
- **Controls.**
  - Within-layer shuffle on each input's own skeleton (`dense.shuffled_entry`).
  - The step-0 checkpoint.
  - Absolute thresholds: per layer, scores ranked against the final
    checkpoint's pooled scores (`dense.reference_entry`).
- **Reference occupation.** p_ref = 2/N, i.e. about 2 kept out-edges per
  hidden node. That is above the mean-field threshold c = 1.
- **Fidelity.** Logit-metric recovery (logit of the answer minus the mean
  logit), accuracy and loss of the per-input kept subgraph, under mean
  ablation.

### Pass/fail criteria

**H1a: dangling edges are inert.**
- At the final checkpoint, keeping only the backbone's edges instead of every
  kept edge changes mean recovery by less than 0.05.
- **Fails if:** the change exceeds 0.05 at any p where recovery is between 0.2
  and 0.9.

**H1b: fidelity collapses on connectivity.**
- Across grokked (D, N) configurations, take the point where mean recovery
  crosses 0.5. Locate it on two axes: the backbone edge fraction (effective
  sparsity) and nominal p.
- The spread of that point is smaller on the backbone axis, measured as the
  coefficient of variation across configurations.
- **Fails if:** the coefficient of variation on the backbone axis is ≥ the one on
  p.

**H2: the init gap is not the whole story.**
- For each grokked run, compute
  Δ = log(median p*_final / median p*_final,shuf) − log(median p*_init / median p*_init,shuf).
- Prediction: Δ < 0 (training lets the graph span with fewer edges than its
  shuffle, relative to init).
- **Fails if:** Δ ≥ 0 or |Δ| < the seed-to-seed standard deviation of Δ.

**H3: connectivity over training.**
- Definitions:
  - t_mem: first step with train accuracy ≥ 0.99.
  - t_grok: first step with test accuracy ≥ 0.95.
  - Tracked quantity: backbone node fraction at p_ref over checkpoints, minus
    the shuffled control's value (the "learned connectivity").
- **H3-rank:** under rank thresholds, the steepest change (per log-step)
  falls between t_mem and 1.5 × t_grok.
- **H3-abs:** under absolute thresholds, the curve after t_mem tracks the weight
  norm (Spearman ρ > 0.8). That would mean absolute thresholds are dominated by
  the Omnigrok norm confound.
- **Reading the result:**
  - H3-rank holds with its change concentrated at t_grok rather than spread
    over (t_mem, t_grok): "connectivity event = cleanup seen through the
    ranking".
  - The change is spread over (t_mem, t_grok): connectivity tracks gradual
    formation.

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
