# e04: the percolation sieve on a real language model (Qwen3-0.6B slice)

Written 2026-10-14, before the e04 decomposition was trained. Only two 60-step
smoke tests had run, to measure speed and memory.

## Goal

e03 found, on a 1.4M-parameter toy and across three decompositions, two
things:

- the percolation sieve finds significant connected structure in VPD's
  component graph;
- keeping only that structure preserves far more of the model's behaviour
  than size- and frequency-matched random components, and more than the same
  number of individually most important components.

e04 tests whether those two results transfer to a real pretrained model.

## Setup

- **Model.** Qwen/Qwen3-0.6B from HuggingFace: 28 layers, d = 1024, GQA
  attention, SwiGLU MLP.
- **VPD slice.** Layers 2 and 3 (revised from 13 and 14; see below), all seven projections each (q, k, v, o,
  gate, up, down). Everything else is frozen and undecomposed. Config:
  `configs/e04/vpd_qwen3_0.6b_L2-3.yaml` (originally `vpd_qwen3_0.6b_L13-14.yaml`).
  - 7,680 components in total.
  - The e03 loss recipe and global CI function, with the delta component on.
  - bf16 autocast.
  - Batch 4 × 128 tokens, 20k steps.
- **Data.** NeelNanda/pile-10k: the first 95% of documents for training, the
  last 5% held out for every measurement below.
- **Topology.** A Qwen3 path schema for param-decomp (fork branch
  `feature/qwen3-topology`).
- **Graph.**
  - Built with param-decomp's dataset-attribution harvester, edge statistic
    |attr_abs| as in e03.
  - The embedding is removed as a source: the slice's input is the residual
    stream after the frozen layers before it (13 originally, 2 after the
    revision), not raw tokens.
  - Removing it is also necessary on memory grounds. Vocabulary × components
    accumulators would need about 9 GB.
- **Nodes and edges.**
  - Nodes are the alive components: those with CI > 0.1 on at least 1 in
    10^5 held-out tokens.
  - Sources are the components of the slice's first layer that read the
    residual stream: q, k, v, gate and up.
  - The sink is the output. Component → output edges are the norm of the
    component's effect in logit space.

## Revision: slice moved to layers 2-3

Made 2026-10-15, before any Q1-Q3 analysis of a trained decomposition.

**What happened with layers 13-14:**
- The first run trained to step 7,000 and then crashed (CUDA out of memory).
  A pipeline test of mine ran alongside it on a nearly full card, which is the
  likely cause.
- Its evaluations by then showed that the slice barely matters to the model's
  output. Zeroing all seven projections in both layers costs only KL ≈ 0.15
  nats.
- The CI-masked decomposition left 69-92% of that gap unrecovered, against the
  quality gate of ≤ 10%. The gate looked unreachable, and Q2/Q3 would have
  measured differences inside a 0.15-nat window.

**New selection rule.** It uses only the target model, measured on held-out
rows (`experiments/e04_layer_ablation.py`, `results/e04/layer_ablation.json`):
decompose the deepest consecutive pair of layers whose zero-ablation costs at
least 1 nat of KL.

| Layer pair | Zero-ablation KL (nats) |
|---|---|
| 0-1 | 11.6 |
| 1-2 | 7.5 |
| 2-3 | 6.5 |
| 3-4 | 0.91 |
| 13-14 | 0.15 |
| deeper pairs | 0.14-0.95 |

Selected: **layers 2 and 3** (`configs/e04/vpd_qwen3_0.6b_L2-3.yaml`).
Everything else is unchanged: C per projection, the loss recipe, data, steps,
the gate, and Q1-Q3.

**Disclosure.** Before this revision I saw the outputs of a pipeline smoke test
on the 13-14 run's step-5,000 checkpoint: 2,048 attribution tokens, 5 shuffles
and 3 random sets. Those were code checks, too small to bear on Q1-Q3. The
revision rests on the decomposition-quality numbers above.

## Decomposition quality gate

The decomposition is interpreted only if the CI-masked model leaves at most
10% of the zero-ablation cross-entropy gap unrecovered: param-decomp's
`ce_unrecovered_ci_masked` ≤ 0.10, at the final evaluation. For scale, e03's
toy run left 1.3%. If the gate fails, the result is reported and Q1-Q3 are not
scored.

### Addendum: what follows each gate outcome

Written 2026-10-18 at 01:11. The third layers 2-3 run (p-1ae4ebad) had started
training at 01:10, and none of its evaluations had been read.

**Background.** The first run was killed by a Windows Update restart at step
3,689. The second (p-d8e53aab) hung at step 19,795 after a power flap. I read
the second run's gate metric while deciding whether to retrain:
- it stayed between 0.07 and 0.105 from step 6,000 on;
- it was 0.096 at step 19,000.

So the gate may go either way. These contingencies are fixed before the
outcome is known:

- **Gate passes.** Q1-Q3 are scored as planned. A replication follows with
  decomposition seed 1, everything else unchanged.
  - It is scored by the same gate and the same Q1-Q3 rules (R-Q1 to R-Q3).
  - It runs when the GPU is available. The card is shared, so the user decides
    when.
- **Gate fails.** Q1-Q3 are not scored on this run, as stated above.
  - e04b retrains with 40,000 steps, the length of e03's decompositions, all
    else unchanged.
  - The same gate applies. If e04b passes, Q1-Q3 are scored on it, still
    confirmatory, since no Q1-Q3 quantity of any layers 2-3 run will have been
    seen.
  - If e04b also fails, e04 reports that VPD at this budget does not decompose
    the slice well enough to test transfer.

## Predictions

**Q1: connected structure.**
- Statistic: the sieve's peak z, as in e03's P1.
- Pass: the peak z exceeds all of 100 shuffles' peaks (p < 0.01) against both
  controls: within-layer shuffles and in-strength-preserving shuffles.

**Q2: the sieved structure carries behaviour.**
- Restrict the slice to the sieved components: the mask is CI × 1[c ∈ S],
  every other component in the slice is off, and the rest of the model is
  intact.
- Measure: KL(target ‖ restricted) on held-out text.
- Pass: the sieved set's KL is below all of 100 random sets matched per
  module in size and firing-rate decile (p < 0.01).

Clarification, added before any e04 analysis: "the sieved components" in Q2 and Q3
are the backbone of the within-layer sieve at its c*, as in e03's P4.

Clarification, added 2026-10-15 while the layers 2-3 decomposition trained, before any
of its outputs existed: the held-out rows are split three ways by one fixed shuffle.
- Firing statistics come from rows 0-1,999.
- Attributions, and so the graph and the backbone, come from rows 2,000-3,999.
- The causal test (Q2, Q3) comes from rows 4,000-5,909.

The rows that select the backbone are never the rows that test it. The pipeline
code had read attributions from the training split; the setup above says held-out.

**Q3: better than picking by importance.**
- Pass: the sieved set's KL is below that of the same-size top-mean-CI set.

**Reference points, reported but not tested:**
- the full CI-masked slice (all components);
- the slice with every component off.

## What would count as transfer

- **Transfer:** Q1-Q3 all pass.
- **Partial:** Q1 passes but Q2 or Q3 fails. That would mean connected
  structure exists but does not single out the behaviourally important
  components at this scale.
