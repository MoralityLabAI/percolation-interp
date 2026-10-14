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
- **VPD slice.** Layers 13 and 14, all seven projections each (q, k, v, o,
  gate, up, down). Everything else is frozen and undecomposed. Config:
  `configs/e04/vpd_qwen3_0.6b_L13-14.yaml`.
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
    stream after 13 frozen layers, not raw tokens.
  - Removing it is also necessary on memory grounds. Vocabulary × components
    accumulators would need about 9 GB.
- **Nodes and edges.**
  - Nodes are the alive components: those with CI > 0.1 on at least 1 in
    10^5 held-out tokens.
  - Sources are the layer-13 components that read the residual stream: q, k,
    v, gate and up.
  - The sink is the output. Component → output edges are the norm of the
    component's effect in logit space.

## Decomposition quality gate

The decomposition is interpreted only if the CI-masked model leaves at most
10% of the zero-ablation cross-entropy gap unrecovered: param-decomp's
`ce_unrecovered_ci_masked` ≤ 0.10, at the final evaluation. For scale, e03's
toy run left 1.3%. If the gate fails, the result is reported and Q1-Q3 are not
scored.

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
