# e05: the percolation sieve on Qwen3-0.6B, with baselines that keep position 0

Written 2026-10-19 and approved as final the same day. It was committed before
the slice-selection measurement and before any e05 decomposition was trained.

## Why e05

e04 asked whether the sieve's toy-model results (e03) transfer to a real model.
It answered no: Q1 failed in both seeds. An exploratory analysis after both
seeds (`e04_results.md`, "After both seeds: the massive activation") showed that
e04's causal quantities measured something else.

- **The zero-ablation baseline is about 94% one vector.** Layer 2's `down_proj`
  writes a norm-6,856 vector at position 0, Qwen's massive activation. It
  anchors the attention sink. Deleting it accounts for 93.5% of the 6.66-nat
  zero-ablation gap.
- **What that inflated.** e04 relied on the zero-ablation gap in three places:
  - the slice-selection rule (zero-ablation KL ≥ 1 nat);
  - the quality gate (unrecovered cross-entropy against zero ablation);
  - the Q2/Q3 restrictions, where every component outside the set was zeroed.
- **The constants test.** Replacing what layers 2-3 write with its
  position-aware mean costs 0.76 nats. The CI-masked decomposition costs 0.71,
  so it barely beats constants.
- **The causal sets never had the sink.** The sieved backbones contained the
  massive activation's writers but not their drivers. So every Q2/Q3 set was
  scored with the attention sink missing.

e05 asks e04's question again, with every baseline keeping position 0. One
further change: the Q1 rule uses 1,000 shuffles, so a single outlying shuffle
cannot decide it.

## Definitions

**Position classes.** Position 0 of each 128-token row, and positions 1-127.

**Position-aware mean ablation (PAMA) of a set of modules.** Each module's output
is replaced by its mean over the position class of the token.
- The means come from the target model on the held-out "stats" rows (rows
  0-1,999 of the fixed shuffle in `e04_common.PARTS`).
- For the slice's residual writers (`o_proj`, `down_proj`), this replaces the
  slice's whole contribution to the residual stream by constants. Ablating the
  writers alone or all seven projections gives the same model.
- KL_PAMA is KL(target ‖ slice writers PAMA-ablated): the cost of the slice's
  token-dependent computation.

**Component-level mean ablation (for Q2-Q4).** Within a decomposed module, the
output is Σ_c ci_c · a_c · U_c, with CI ci, activation a and output direction U.
- To keep a set S of components, components in S run as in e04 (CI × activation).
- Every other component's contribution is replaced by its position-class mean
  of ci_c · a_c, times U_c.
- Those means are measured under the full CI-masked decomposition on the stats
  rows.
- With S empty, this is the decomposition at its means: "none". With S = all,
  it is the CI-masked decomposition: "all". The gap is KL_none − KL_all.

## Setup

- **Model and data.** As e04: Qwen/Qwen3-0.6B; NeelNanda/pile-10k, first 95% of
  documents for training and the last 5% held out. The held-out rows are split
  three ways as in e04: stats, attributions and causal rows.
- **Slice selection** (target model only, stats rows), decided before
  measuring:
  1. Exclude any layer whose residual writer puts out, at position 0, a mean
     norm above 100 times its mean norm at other positions. These layers write
     a massive activation. Layer 2's `down_proj` (about 1,300×) is excluded by
     this rule. Layer 2's `o_proj` (about 6×) would not be.
  2. Among consecutive layer pairs with neither layer excluded, pick the pair
     with the largest KL_PAMA.
  3. All pairs' KL_PAMA and the excluded layers are reported.

  Why exclude massive-activation layers: VPD's training loss is computed over
  all positions, so a slice that writes the massive activation spends its
  decomposition on one constant vector (e04 again).
- **Decomposition.** As e04: param-decomp VPD on all seven projections of both
  layers, 7,680 components, the e03/e04 loss recipe and config.
  - 20,000 steps, seeds 0 and 1, both pre-registered here.
  - Run by `experiments/e04_pipeline.py`'s watchdog driver.
- **Graph and sieve.** As e04: attribution harvester on the attributions rows,
  |attr_abs|, alive threshold 10^-5 on 102,400 tokens, sources are the first
  layer's readers, sink is the output.
  - Within-layer and in-strength-preserving shuffles, now 1,000 of each.

## Quality gate

The decomposition is interpreted only if the CI-masked model recovers at least
75% of the slice's token-dependent computation:

    KL(target ‖ CI-masked) ≤ 0.25 × KL_PAMA

This is measured on the causal rows after training, separately for each seed.
For scale, e04's layer 2-3 decompositions were at 0.71 and 0.69 against a
KL_PAMA of 0.76, so they would have failed.

If a seed fails, its Q1-Q4 are not scored for that seed. A 40,000-step retrain
follows under the same gate. If that also fails, e05 reports that VPD at this
budget does not decompose the slice beyond its constants.

## Predictions

Each prediction is scored separately per seed. **Transfer requires every
prediction to pass in both seeds.**

**Q1: connected structure.**
- Statistic: the sieve's peak z, as in e03 and e04.
- Pass: p ≤ 0.01 against both controls (within-layer and in-strength shuffles).
  p is the share of the 1,000 shuffles whose own peak z is at least the
  graph's, with the +1 correction.

**Q2: the sieved structure carries behaviour.**
- Restriction: the within-layer sieve's backbone S at its c* is kept, and every
  other component is at its position-class mean (component-level mean
  ablation).
- Measure: KL(target ‖ restricted) on the causal rows, full vocabulary, all
  positions.
- Pass: below all of 100 random sets matched per module in size and
  firing-rate decile (as e04).

**Q3: better than picking by importance.**
- Pass: the backbone's KL is below that of the same-size top-mean-CI set.

**Q4: better than connected noise.** New in e05. It answers e04's finding that
per-module random sets rarely contain both halves of a route.
- Null sets: for each of 100 within-layer shuffles from Q1, take the backbone of
  the shuffled graph at the same c* (relabelled onto the real components).
  - Trim or pad it, by node entry order, to the real backbone's size in each
    module.
  - These are connected structures found in noise.
- Pass: the backbone's KL is below all 100.

**Reported, not tested:**
- KL_none, KL_all, KL_PAMA, and every slice pair's KL_PAMA;
- whether the backbone contains any component whose firing is concentrated at
  position 0 (more than 90% of its CI mass there).

## What counts as transfer

- **Transfer:** Q1-Q4 pass in both seeds.
- **Structure without function:** Q1 passes in both seeds but Q2, Q3 or Q4
  fails in either.
- **Function without structure:** Q2-Q4 pass in both seeds but Q1 fails in
  either.
- **No transfer:** anything else.

## Order of work

1. Commit this design.
2. Measure slice selection (`experiments/e05_select.py`; about 10 GPU minutes)
   and commit the result.
3. Smoke-test the component-level mean ablation and Q4's null on e04's layer 2-3
   decomposition. These are code checks only, not data.
4. For seeds 0 and 1: decomposition (about 3.5 h), gate, stats, attributions
   (about 1.75 h), sieve, causal test. About 11-12 GPU hours for both seeds.
5. Write up `e05_results.md`.
