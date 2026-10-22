# E09 decision audit: percolation's actual utility

**Date:** 2026-10-22. **Status:** Retrospective, exploratory; no new model evaluations.
**Decision:** **NARROW RETAIN.** Keep the percolation sieve as a candidate
component-subset selector and cheap graph calibration tool. Stop treating
finite graph cluster thresholds or the synthetic Gödel Globe as semantic
circuit discoveries.

The script experiments/e09_utility_audit.py reanalyzes committed E02, E03,
E04 and E06 JSON files. This is not an independent confirmation and does
not supersede their original hypotheses.

## Causal component selection

| Target, decomposition | Sieve KL | Top mean-CI KL | Sieve gain | Extra full gap closed |
|---|---:|---:|---:|---:|
| SimpleStories seed 0 | 6.330 | 7.914 | 1.584 | 5.84 points |
| SimpleStories seed 1 | 6.164 | 7.606 | 1.442 | 5.31 points |
| SimpleStories seed 2 | 5.729 | 8.099 | 2.370 | 8.73 points |
| Qwen3-0.6B L2-3 seed 0 | 6.317 | 6.398 | 0.081 | 1.36 points |
| Qwen3-0.6B L2-3 seed 1 | 6.433 | 6.435 | 0.002 | 0.04 points |

Lower KL is better. Extra gap closed =
(KL_top-CI - KL_sieve) / (KL_none - KL_all). These are saved causal
evaluations, with decomposition seeds of the same target, not
independently trained models. E03's causal predictions were written as
an addendum before the causal runs. Qwen's registered connectivity
criterion Q1 failed in both seeds, and one attention value/output pair
explains most of its apparent causal gain.

**Surviving positive finding:** On the small language model,
percolation-selected subsets beat top mean-CI and size-matched random
subsets. Their incremental value against a strong degree/local
alignment selector has not been tested causally.

## New retrospective out-of-configuration fidelity prediction: E02

Question: does backbone fraction forecast held-out fidelity curves better
than the simple nominal kept-edge fraction?

Source: results/e02_depth/report_summary.json. The 13 grokked runs are
grouped into five (depth, width) configurations, with *all* seeds of one
configuration held out at a time. A predictor maps its x-coordinate to
recovery by averaging piecewise-linear training-run templates. All
predictors use the same training data and testing procedure.

The axes are log(c/width), log(c), and the raw backbone-edge fraction.
A fourth model adds a training-fitted linear correction from the
backbone fraction residual after conditioning on log(c/width).
MAE is macro-averaged across the five held-out configurations.

| Predictor | MAE, all 15 occupation points | MAE, recovery 0.05-0.95 |
|---|---:|---:|
| **Nominal kept fraction** | **0.02077** | **0.05945** |
| Kept outdegree | 0.04543 | 0.13483 |
| Backbone edge fraction | 0.05236 | 0.16006 |
| Kept fraction + backbone residual | 0.02199 | 0.06007 |

The backbone correction worsens all-points MAE in **five of five**
configuration holdouts. The simple kept fraction predicts recovery
best in this retrospective comparison. Endpoint clamping, one arithmetic
task, one shared training setup and only five independent configuration
groups limit generalization. This is a descriptive stress test, not a
significance test. E02's registered stricter H1b also failed.

## E06 wiring beyond local degree

Under E06's pre-registered degree-preserving null at c=0.1:

| Seed | Share of backbone determined by local in/out degree | One-sided wiring p |
|---|---:|---:|
| E03 seed 1 | 0.99555 | 0.0789 |
| E03 seed 2 | 0.99810 | 0.0729 |

The registered corrected threshold was p <= 0.005. At c=1,
degree-preserving rewiring could not change backbone membership at
all. The significant E06 alignment result remains real: it tests how
node inputs and outputs correspond under an in/out-decoupling null.
The observed shallow backbone does not establish long-range wiring.

## E07 and the contact graph

The saved E07 exploratory report, still absent from the public
repository, gives learned-vs-same-degree differences <= 0.00197 in
the 50%-largest-cluster crossover across twelve Qwen coordinate maps.
Top-eight contacts hold only 2.72-3.64% of a median direction's
squared energy. The resulting graph is a geometric contact diagnostic.
It has not measured semantic polysemanticity, model-lesion behavior
or a universal percolation critical point.

## E09 proof-of-capability and proof-of-limit fixtures

These are mathematical tests, not evidence about learned models:

1. Two four-layer directed graphs have identical per-node in/out
   degree sequences and identical largest undirected cluster sizes.
   Swapping a middle-layer perfect matching changes whether a
   designated source can reach the designated sink. Directed
   percolation detects a property missing from those simpler summaries.
2. Two two-path linear networks have identical connectivity and the
   same absolute edge weights, but one outputs 2 and the other 0
   because of signed path cancellation. Connectivity alone does
   not determine function.

## Research gate

| Proposition | Current status |
|---|---|
| Fast ranked influence-graph sweep and null calibration | **Useful engineering primitive** |
| Special universal cluster transition in trained networks | **Unsupported** |
| Backbone fraction predicts fidelity across size/depth | **Failed in E02 and E09** |
| Component subsets selected using the sieve can preserve behavior | **Promising in E03 only** |
| Qwen replicates the same mechanism | **Unproven; Q1 failed** |
| Structural contacts quantify polysemanticity | **Unmeasured** |
| Globe animation proves Qwen holonomy | **Synthetic calibration only** |

**Decisive next gate:** On a faithful VPD target, freeze component
budgets and thresholds before evaluating disjoint held-out prompts.
Compare sieve selection to a local-degree/in-out-alignment selector,
top mean-CI, strongest attribution and frequency/module-matched random
subsets. Measure KL and recovered effect by paired prompt or sequence,
then repeat across decomposition seeds.

Prospective practical expansion rule: at least **two additional
percentage points of the available ablation gap** versus every
deterministic baseline in two or more decomposition seeds, without
failing the decomposition quality gate. Chosen after reviewing
E02-E07, this criterion applies only to untouched future evaluations.
It is not a significance threshold.

If this comparison fails, retire *percolation as a special circuit
discovery theory* and retain the cheap graph tools. If it passes,
call it a selection heuristic pending harder degree-matched and
path-intervention tests. Further Qwen holonomy work requires actual
activation-conditioned shared-rank feature frames.
