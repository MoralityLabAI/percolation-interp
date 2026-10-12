# e03 replication: decomposition seeds 1 and 2

Written 2026-10-12, before any replication run. It covers the same target
model as e03 (pretraining's seed is fixed in param-decomp, so the target is not
re-drawn), with new seeds for the SAE tower and for VPD. Every stage repeats
e03 exactly, except as noted:

- harvest uses 300 batches instead of 500;
- each graph is also sieved against the in-strength-preserving control
  (within-target shuffles, 100 shuffles).

The driver is `experiments/e03_replicate.py`.

## What is being tested

Seed 0's confirmed findings, plus three exploratory ones that are now stated
as predictions:

| ID | Prediction | Origin |
|---|---|---|
| R1 | VPD P1 passes against the within-layer control and against the in-strength control (each p < 0.01, 100 shuffles) | P1 confirmed; in-strength was exploratory |
| R2 | SAE graph passes against the in-strength control (p < 0.01, 100 shuffles) | exploratory on seed 0 |
| R3a | P3 (weight-space alignment of the sieved sets) passes, p < 0.01 | P3 confirmed |
| R3b | Jaccard co-firing, with both sides frequency-matched, is enriched (p < 0.01) | exploratory on seed 0 |
| R4a | P4 passes: the VPD backbone's KL is below all 100 matched random sets | P4 confirmed |
| R4b | The VPD backbone's KL is below that of the same-size top-mean-CI set | exploratory on seed 0 |
| R5 | P5 passes in at least 2 of 3 levels, using the SAE backbone from the in-strength sieve | P5 failed on seed 0 with the within-layer backbone |

P2 (PMI co-firing) is reported but not predicted either way.

**Counting.** Each prediction is scored for seeds 1 and 2 separately.

- **Replicates:** passes in both seeds.
- **Partial:** passes in one.
- **Does not replicate:** passes in neither.

Results go in `e03_results.md` under "Replication".
