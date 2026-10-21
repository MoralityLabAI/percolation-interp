# Exploratory PAMA sampling sensitivity

Plan fixed on 2026-10-21 before running this diagnostic, while e05b seed 1
trains remotely. This is a robustness check of the baseline, not a new
confirmatory prediction, slice selection, or change to e05/e05b's gate.

Use the cached Qwen3-0.6B target in float32 at context 128. Compare position-aware
writer means estimated on nested prefixes of 64, 256 (the gate's existing
setting), and 1,024 held-out stats rows. Evaluate all conditions on the same
first 512 causal rows, batch 4. Check the already-selected layers 0-1 and the
earlier massive-activation slice 2-3; do not choose a different slice.

Report full-vocabulary KL(target || PAMA), the first 128 causal rows used by
the original gate, implied 0.25-times-KL thresholds, and paired differences
against the 256-row mean estimator. Bootstrap the 512 row observations with
2,000 draws, seed 51021. These intervals condition on the fitted writer means;
they do not include training-mean uncertainty. Documents may span token rows,
so the row bootstrap is descriptive and does not guarantee document-level
coverage. The nested sample differences are sensitivity measurements, not
independent replications.

Run `experiments/e05_pama_sensitivity.py` with the param-decomp Python through
a 6 GiB VRAMorama lease. It uses only locally cached model weights and tokens,
writes `results/e05_pama_sensitivity.json`, and leaves every existing gate and
analysis output unchanged. No VPD training is launched. Stop after this fixed
grid; do not expand it based on the observed values.

## Results

Completed on 2026-10-21 on the local RTX 5080 Laptop GPU through VRAMorama
(6 GiB lease; about 4 GiB observed use, no spill). Computation after model
loading took 79 seconds. The existing suite passed: 62 tests.

| Mean-estimation rows | Layers 0-1 KL | Implied threshold | Layers 2-3 KL |
|---|---:|---:|---:|
| 64 | 7.65825 | 1.91456 | 0.77413 |
| 256 (existing setting) | 7.73262 | 1.93315 | 0.77392 |
| 1,024 | 7.69584 | 1.92396 | 0.77179 |

All table values use the same 512 causal rows. At the existing 256-row
mean-estimation setting, the row-bootstrap 95% interval is [7.62394, 7.84612]
for layers 0-1 and [0.74632, 0.80385] for layers 2-3.

Paired differences against that setting on layers 0-1 are -0.07437 nats
([ -0.08462, -0.06294 ]) for 64 rows and -0.03678 nats
([ -0.04421, -0.02949 ]) for 1,024 rows. Differences on layers 2-3 are
+0.00021 nats ([ -0.00116, +0.00170 ]) and -0.00213 nats
([ -0.00295, -0.00128 ]), respectively. These intervals have the limitations
described above; they are not tests of independently fitted mean estimators.

**Interpretation.** Across this fixed nested sample grid, the layers 0-1
baseline varies by less than 1%, and the implied gate threshold by about
0.019 nats. This sensitivity cannot explain the earlier local e05 seed 0's
failed gate (CI-masked KL 2.84335). It does not predict e05b's gate outcome.
The layers 2-3 baseline is also stable on this grid. None of these findings
changes the pre-registered gate, its sample counts, or the slice choice.

**Numerical cross-check.** Restricting the 256-row condition to the first 128
causal rows reproduces the recorded e05 seed 0 KL_PAMA: 7.77388472 against
7.77388540 in its existing gate.json (absolute difference 6.75e-7 nats).
The new evaluation uses the target model only; no VPD checkpoint was loaded.

Raw per-row values, paired intervals, hardware and library metadata are in
`results/e05_pama_sensitivity.json`. Existing gate and analysis files were
not overwritten.
