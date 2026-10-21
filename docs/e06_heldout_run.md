# e06 held-out run: implementation notes

How `docs/e06_design.md` (pre-registered, 4d49a98) was implemented. This note covers the code
and the places where the design left a choice open. Each choice was fixed before any held-out
graph was analysed. Results are in `results/e06/heldout_*.json`.

## Code

- `src/percinterp/diagnostics.py` (additions only):
  - `sequential_edge_swaps` and `joint_degree_null_sequential`: exact sequential
    Maslov-Sneppen double-edge swaps, one swap at a time.
    - Each step picks an unordered pair of edges uniformly among all same-block pairs.
    - Blocks are invariant under swaps, so the proposal is symmetric. A rejected proposal is
      a lazy step.
    - The chain is therefore uniform over the swap-reachable edge sets. A test checks this
      on an enumerated degree class.
  - `skeleton_keys`: the skeleton key set, prebuilt once per graph.
  - `decompose_fast`: the same rng draws as `decompose(null="shuffle")`, with bit-identical
    results (tested).
    - It caches the shuffle blocks and the ranking groups, and replaces a random-order
      `searchsorted` with one argsort per group.
    - About 1 s per draw on e03 VPD, against about 4 s for `decompose`.
- `experiments/e06_heldout.py`: one graph per run. Input is `--e03-dir`, `--e05b-dir` or
  `--qwen-dir` (code check).
- `experiments/e06_score.py`: the design's pass rules for P1-P5. It also says whether each
  prediction holds across the graphs it is given.
- Tests: `tests/test_sequential_swaps.py`.

## Running e05b (once `out/e05b_seed{0,1}` hold `dataset_attributions.pt`, `stats.pt`, `gate.json`)

From the repo root, in a single command (the two graphs run sequentially):

    PYTHONPATH=src .venv/Scripts/python.exe -u experiments/e06_heldout.py \
        --e05b-dir out/e05b_seed0 out/e05b_seed1 > out/e06_heldout_e05b.log 2>&1
    PYTHONPATH=src .venv/Scripts/python.exe experiments/e06_score.py results/e06/heldout_*.json

- It writes `results/e06/heldout_e05b_seed{0,1}.json`, with the gate status in `"gate"`.
- Per-draw arrays go to `out/e06_heldout/<name>/draws.npz`.
- Code check: the same path, run on `out/e04` (seed 0) with 1,000 draws, took 54 s. Its
  outputs went to a scratch location and are not committed.

## Choices the design left open

1. **Which e03 graphs.**
   - The held-out e03 graphs are the VPD graphs of seeds 1 and 2.
   - The design's cost estimate and decision 7 name only the e03 VPD graphs, so the SAE
     graphs were not run.
   - The VPD graphs were built exactly as `e06_explore.build("e03_vpd")` built seed 0's.
2. **c values and the grid.**
   - c_s = 0.1, c = 1 and c_L = 4 are evaluated as extra points.
   - The max-statistic (secondary) uses only the sieve's default 48-point grid, as `sieve()`
     does.
   - z at a fixed c uses the sieve's sd floor of one node (1/n_interior).
3. **One-sided p-values.** p = (1 + #{null ≥ real}) / (1 + 1000), for P2, P4 and P5.
4. **N-JD, 30 rounds.**
   - With the parallel sampler, a round is one pairing of every block's edges.
   - With the sequential sampler, 30 rounds means 30 × ⌊E/2⌋ proposals, the same number of
     proposals.
   - Every draw restarts from the real kept graph, uses skeleton edges only, and uses
     (source layer, destination layer) blocks.
5. **P4 zero spread.** A cell is not testable when every N-JD draw gives the same backbone
   fraction, even if the real value differs from it.
6. **Mixing check (e03).**
   - 200 exact sequential draws per cell, compared with the 1,000 parallel draws: mean
     difference, sd ratio, two-sample KS test.
   - Following decision 7, the parallel draws score P4. The sequential p is reported
     alongside.
7. **P5 ρ.**
   - Interior nodes with in- and out-strength > 0, in layers with at least 3 such nodes.
   - Ranks are standardised within each layer and pooled (as `e06_report.rate_partial`,
     which gave the discovery values).
   - Partial: both rank vectors are residualised on the firing-rate rank by one pooled
     regression.
   - Null: the out-strength ranks are permuted within each layer.
8. **Float slack.** P1's `Δalignment ≥ 0.5 Δb` and P3's `Δcoverage ≤ Δb` and
   `Δalignment ≥ −0.1 |Δb|` allow 1e-9.
   - At c_L, Cov is often exactly 0, so `Δcoverage = Δb` holds only up to rounding.
   - 1e-9 is far below one node.
9. **Seeds.** One rng seed per null, 6101-6109 (`SEEDS` in the script).

## e03 seeds 1 and 2 (confirmatory for P1, P2, P4, P5; P3 reported only)

The run was made once, on 2026-10-21. Seed 1 took 2.4 h and seed 2 took 1.9 h, sequentially,
on CPU. Log: `results/e06/heldout_e03.log`.

| | Seed 1 | Seed 2 |
|---|---|---|
| Graph (nodes / edges / interior) | 5,333 / 3.67M / 1,574 | 5,349 / 3.68M / 1,582 |
| **P1** Δb = Δcoverage + Δalignment at c_s, N-WL | **pass**: +0.052 = +0.017 + 0.035 (67%), z 16.4 | **pass**: +0.058 = +0.020 + 0.038 (66%), z 17.8 |
| **P2** backbone at c_s against N-DC | **pass**: 0.0610 vs 0.0065 ± 0.0027, z 20.4, p 0.001 | **pass**: 0.0670 vs 0.0066 ± 0.0028, z 21.3, p 0.001 |
| P2 secondary: grid max-statistic | z 31.0 (null max 4.95), p 0.001 | z 30.7 (null max 7.27), p 0.001 |
| P3 (reported only) at c_L, N-WL | would pass: z −98.2; −0.282 = −0.282 + 0.000; Δr^F +0.000, Δr^B −0.283 | would pass: z −98.5; −0.284 = −0.284 + 0.000; Δr^F +0.003, Δr^B −0.286 |
| P3 extras: N-OS / N-IS z at c_L | −2.1 / −117.2 | −0.1 / −123.5 |
| **P4** c_s (S) | **pass**. Testable: 0.0610 vs 0.0590 ± 0.0012, p 0.079 (S 0.996) | **pass**. Testable: 0.0670 vs 0.0651 ± 0.0011, p 0.073 (S 0.998) |
| P4 c = 1 (S) | not testable (zero spread; S 1.000) | not testable (zero spread; S 1.000) |
| **P5** ρ(in, out), p; rate-partialled ρ | **pass**: +0.636, p 0.001; −0.019 | **pass**: +0.634, p 0.001; +0.003 |
| N-AB (reported) at c_s; grid max-statistic | z 6.3, p 0.001; z 10.7, p 0.001 | z 7.4, p 0.001; z 11.9, p 0.001 |

**Mixing check.** 200 exact sequential draws were compared with the 1,000 parallel draws.
- At c_s:
  - Seed 1: means 0.0591 vs 0.0590 (difference +1.0 SE), sd ratio 1.09, KS p 0.75.
  - Seed 2: means 0.0649 vs 0.0651 (difference −2.2 SE), sd ratio 1.06, KS p 0.17.
  - With the sequential draws as the null, P4's c_s p would be 0.13 and 0.050. Both are far
    above 0.005, so the P4 outcome is the same under either sampler.
- At c = 1, both samplers have zero spread.
- A correction made after the run, to a reported field only: in the c = 1 cells,
  `mean_diff_z` and `sd_ratio` came out as float-rounding artifacts (−31.6 and 0.0) because
  both samplers had zero spread. They were set to null in the JSON, and the script now does
  this itself.
