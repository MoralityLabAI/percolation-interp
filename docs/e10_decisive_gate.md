# E10: causal selector head-to-head and early stopping rule

**Date fixed:** 2026-10-22. **Status:** *Design and implementation*. The next causal inference run has **not** been performed. This study follows the retrospective E09 audit, so all ideas and thresholds here are prospective only for untouched E10 evaluation rows. Earlier E03 causal outcomes and E06/E07 degree results influenced this design. That limits confirmatory interpretation.

## Goal

Identify whether selecting components using the **percolation sieve's source-to-sink backbone** has incremental causal utility beyond simple local graph statistics, component importance, attribution mass, and activity. The primary target for which usable saved data may exist is E03's two-layer SimpleStories model. Start with decomposition seeds 1 and 2; both seeds are of the **same trained target model** and previously produced E03 causal results. A Qwen-specific transfer claim would require E05b-style quality-gated results on separate target slices.

The public GitHub repository currently contains summary JSON, but *not* the large E03 `crossref.pt`, `dataset_attributions.pt`, `vpd_run.txt`, full component model checkpoints, target model or token data. Raw artifacts are intentionally neither bundled with the site nor committed to GitHub. Consequently the GitHub Actions job can validate selector logic, manifest generation, analytical uncertainty and program syntax; it cannot execute new real-model causal forward passes without the original saved inputs.

## Freeze before evaluating (pre-score stage)

`experiments/e10_prepare.py` loads exactly the original E03 analysis/stats and attribution graph (never a causal-results file), derives a per-module quota from the E03 recorded backbone at its fixed `c_star`, and freezes a JSON manifest. The manifest contains graph, analysis and statistics fingerprints. Retain the initial manifest and refuse implicit overwrite.

All deterministic selectors receive *exactly the same per-module number of alive components*:

1. **Sieve (candidate):** the previously frozen E03 source-to-output backbone at `c_star`.
2. **Local degree:** positive retained in- and out-contact flags, then `min(in_degree, out_degree)`, then total retained degree. This computation never traverses a path.
3. **Local weighted degree:** geometric mean of retained incoming/outgoing **absolute** attribution strength, with total retained strength for ties. Again, no path traversal.
4. **Top CI:** individual mean causal-importance rank among alive components.
5. **Top attribution:** greatest sum of absolute incoming/outgoing attribution over the complete graph.
6. **Top firing rate:** greatest mean firing rate from statistics, a direct activity confound.

Also report 20 exploratory random subsets, drawn without replacement with firing-rate-decile matching and documented fallback when a decile is depleted. Keep all candidate pools, inclusion rules, tie-breakers and rate data fixed. Use graph weights and CI from earlier attribution/statistics splits.

**Cheap falsification gate:** report Jaccard overlap and per-module replacements between each selector and the sieve. If local-degree selection equals the sieve, the two component masks are identical. Their causal outcomes must be identical, so the claim that path percolation adds causal value beyond this local selector *already fails for that decomposition*. This observation needs no expensive forward pass.

The degree and activity selectors are purposefully chosen from the E09 analysis. A success here would establish utility conditional on the candidate implementation; it would not certify a model-independent effect.

## E03 held-out forward pass (separate stage)

`experiments/e10_e03_causal.py` takes the frozen manifest, verifies its input hashes, loads the existing target and VPD decomposition, and restricts the original component model using `CI × selector_membership` as in E03 P4. Both the full CI-masked model (`all`) and the fully zero-masked model (`none`) are retained as ablation references. Exactly identical component masks are deduplicated to avoid redundant inference.

For each held-out sequence, compute mean KL(target || restricted) over its 512 tokens and the full target vocabulary. The evaluation input loader uses a new shuffle seed (20261023) and explicitly excludes exact 512-token sequence hashes from the original E03 P4 seed-4 run (first eight batches of size eight). This certifies *no exact-sequence reuse*. It does **not** guarantee source-document disjointness or an independent test corpus. The original E03 tests already used this target, and model-level confirmation remains unavailable.

Default budget: eight evaluation batches of eight sequences, 64 distinct sequences, twenty rate-matched random sets, 2,000 **paired sequence bootstrap** draws. Do not tune a selector or its occupation after inspecting these outcomes. Preserve per-sequence KL arrays for audit. Statistical independence at the source-document level remains unresolved, so confidence intervals are descriptive for this held-out sample.

## Fixed practical decision rule

For each decomposition seed, require `KL_all / KL_none ≤ 0.25` as a toy-model quality check, and let `gap = KL_none − KL_all > 0`. A comparator's recovered-gap advantage is

`extra = (KL_comparator − KL_sieve) / gap`.

Positive values favor the sieve. The **incremental percolation gate** passes on a seed only if, against **every deterministic baseline**, `extra ≥ 0.02` and the lower bound of its paired 95% bootstrap interval is strictly above zero. Report individual effects even on a failure. The 0.02 practical effect threshold was set in E09 and is prospective only for untouched E10 rows. A failure on a matching local-degree set is exact, independent of bootstrap.

Two *distinct frozen decomposition manifests* must pass for `WITHIN_TARGET_SEED_GATE_PASS`. Since the target model is shared, that label is intentionally limited. If any deterministic baseline ties or beats the sieve, the target's E10 incremental-utility gate fails. Neither passing nor failing on E03 alone proves results for Qwen or for other model classes.

**Scientific stopping rule:** If the two seed results cannot beat a cheap degree/importance selector, retire claims that the percolation operation uniquely discovers mechanistic circuits in this setting. Retain the O(E) graph routines and the visualization as mathematical utilities. A passing E10 requires a genuinely independent faithful target before expanding claims or spending compute on inferred holonomy.

## Run with the user's existing local artifacts

The commands below operate on `out/e03_seed1` and `out/e03_seed2` if those directories and the saved `~/param_decomp_out/` attribution/checkpoint directories still exist. Use the param-decomp Python venv for VPD forward passes; no training or paid pod is required.

```powershell
# CPU, using the analysis environment with numpy, scipy, torch:
python experiments/e10_prepare.py --e03-dir out/e03_seed1 --output out/e10_seed1/manifest.json
python experiments/e10_prepare.py --e03-dir out/e03_seed2 --output out/e10_seed2/manifest.json

# Review overlap (the command prints local-degree/sieve agreement).
# GPU inference, under your existing vramorama lease/watchdog:
python experiments/e10_e03_causal.py --e03-dir out/e03_seed1 --manifest out/e10_seed1/manifest.json --output out/e10_seed1/causal.json --device cuda
python experiments/e10_e03_causal.py --e03-dir out/e03_seed2 --manifest out/e10_seed2/manifest.json --output out/e10_seed2/causal.json --device cuda

# Pure CPU scoring:
python experiments/e10_score.py --results out/e10_seed1/causal.json out/e10_seed2/causal.json --output out/e10_final.json
```

For a no-checkpoint CPU sanity check:

```bash
python -m pytest -q tests/test_e10_selectors.py
python experiments/e10_prepare.py --fixture --output out/e10_fixture/manifest.json
```

The `--fixture` route is explicitly tagged synthetic and cannot be passed to the real causal runner or the combined seed scorer.

## Related evidence

- E03: causal VPD subset selection beats top mean-CI across three decomposition seeds of the SimpleStories target.
- E04: Qwen3-0.6B transfer criterion Q1 fails on both seeds, with a pair-dependent, low-magnitude causal effect.
- E06: local in/out degree determines nearly all tested backbone membership; the registered degree-null wiring-excess test did not pass.
- E07 (saved report outside GitHub): learned and degree-rewired component/neuron cluster curves nearly coincide; feature semantics unmeasured.
- E09: retained-edge fraction predicts fidelity across depth/width more accurately than backbone fraction.

This study tests a narrow empirical promise: **does the graph traversal identify better components after controlling local degrees and importance?** It keeps the hypothesis falsifiable with a fixed cost and a strict provenance boundary.
