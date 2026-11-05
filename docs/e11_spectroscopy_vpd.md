# E11: Spectroscopy × VPD, a conditional causal-selection test

**Pre-score protocol drafted 2026-11-02.** This is a new study; E03–E10 outcomes informed the question. The tools below have passed synthetic CPU unit tests. **No real VPD response matrix or new causal outcome has been scored.** E10 is an independently frozen comparison and must stay unchanged.

## Question

Does the *shape* of each VPD component's signed response across calibration data predict a useful set of components after accounting for local in/out degree, individual causal importance (CI), total absolute attribution, firing rate and module identity?

Resolution (formerly Timaeus), *Spectroscopy at Scale: Finding Interpretable Structure in Pythia-1.4B* (2026-04-21), clusters token-specific **weight-space susceptibility** vectors. Its components are attention heads/MLP layers, and its actual estimator is based on posterior sampling. In *How to Scale Susceptibilities*, a component-specific response is associated with a restricted posterior on a slice of parameter space, plus an unrestricted chain. Consequently an attribution score, gradient, activation or arbitrary perturbation fingerprint is **not** a Resolution susceptibility. Our first phase uses a cheaper `vpd_response_proxy` and labels it as such. A true SGLD susceptibility replication is a separate phase with sampler quality controls.

Sources:
- https://timaeus.co/research/2026-04-21-spectroscopy-main/
- https://timaeus.co/research/2026-04-21-spectroscopy-definitions
- `docs/e09_utility_decision.md`; `docs/e10_decisive_gate.md`

## Explicit hypothesis and unit of intervention

For calibration perturbation/token `i` and fixed VPD component `c`, export a signed response `R[i,c]`. For the inexpensive probe, use the per-example derivative of the loss with respect to a frozen component gate, `d loss_i / d gate_c` at the unmasked VPD reconstruction. This is a **forward/backprop sensitivity proxy**, not a distribution-shift derivative of a local posterior. Use a fixed trained target, identical frozen VPD decomposition, canonical `(module, local_component_id)` identities and a locked calibration split. For real Resolution-style spectroscopy, replace `R` with component-specific SGLD susceptibilities and report restricted/full-chain estimators, burn-in, mixing, temperature and Gaussian/infinite-temperature controls.

The **outcome** is new held-out paired sequence KL from the target to the CI-masked VPD component model, with a fixed number of retained components in each module. Single-component CI is part of the baseline, not the definition of success. A selector may score well on individual CI yet fail when components interact or cancel each other.

Null: conditioning on degree, CI, attribution, rate and module makes the response geometry uninformative about the held-out causal outcome. Alternative: residual response geometry gives a positive, independent intervention advantage.

## E11 CPU selection, frozen before model inference

A response exporter writes `.npz` arrays with `allow_pickle=False`:

| Key | Shape/type | Requirement |
|---|---|---|
| `responses` | `[calibration_tokens, components]`, float | Signed, finite. Train/calibration only |
| `modules`, `local_ids` | `[components]`, strings/integers | Unique IDs; only eligible alive VPD components; align exactly with E10 graph |
| `nuisance` | `[components,5]`, float | In-degree, out-degree at E10 `c_star`, mean CI, total absolute attribution, firing rate, in that order |
| `graph_sha256` | scalar string | Must match the original E10 selector manifest |
| `vpd_run` | scalar string | Must match the frozen VPD checkpoint identifier |
| `calibration_split` | scalar string | `train` for real inputs |
| `response_source` | scalar string | `vpd_response_proxy`, `sgld_susceptibility`, or `synthetic_fixture` |

The exporter must obtain the nuisance columns from **the same E10 attribution graph and cross-reference statistics**, preserve the E10 alive-component pool, document training sample hashes/source IDs and archive the export provenance. These are prerequisites for a real-data experiment. The E11 selector verifies the graph/checkpoint identifiers and that every original E10 selected component exists; source-document disjointness still needs an external audit. A `sgld_susceptibility` label alone does not certify the physical estimator.

1. Establish a **joint degree+CI** reference: equal-weight mean of within-module standardized `log1p(in_degree + out_degree)` and `log1p(mean_CI)`. Deterministic ties use the component index. Its per-module quota is the already frozen E10 sieve budget.
2. For every module, bin candidates into three quantiles each of degree and CI. Freeze the **joint** `(module, degree_bin, CI_bin)` counts occupied by the reference selection. The spectral selector must preserve each count. Some bins are degenerate under tied values; report number of possible replacements.
3. Regress each component's centered response profile on *all five nuisance features* plus module indicators, using least squares across components; then normalize residual component profiles to unit length. Treat near-zero residuals as zero. This removes linear nuisance-predictable response and overall response amplitude. The resulting Gram matrix `K[c,d]` preserves signed angular relations and is invariant to independently flipping the signs of component signatures in its log-determinant objective.
4. Greedily maximize `log det(I + K[S,S])` under the exact joint-stratum budgets, with fixed index tie-breaks. This selects components spanning distinct residual response directions. Compare with `joint_degree_ci`, every original E10 deterministic selector, and 20 conditional permutation selections obtained by shuffling component signatures **within joint strata** before running the same algorithm.

The spectral selector could select identical components to a local reference. Such equivalence is a CPU-only stop: the masks necessarily give identical model outputs. Neither an interesting cluster map nor residual variance is itself a causal result.

## Locked held-out scoring

- Use untouched E10 **test** sequences and the original E10 model/CI mask code. Export only calibration data from `train`. E10 excludes exact 512-token sequences from its earlier E03 causal run; further document-level disjointness should be audited. Score identical sequences across all masks.
- Evaluate `all`, `none`, `spectral_geometry`, `joint_degree_ci`, the six E10 deterministic selectors, and the conditional permutation sets at the **same per-module budget**. Avoid tuning the spectral selector or threshold on evaluation outcomes.
- Measure `extra(B) = (mean_KL(B) - mean_KL(spectral_geometry)) / (mean_KL(none) - mean_KL(all))`, positive when spectral geometry is better. Compute paired 95% bootstrap intervals over whole 512-token sequences, with 2,000 draws. The minimum practical advantage is `extra >= .02` **and** a lower 95% bootstrap bound above zero for **every** deterministic reference including the percolation sieve. Retain the existing VPD quality rule `KL_all / KL_none <= .25`.
- Report one-sided conditional permutation `p = (1 + #{permutation_KL <= spectral_KL}) / (1 + n_permutations)`. With 20 permutations the smallest possible p is 1/21 = 0.0476. Treat p <= .05 as a secondary exploratory condition; conditional permutation matching uses bins and hence does not guarantee exact continuous-nuisance matching.
- A pass in two *frozen VPD decompositions of the same SimpleStories target* supports **within-target** usefulness only. Strong evidence of transfer requires a separately trained faithful target and new prompts. Report failed or underpowered gates, exact selector overlaps, model/source limitations and partial effects without rescoring thresholds.

## Commands and code

```bash
# CPU selection and unit tests; Python with numpy and pytest:
python -m pytest -q tests/test_e11_spectral_geometry.py
python experiments/e11_spectral_geometry.py \
  --e10-manifest out/e10_seed1/manifest.json \
  --responses out/e11_seed1/train_component_responses.npz \
  --output out/e11_seed1/selection.json --n-nulls 20

# Existing E10 runner recognizes extra selectors at identical module budgets.
# Execute ONLY after freezing the selection JSON, using the local VPD checkpoint:
python experiments/e10_e03_causal.py --e03-dir out/e03_seed1 \
  --manifest out/e11_seed1/selection.json \
  --output out/e11_seed1/causal.json --device cuda

# Independent CPU scoring, with original manifest hash verification:
python experiments/e11_spectral_score.py \
  --manifest out/e11_seed1/selection.json \
  --causal out/e11_seed1/causal.json --output out/e11_seed1/score.json
```

On the shared GPU, use the existing `vramorama` lease/watchdog as E10 documents. Costs can be bounded by first running CPU selector diagnostics and stopping if no new masks or no residual signal occur. The E11 `.npz` exporter and real posterior samples have **not** yet been produced. The six unit tests use synthetic matrices; positive or negative outcomes there establish implementation behavior, not VPD effectiveness.

## Follow-up if the gate passes

On a separate trained target, compare the response proxy against actual restricted/full-chain SGLD susceptibilities and a matched infinite-temperature/Gaussian sampler. Also evaluate leakage from position-zero attention sinks, basis rotations/permutations across VPD seeds, component-interaction synergy and sign cancellations. A source-document-grouped bootstrap and model-level replication should precede a strong explanatory claim.

**Scientific stop:** If the geometry selector loses or only ties the degree+CI reference after the faithful-VPD gate, record the negative outcome and keep the response-geometry code as a calibrated diagnostic. If the VPD reconstruction fails quality control, classify the study as untestable on that decomposition rather than as evidence about geometry.