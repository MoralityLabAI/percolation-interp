# Handoff, 2026-10-22 05:04 Singapore

Snapshot of everything in flight for the next session, human or agent. The repo
is `MoralityLabAI/percolation-interp` (GitHub, branch `master`); the local checkout
is `C:\Users\patri\Documents\GitHub\MoralityLabAI\percolation-interp`.

Read `docs/e05_design.md`, `docs/e05b_design.md` and `docs/e06_design.md` before
touching anything below. All three are pre-registered. Never change a criterion
after seeing results; log deviations in the design doc.

## Live processes

### 1. e05b: both remote training runs complete; paid pod deleted

- Pod `a818c7656db149eba8122c9956b20f85` was deleted by the original fetcher
  at 2026-10-22 01:14 Singapore. A successful Prime API list confirmed its
  absence at 01:15:47; independent follow-up lists also returned no pods.
  No new pod, training extension, or deadline extension was requested.
- Both complete run directories (70 files total) are local under
  `C:\Users\patri\param_decomp_out\decompositions\`:
  seed 0 `p-7e116244`, seed 1 `p-06edd191`. Both are layers 0-1, batch 32,
  20,000 steps. Every manifest filename/size was checked; both final checkpoints
  (2,528,576,999 bytes each) matched SHA-256 hashes captured on the remote pod.
  Evidence and hashes: `results/e05b_retrieval.json`.
- Full copies completed at 00:57 (seed 0) and 01:14 (seed 1). Fetch log:
  `out/e05b/fetch_2.log`. SSH failure alone was not used as deletion evidence.
- The original 02:30 cutoff was never changed. There is no paid compute left
  to guard; do not relaunch the old pod fetcher/watchdog or rent a replacement.

### 2. Local follow-through and resource state

- The original e05b driver is running under VRAMorama; seed 0 passed its gate
  and completed stats. Its 13 GiB attribution lease c7a051ce was admitted at
  approximately 05:04 Singapore after the GPU blocker was removed.
- The human explicitly authorized stopping the idle nl-steering process.
  PID 30092 was re-verified as the expected text_effects.py job (0% GPU
  utilization, 3.8 GiB held) and stopped at 05:03. GPU memory was released;
  no unrelated process was stopped. Attribution batch/sample counts and
  VRAMorama admission rules were not changed.
- A small follow-through supervisor is already running (owner
  `e05b-followthrough`); log `out/e05b/followthrough.log`. Its temporary script
  and manifest live in the Codex chat workspace's `work/` directory. It verified
  copies/deletion, waits for the existing driver to finish, fills any missing
  analysis stages without ever training, then runs the remaining e06 checks.
  Do not launch a duplicate supervisor or driver.
- The two new `param-decomp-p-...` directories in `param_decomp_out/runs/` are
  directory junctions to their matching decomposition folders, avoiding about
  5 GB of duplicate final checkpoints. Preserve these targets.
- Disk space after full retrieval was about 3 GB. Unused intermediate
  checkpoints were losslessly compressed in place: model_5000.pth for both
  seeds and seed 0 model_10000.pth. SHA-256 was identical before and after
  each compression; free space increased to about 7.18 GB. No checkpoint
  contents were discarded or changed.
- The human requested sparse agent follow-ups, roughly four hours for the
  remaining analysis, rather than ten-minute polling. Do not push or publish;
  commit completed local results/docs after appropriate checks, staging only
  task files and leaving unrelated changes untouched.


### 3. e05b local analysis (seed 0 attributions admitted)

Seed 0 passed the gate: CI-masked KL 1.763079 versus limit 1.943471
(KL_PAMA 7.773885); stats.pt is present. Seed 1 gate is still pending.
The attribution stage was admitted after the GPU became available. For e06, both graphs count even if
a gate fails: harvest their stats/attributions without scoring failed seeds
on e05 Q1-Q4. Never retrain e05b.

Per seed, under vramorama leases with watchdogs (the e05 driver):
1. Gate: `e05_gate.py`, KL(CI-masked) ≤ 0.25 × KL_PAMA. If it fails, Q1-Q4 are not scored.
2. Stats.
3. Attributions: batch 16 × 50, about 1.75 h, 13 GB.
4. Sieve: `e05_analyze.py`, 1,000 shuffles, both controls, plus Q4 null sets.
5. Causal: `e05_causal.py`, component-level mean ablation, Q2-Q4.

Outputs go to `out/e05b_seed0/` and `out/e05b_seed1/`, with `gate.json`,
`analysis.json` and `causal.json`. Expect about 2-2.5 h per seed after GPU admission, subject to CPU analysis time.

### 4. e06 held-out run: e03 done, e05b pending

- **Merged into master:** the held-out runner `experiments/e06_heldout.py`, the
  scorer `experiments/e06_score.py`, exact sequential swaps in `diagnostics.py`, and
  62 passing tests. Details: `docs/e06_heldout_run.md`.
- **Confirmatory e03 seeds 1 and 2 (VPD): P1, P2, P4 and P5 pass on both.**
  - P2: z 20.4 and 21.3, p 0.001.
  - P4: the c_s cell is testable with no wiring excess (p 0.079 and 0.073); the
    c = 1 cell is not testable.
  - P5: ρ 0.64, falling to about 0 once firing rate is partialled out.
  - P3, reported only for these graphs: coverage, on the sink side.
  - Results: `results/e06/heldout_e03_seed{1,2}.json`.
- **Remaining:** once `out/e05b_seed{0,1}/` have `dataset_attributions.pt`,
  `stats.pt` and `gate.json`, run from the main checkout:

      PYTHONPATH=src .venv/Scripts/python.exe -u experiments/e06_heldout.py --e05b-dir out/e05b_seed0 out/e05b_seed1
      .venv/Scripts/python.exe experiments/e06_score.py results/e06/heldout_*.json

  Then write `docs/e06_results.md`. A prediction holds only if it passes on every
  held-out graph it applies to; P3 counts on the e05b graphs only.

## Local exploratory baseline check, 2026-10-21

Completed a target-only PAMA sampling diagnostic through a 6 GiB VRAMorama lease
while remote e05b training continued. Layers 0-1 KL_PAMA varied by less than 1%
across means estimated from 64, 256 and 1,024 stats rows, evaluated on the same
512 causal rows; implied thresholds ranged from 1.91456 to 1.93315 nats. This
sensitivity does not explain the earlier e05 failure at KL 2.84335, and says
nothing yet about e05b gate results. The original baseline reproduces within
6.75e-7 nats. No gate, slice or sample-count settings were changed. Details:
`docs/e05_pama_sensitivity.md`; raw rows: `results/e05_pama_sensitivity.json`.
Existing suite: 62 tests pass.

## e07: component/neuron percolation exploration

The human requested using the saved VPD to explore percolation math applied
to polysemanticity. New exploratory plan: `docs/e07_design.md`; derivation
and interpretation limits: `docs/e07_math.md`. The implementation is
`src/percinterp/polysemanticity.py`, tested against independent sparse graph
measurements and known fixed points (68 tests total pass). It treats static
component/coordinate sharing as a structural proxy, not semantic labels.

`experiments/e07_geometry.py` is queued as a CPU-only job under VRAMorama
owner `percinterp-e07-geometry`, lease f7c60a48. The system commit-charge
guard has not admitted it yet; preserve that guard and the existing GPU
harvest. It will analyze both saved seeds, write `results/e07/geometry.json`,
and produce a scientific PNG in this chat workspace outputs directory.
After it completes, inspect actual results/figure, write the exploratory
results report, and commit task changes locally. Do not duplicate the job
or claim its mathematical null prediction as an observed artifact result.

## Research state

| Exp | Question | Status |
|---|---|---|
| e01 | Controls match exact percolation theory | done |
| e02 | Grokking deep MLPs | done: H2′ passes; H1a, H1b-strict and H3-peak fail |
| e03 | Toy model, VPD + SAE sieve | done, three seeds: VPD sieve + causal pass, SAE fails |
| e04 | Transfer to Qwen3-0.6B, layers 2-3 | done, two seeds: **not transfer**, Q1 fails. Addendum: about 94% of the zero-ablation gap is the position-0 massive activation, and the causal effect is one attention-sink pair |
| e05 | Same question, position-aware mean-ablation baselines, layers 0-1 | local seed 0 failed the gate at 20k (KL 2.84 vs 1.94, plateau); seed 1 cancelled |
| e05b | e05 with an 8× batch (32 × 128) on a rented A6000 | both trained and copied; pod deleted; seed 0 gate passes, attribution job admitted |
| e06 | Null models for the sieve's z-curve: coverage vs alignment | pre-registered; e03 seeds 1-2 confirmatory: P1, P2, P4, P5 pass. Waiting on e05b graphs |

**Discovery findings behind e06** (`docs/e06_exploratory.md`): the small-c excess is
node-level alignment (Cov term) and the large-c deficit is coverage. The
joint-degree null is degenerate on these graphs, because backbone membership is
decided by degree. The in/out decoupling null still finds the excess. Strength
correlation is mostly firing frequency.

## Machine and repo notes

- **GPU:** the RTX 5080 laptop card, 16 GB, shared. Always go through vramorama
  leases.
  - If jobs crawl, check `nvidia-smi -q -d POWER`: a 15 W cap means MSI Center's
    User Scenario must go back to **Extreme Performance**.
  - Windows Update and charger power flaps have killed runs before; the watchdog
    drivers handle stalls.
- **param-decomp fork:** on the user's `rust-accel-scoring` branch. e04/e05
  register the Qwen3 topology schema at runtime (`e04_common.install_qwen3_schema`),
  so no fork changes are needed. The local branch `feature/qwen3-topology` is
  unneeded and unpushed.
- **Ruled out for VPD training:** the May 2026 Rust "CPU/RAM" fork (forward-only
  scoring), CPU training, the NPU, a 1-bit target, and gradient accumulation (no
  longer supported by param-decomp).
- **Visibility:** the user asked to make the repo public. The settings page was
  opened; whether they flipped it is unconfirmed.
- **Stray file:** `results/layer_ablation.json` is an untracked duplicate; safe to
  delete.
- **Commit convention:** commit and push at the end of each piece of work, with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
