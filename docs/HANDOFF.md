# Handoff, 2026-10-24 03:10 Singapore

## Current blocker, 2026-10-24 03:10 Singapore

E08's reviewed tokenizer repair was accepted and its queued request admitted
around 23:29 October 23. The collector then failed before saving a context
row: the saved seed-0 final checkpoint path did not exist. At this check the
entire C:\Users\patri\param_decomp_out directory is absent, including both
expected run directories. No VRAMorama lease or e08 Python worker remains.
Do not resubmit a driver until the saved artifacts are found and verified.

Read-only filename searches across the user profile and local shared storage
found neither final checkpoint or run ID; Recycle Bin original-path metadata
had no match. Only C: is mounted. This does not prove deletion or rule out a
renamed archive, disconnected drive or backup. The cause is unknown. A human
question requests the relocated/backup location. No new training, rentals,
downloads or cloud calls were made. Evidence and search limits are recorded
in results/e08/missing_artifacts_20261024.json.

The completed e05b/e06 records and all 12 e08 weighted-map results are still
present. Reuse them. The paid pod's earlier deletion remains confirmed; the
new issue is local artifact availability. Keep four-hour checks sparse and
avoid repeated notifications or broad rescans while awaiting the location.
If a copy becomes available, verify both final sizes and SHA256s against
results/e05b_retrieval.json before resuming through VRAMorama. Preserve the
frozen design and applied implementation manifest. No semantic e08 outcome
has been measured, and the task is not complete.

Snapshot of everything in flight for the next session, human or agent. The repo
is `MoralityLabAI/percolation-interp` (GitHub, branch `master`); the local checkout
is `C:\Users\patri\Documents\GitHub\MoralityLabAI\percolation-interp`.

Read `docs/e05_design.md`, `docs/e05b_design.md` and `docs/e06_design.md` before
touching anything below. All three are pre-registered. Never change a criterion
after seeing results; log deviations in the design doc.

## Current e08 state, 2026-10-23 late check

The weighted all-energy stage completed all 12 maps (32 draws each) in
about 296 s. Raw files are in results/e08/weighted.json and weighted/;
retain and reuse them. The following context stage exited before any row
was saved because the tokenizer library attempted a model-ID metadata
lookup that offline mode blocked. The e08 driver and worker were confirmed
absent before repair/resubmission; do not wait on old handles ae19d485.

The collector now loads the identical cached snapshot by local path, with
HTTP still forbidden. Full suite: 89 tests pass; actual tokenizer preflight
also passes with HTTP/model_info calls forbidden. Design/sampling settings
are unchanged. See docs/e08_implementation_notes.md and the exact receipt
results/e08/implementation_repair_20261023_offline_tokenizer.json.
Resume the same driver with --reviewed-repair pointing at that receipt;
the original frozen manifest is preserved and exact before/after hashes
are checked. Do not bypass or silently overwrite a manifest mismatch.

Another active phb-h1-qwen3-4b job owns most GPU memory and its power cap
was 15 W; leave it untouched and respect guards. E08 can queue. Continue
four-hour agent checks, and require blinded review before any effects.
E05b/e06 results remain final and committed (3158641, release c9295b0).
The paid pod is deleted; no rentals, downloads, retraining or pushes.

## Prior state: e05b/e06 complete; e08 ready for release

Final local review on 2026-10-23 confirms both e05b gates and Q1-Q4 pass.
The fixed transfer rule is met. Backbones have 129/115 components and retain
61.61%/63.20% of the decomposition's token-dependent KL gap. Full report:
docs/e05b_results.md; gate/sieve/causal raw records are in results/e05b.
Both final attribution archives passed full ZIP CRC validation and have
SHA256/size manifests there. Seed-1 recovery completed successfully.

All four e06 graphs are complete. P1-P3 hold across applicable graphs; P4
has no failing testable cell (Qwen seed0 is untestable); P5 fails because
Qwen seed0 retains .531335 of raw correlation after rate control, versus
.487233 for seed1. The reported-only N-AB control gives p about .48 on
both Qwen graphs. Report: docs/e06_results.md; full scores and raw records
are in results/e06. Do not describe these selected sets as proven wiring
circuits or interpret the P5 threshold failure as a semantic mechanism.

The local resume supervisor has exited successfully. The only current
workflow lease is e08-driver ae19d485, launcher14144/job31640; it is waiting
for the reviewed-results commit marker. After committing this reviewed
report/results, write results/e08/prerequisites_reviewed.json naming the
verified local e06_commit. Preserve the live waiting driver; do not duplicate
or restart it. It will launch its own stages when its checks pass.
No e08 artifact outcome has been claimed yet. Continue four-hour checks,
respect guards, keep all saved VPD files and never push/rent/retrain.

The snapshots below are historical; this current-state section supersedes
their pending-stage descriptions. The e07/PAMA work remains complete.

## Recovery check, 2026-10-23 11:08 Singapore

This section supersedes the earlier live-process snapshot below. At this
check no original e05/e06/e08 workers or leases remained. Seed 0 is complete:
gate and Q1-Q4 all pass. Seed 1 also passed its gate (KL 1.925512 versus
limit 1.943471), but its attribution serialization failed with a file-write
error after computing the batches. A 76,562,432-byte truncated file was
mistaken for a finished stage; the subsequent analyzer and supervisor exited
because its ZIP central directory was absent. The original cause of the
write failure is not proven; free disk is now about 16.8 GB.

The partial archive and failure logs were preserved in out/e05b_seed1 with
failed_20261023T111334 suffixes. Do not use them as scientific results.
The harvest now checks free space before loading a model, writes to a
temporary archive, validates every ZIP member CRC, flushes and atomically
publishes only a complete file. Samples, batch16 x50, seeds and scientific
criteria are unchanged. Full CPU suite: 87 tests pass, including interrupted
writes, CRC failure, disk preflight and real Torch serialization.

A new local-only resume helper, work/e05b_resume_analysis.py in the Codex
chat workspace, was launched through VRAMorama owner e05b-resume (initial
launcher 22660). It calls only the original helper's --seed entry, skips
completed seed-0 stages, reruns missing seed-1 attribution/analysis/causal
stages and finishes e06. It never calls retrieval/API code or the training-
capable e05 driver. Logs: out/e05b/resume_driver.log and appended
out/e05b/followthrough.log. Verify its live handles before any resubmission.
The e08 waiting driver was also restored only after verifying its previous
lease and worker were absent (new initial launcher 14144). No e08 model
run has started; its predecessor-review marker remains absent.

An unrelated active llama-server is using host RAM; leave it untouched.
Admission may queue behind VRAMorama's normal memory guards. Continue sparse
four-hour checks. Both final paid checkpoints remain local and unchanged;
no network, rental or training was performed during this recovery.

## Live processes (earlier snapshot)

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

- The original e05b driver is running under VRAMorama. Seed 0 completed
  all 50 attribution batches (102,400 tokens) at 08:43 and saved a
  1,323,117,113-byte dataset_attributions.pt. The 1,000-shuffle sieve
  completed at 09:31 with Q1 passing. Its causal stage is active under
  lease 8865b99c; 24/32 batches were logged at this check. Seed 1 has
  not yet reached its gate in the sequential driver.
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


### 3. e05b local analysis (seed 0 Q1 passes; causal running)

Seed 0 passed the gate: CI-masked KL 1.763079 versus limit 1.943471
(KL_PAMA 7.773885); stats.pt is present. Seed 1 gate is still pending.
Seed 0 has completed attribution/sieve analysis and passes Q1; Q2-Q4
remain pending causal output. This is a partial result, not an e05b
across-seed verdict. For e06, both graphs count even if
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

## e07: component/neuron percolation exploration (complete)

Completed the exploratory study requested by the human using both saved VPD
artifacts. Definitions: `docs/e07_math.md`; fixed protocol:
`docs/e07_design.md`; observed results: `docs/e07_results.md`. All 12 MLP
coordinate maps were measured. Degree-preserving rewiring reproduces the
learned mean percolation curves within 0.0243 largest-cluster fraction on
the fixed grid; 50%-cluster crossovers differ by at most 0.00197.
Layer-0 gate writers cross near p=0.15, compared with p=0.33 for uniform
contacts. This is structural sharing, not measured semantic polysemanticity;
the top-eight contacts retain only 2.72-3.64% of median direction energy.

The original e07 bookkeeping lease f7c60a48 was cancelled while still
waiting. A CPU-only NumPy reader completed the unchanged protocol in 120 s,
under a 512 MiB process-commit cap (265.5 MiB measured before plotting), without
CUDA/PyTorch model loading or any GPU guard change. The prior GPU harvest
continued. 70 tests pass, including independent graph calculations and
archive-reader compatibility. Raw measurements: `results/e07/geometry.json`;
figures: `results/e07/overview.png` and `curves.png`. No e07 compute is left
queued. The e05b/e06 pipeline is separate and remains in progress.

## e08: implemented, deferred until e05b/e06 finishes

The human authorized local execution using saved artifacts after e05b/e06.
Protocol: `docs/e08_design.md`; native deferred driver:
`experiments/e08_pipeline.py --wait-for-e05b-e06`. It uses VRAMorama, never
trains/rents/downloads, and refuses to compete with the original workers.
Before releasing it, finish/review e05b/e06, write `docs/e06_results.md`,
commit that work locally, and create `results/e08/prerequisites_reviewed.json`
with the verified local commit in `e06_commit`. The driver checks actual
worker exits as well as result files and the prior supervisor's success.

Then it runs the fixed all-energy geometry and both context harvests. It
pauses with `out/e08/status.json` state `awaiting_blinded_review` if
`out/e08/labels.json` is absent. Review `out/e08/blinded_contexts.json` only,
following the frozen annotation rules; do not inspect contribution groups
or intervention effects to choose labels. A single agent reviewer is not
independent human validation. Resume the same driver through VRAMorama once
the previous driver has exited. The OS lock prevents duplicate drivers.
The first deferred launch (b830d0ea) exited before starting Python because
its log directory did not exist. The directory was created and absence
of that old lease/worker was verified before one corrected launch.
The corrected driver is admitted as lease f99d2fa9 (launcher 33836,
job 42444, Python 55544); out/e08/status.json confirms it is waiting
for e05b/e06, with no e08 model run started. Preserve this live driver.
Final outputs: `results/e08/contextual.json`, `weighted.json`, a figure and
`docs/e08_results.md`; review them before committing/reporting completion.
Preserve the four-hour model follow-up interval and all existing science
criteria. Never push or rent; leave unrelated files untouched.

## Research state

| Exp | Question | Status |
|---|---|---|
| e01 | Controls match exact percolation theory | done |
| e02 | Grokking deep MLPs | done: H2′ passes; H1a, H1b-strict and H3-peak fail |
| e03 | Toy model, VPD + SAE sieve | done, three seeds: VPD sieve + causal pass, SAE fails |
| e04 | Transfer to Qwen3-0.6B, layers 2-3 | done, two seeds: **not transfer**, Q1 fails. Addendum: about 94% of the zero-ablation gap is the position-0 massive activation, and the causal effect is one attention-sink pair |
| e05 | Same question, position-aware mean-ablation baselines, layers 0-1 | local seed 0 failed the gate at 20k (KL 2.84 vs 1.94, plateau); seed 1 cancelled |
| e05b | e05 with an 8× batch (32 × 128) on a rented A6000 | complete: both gates and Q1-Q4 pass; transfer rule met; pod deleted |
| e06 | Null models for the sieve's z-curve: coverage vs alignment | complete: P1-P3 hold; P4 has no failing testable cell; P5 fails on Qwen seed 0 |
| e07 | Bipartite percolation of VPD component/neuron sharing | done, both saved seeds: contact degree concentration explains most early structural mixing; semantic labels not measured |

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
