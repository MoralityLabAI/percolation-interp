# Handoff, 2026-10-21 16:00

Snapshot of everything in flight for the next session, human or agent. The repo
is `MoralityLabAI/percolation-interp` (GitHub, branch `master`); the local checkout
is `C:\Users\patri\Documents\GitHub\MoralityLabAI\percolation-interp`.

Read `docs/e05_design.md`, `docs/e05b_design.md` and `docs/e06_design.md` before
touching anything below. All three are pre-registered. Never change a criterion
after seeing results; log deviations in the design doc.

## Live processes

### 1. e05b: VPD training on a Prime Intellect pod (costs money)

- **Pod:** `a818c7656db149eba8122c9956b20f85`
  - RTX A6000 48 GB, MassedCompute, $0.54/h, rented 2026-10-21 about 08:05.
  - SSH: `ssh -i ~/.ssh/id_ed25519_prime ubuntu@64.247.206.119`
- **What's running:**
  - `~/param-decomp` with its own `.venv`: Python 3.13, packages pinned to the local
    versions, e.g. transformers 4.57.3 and wandb 0.23.1.
  - Seed 0: config `~/e05b/vpd_qwen3_0.6b_L0-1_seed0_b32.yaml`, log `~/e05b/seed0.log`,
    run dir `~/param_decomp_out/decompositions/p-7e116244`. At 15:59 it was at step
    18,664/20,001, 1.42 s/step, so done about 16:30.
  - Seed 1 starts automatically when seed 0's Python process exits: a `bash` loop
    waiting on PID 4740 runs the `seed1_b32` config, log `~/e05b/seed1.log`.
    About 7.9 h, so done about 00:30 Thursday.
  - `p-bad7e278` on the pod is the dead batch-64 run (out of memory); ignore it.
- **Check progress:**

      ssh -i ~/.ssh/id_ed25519_prime ubuntu@64.247.206.119 'tail -c 300 ~/e05b/seed1.log'

- **Budget:** wallet $25.41 at the start; about $10 expected in total. The user
  approved spending it on this.

### 2. Local guards for the pod, detached on this PC

Run with the Prime CLI venv's Python, `C:\Users\patri\.venvs\prime\Scripts\python.exe`.

- **`cloud/pod_watchdog.py`** deletes the pod at the Unix time in
  `out/e05b/pod_deadline.txt`, currently **Thu 2026-10-22 02:30**. Log:
  `out/e05b/pod_watchdog_2.log`.
- **`cloud/e05b_fetch.py`** polls the pod every 10 min. When both seeds have
  `model_20000.pth`, or training stops with at least one finished, it:
  - copies the run dirs to `~/param_decomp_out/decompositions/`;
  - checks the checkpoint sizes;
  - deletes the pod;
  - starts the local analysis: `experiments/e05_pipeline.py --seeds .. --batch 32`,
    detached, log `out/e05b/pipeline.log`.

  Fetcher log: `out/e05b/fetch_2.log`.
- **Pitfall:** both guards died once when the Claude session restarted (12:03 today)
  and were relaunched. **After any restart, check they're alive:**

      powershell "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { $_.CommandLine -like '*pod_watchdog*' -or $_.CommandLine -like '*e05b_fetch*' }"

  Relaunch commands:

      vramorama run --vram 64M --owner e05b-pod-watchdog --detach --out <repo>/out/e05b/pod_watchdog_3.log -- C:/Users/patri/.venvs/prime/Scripts/python.exe -u <repo>/cloud/pod_watchdog.py a818c7656db149eba8122c9956b20f85 <repo>/out/e05b/pod_deadline.txt
      vramorama run --vram 64M --owner e05b-fetch --detach --out <repo>/out/e05b/fetch_3.log -- C:/Users/patri/.venvs/prime/Scripts/python.exe -u <repo>/cloud/e05b_fetch.py a818c7656db149eba8122c9956b20f85 ubuntu@64.247.206.119

  `vramorama` is `C:\Users\patri\AppData\Local\Programs\vramorama\vramorama.exe`.
- **Manual pod control:**

      C:/Users/patri/.venvs/prime/Scripts/python.exe cloud/prime_pod.py status|delete a818c7656db149eba8122c9956b20f85

  Set `PRIME_DISABLE_VERSION_CHECK=1 PYTHONIOENCODING=utf-8` first. If the CLI's
  auth has expired, run `prime login --headless` and open its link in **Firefox**
  (the user's preference, not the in-app browser).

### 3. e05b local analysis (starts automatically after the fetch, about 00:40)

Per seed, under vramorama leases with watchdogs (the e05 driver):
1. Gate: `e05_gate.py`, KL(CI-masked) ≤ 0.25 × KL_PAMA. If it fails, Q1-Q4 are not scored.
2. Stats.
3. Attributions: batch 16 × 50, about 1.75 h, 13 GB.
4. Sieve: `e05_analyze.py`, 1,000 shuffles, both controls, plus Q4 null sets.
5. Causal: `e05_causal.py`, component-level mean ablation, Q2-Q4.

Outputs go to `out/e05b_seed0/` and `out/e05b_seed1/`, with `gate.json`,
`analysis.json` and `causal.json`. Expect each seed's verdict about 2-2.5 h after
the fetch.

### 4. e06 held-out run: a subagent working in a worktree

- **Worktree:** `.claude/worktrees/agent-aa23bb7da98c92697`, branch
  `worktree-agent-aa23bb7da98c92697`.
- **Committed:** `e7bd7e7`, the held-out runner (`experiments/e06_heldout.py`),
  `e06_score.py`, and the exact sequential swaps in `diagnostics.py`.
- **Uncommitted at 16:00:** `results/e06/heldout_e03_seed1.json`,
  `docs/e06_heldout_run.md`. The e03 seed 2 run was probably still going.
- **To finish:**
  1. Let it complete, or rerun seed 2 by hand with the documented command in
     `docs/e06_heldout_run.md`.
  2. Review the diff against master.
  3. Run the tests: `PYTHONPATH=<worktree>/src .venv/Scripts/python.exe -m pytest -q tests`.
  4. Merge the branch into master (`--no-ff`) and push.
  5. Remove the worktree.
- **e06 still needs the e05b graphs:** run the documented e05b command on
  `out/e05b_seed0` and `out/e05b_seed1` once their `dataset_attributions.pt` and
  `stats.pt` exist.
  - They count whatever their gate result; report the gate status.
  - Score P1-P5. P3 counts only for e05b; for e03 seeds 1-2 it is reported only.
  - Write `docs/e06_results.md`.

## Research state

| Exp | Question | Status |
|---|---|---|
| e01 | Controls match exact percolation theory | done |
| e02 | Grokking deep MLPs | done: H2′ passes; H1a, H1b-strict and H3-peak fail |
| e03 | Toy model, VPD + SAE sieve | done, three seeds: VPD sieve + causal pass, SAE fails |
| e04 | Transfer to Qwen3-0.6B, layers 2-3 | done, two seeds: **not transfer**, Q1 fails. Addendum: about 94% of the zero-ablation gap is the position-0 massive activation, and the causal effect is one attention-sink pair |
| e05 | Same question, position-aware mean-ablation baselines, layers 0-1 | local seed 0 failed the gate at 20k (KL 2.84 vs 1.94, plateau); seed 1 cancelled |
| e05b | e05 with an 8× batch (32 × 128) on a rented A6000 | **training**; see above |
| e06 | Null models for the sieve's z-curve: coverage vs alignment | pre-registered (`docs/e06_design.md`); held-out run in progress |

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
