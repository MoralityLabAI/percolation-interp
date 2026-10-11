# Cloud jobs

CPU-only work for Claude Code cloud sessions. One routine runs per job; each
session clones this repo, runs `cloud/setup.sh`, trains, and pushes small
result files (history JSON, never checkpoints) to its own `claude/...` branch.

Cloud VM facts that shape the jobs (October 2026):
- 4 vCPUs, 16 GB RAM, no GPU.
- A background command lasts at most 2 hours unless the environment sets
  `BASH_MAX_TIMEOUT_MS`.
- An idle VM can pause and lose running work, so a session must stay active
  until its runs finish.

Timings on CPU at width 128, with denormals flushed, are about 55 ms per step
at depth 8, so a 40k-step run takes roughly 40 minutes. Width 512 would take
hours per run, so cloud jobs stay at width 128 (or 256).

## Depth search (`docs/plan.md`, "Depth arm")

The prompt for setting `<S>`; `<FLAGS>` and `<DEPTHS>` come from the plan's
table:

```
You are running one job of the e02 depth hyperparameter search in this repo
(percolation-interp). Do exactly this and nothing else.

1. Run `bash cloud/setup.sh` from the repo root.
2. For each depth D in <DEPTHS>, one at a time:
   - Start this as a background command with a 2-hour timeout:
     .venv/bin/python experiments/e02_train.py --depths D --widths 128 --seeds 0 \
       --steps 40000 --device cpu <FLAGS> --progress 2000 \
       --export results/e02_search > logs/search_<S>_D<D>.log 2>&1
   - Keep the session active until it finishes, checking the log about every
     5 minutes (for example with the Monitor tool on the log file). Do not end
     your turn while training runs: an idle cloud VM can pause and kill it.
3. Commit only the new files under results/e02_search/ and logs/ to a new
   branch claude/e02-search-<S>, and push it. Change no other file. Do not open
   a pull request.
4. End with the line containing "final test acc" from each run.

If setup or training fails, push the logs only and report the error.
```
