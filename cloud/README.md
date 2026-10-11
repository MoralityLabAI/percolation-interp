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

Three cloud sessions, each running two of the plan's six settings, about 2-3
hours each. Start each from the desktop app (Cloud, repo
`MoralityLabAI/percolation-interp`, branch `master`), with `<S1> <S2>` set to
`A B`, `C D` or `E F`:

```
Run part of the e02 depth hyperparameter search in this repo (percolation-interp),
settings <S1> and <S2>. Do exactly this and nothing else.

1. Run `bash cloud/setup.sh` from the repo root.
2. For each setting S in <S1>, <S2>, one at a time, start this as a background
   command with a 2-hour timeout (7200000 ms):
     .venv/bin/python experiments/e02_search.py --settings S --device cpu --progress 2000 > logs/search_S.log 2>&1
   Stay active until it finishes, checking the log about every 5 minutes (for
   example with the Monitor tool on the log file). Do not end your turn while
   training runs: an idle cloud VM can pause and kill it.
3. Run: .venv/bin/python experiments/e02_search.py --report > logs/search_report_<S1><S2>.txt
4. Commit only new files under results/e02_search/ and logs/ to a new branch
   claude/e02-search-<S1><S2> and push it. Change no other file. Do not open a
   pull request.
5. Finish with the report output and the line containing "final test acc" from
   each run.

If setup or training fails, push the logs only and report the error.
```

Afterwards, locally: fetch the three branches, merge `results/e02_search/`, and run
`python experiments/e02_search.py --report` to apply the selection rule.
