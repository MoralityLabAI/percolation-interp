"""e04 driver: wait for the decomposition, check the quality gate, then run Q1-Q3's stages.

Stages run in order, each GPU stage under its own vramorama lease, logging to
out/e04/<stage>.log. A stage is skipped when its output already exists, so the driver can be
rerun after an interruption. If the gate fails (docs/e04_design.md), the driver writes
out/e04/gate.json and stops: Q1-Q3 are not scored.

  python experiments/e04_pipeline.py
(run with the param-decomp venv's python; the analysis stage uses this repo's venv)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from e04_common import OUT, PD_OUT, ROOT, STEPS, find_decomposition

VRAMORAMA = r"C:\Users\patri\AppData\Local\Programs\vramorama\vramorama.exe"
PD_PY = sys.executable  # param-decomp venv
LOCAL_PY = str(ROOT / ".venv" / "Scripts" / "python.exe")  # analysis (numpy, CPU)
GATE = 0.10
GATE_KEY = "ce_kl/ce_unrecovered_ci_masked"


def run(stage, cmd, gpu=None):
    log = OUT / f"{stage}.log"
    if gpu:
        cmd = [VRAMORAMA, "run", "--vram", gpu, "--owner", f"e04-{stage}", "--"] + cmd
    print(f"[{stage}] {' '.join(cmd)}", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=ROOT, env=dict(os.environ), stdout=f, stderr=subprocess.STDOUT, check=True)


def wait_until_complete(run_dir: Path, stall_minutes: int = 45):
    """Block until the final checkpoint exists (the run trains in another process)."""
    final = run_dir / f"model_{STEPS}.pth"
    while not final.exists():
        age = time.time() - max(p.stat().st_mtime for p in [run_dir, *run_dir.iterdir()])
        if age > 60 * stall_minutes:
            raise RuntimeError(f"{run_dir.name} has not progressed for {age / 60:.0f} minutes")
        print(f"{time.strftime('%H:%M')} waiting for {run_dir.name}", flush=True)
        time.sleep(300)
    time.sleep(60)  # let the save finish


def gate(run_dir: Path) -> dict:
    evals = [r for r in map(json.loads, (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines())
             if GATE_KEY in r]
    final = max(evals, key=lambda r: r["step"])
    assert final["step"] == STEPS, f"last evaluation is at step {final['step']}, not {STEPS}"
    res = {"run_id": run_dir.name, "step": final["step"], "threshold": GATE, GATE_KEY: final[GATE_KEY],
           "pass": bool(final[GATE_KEY] <= GATE),
           "ce_kl": {k: v for k, v in final.items() if k.startswith("ce_kl/")},
           "trajectory": [(r["step"], r[GATE_KEY]) for r in sorted(evals, key=lambda r: r["step"])]}
    (OUT / "gate.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run_dir = find_decomposition()
    assert run_dir is not None, f"no e04 decomposition under {PD_OUT}"
    wait_until_complete(run_dir)
    g = gate(run_dir)
    print(f"gate: {GATE_KEY} {g[GATE_KEY]:.4f} (threshold {GATE}): {'pass' if g['pass'] else 'FAIL'}", flush=True)
    if not g["pass"]:
        print("gate failed: Q1-Q3 are not scored", flush=True)
        return
    exp = str(ROOT / "experiments")
    if not (OUT / "stats.pt").exists():
        run("stats", [PD_PY, "-u", f"{exp}/e04_stats.py"], "5G")
    if not (OUT / "dataset_attributions.pt").exists():
        run("attributions", [PD_PY, "-u", f"{exp}/e04_attributions.py"], "8G")
    if not (OUT / "analysis.json").exists():
        run("analyze", [LOCAL_PY, "-u", f"{exp}/e04_analyze.py", "--n-shuffles", "100"])
    if not (OUT / "causal.json").exists():
        run("causal", [PD_PY, "-u", f"{exp}/e04_causal.py"], "5G")
    print("e04: all stages done", flush=True)


if __name__ == "__main__":
    main()
