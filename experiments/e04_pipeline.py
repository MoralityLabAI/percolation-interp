"""e04 driver: train the decomposition, check the quality gate, then run Q1-Q3's stages.

Unattended: every GPU job runs under its own vramorama lease and a watchdog. A job whose log
stops growing (a wedged CUDA context after a power flap did this once) or that exits with an
error is stopped, process tree and all, and started again: the decomposition up to three
times (from scratch: param-decomp cannot resume), each later stage twice. Logs go to
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

from e04_common import OUT, ROOT, STEPS, find_decomposition

VRAMORAMA = r"C:\Users\patri\AppData\Local\Programs\vramorama\vramorama.exe"
PD = Path(r"C:\Users\patri\Documents\GitHub\MoralityLabAI\param-decomp")
PD_PY = sys.executable  # param-decomp venv
LOCAL_PY = str(ROOT / ".venv" / "Scripts" / "python.exe")  # analysis (numpy, CPU)
CONFIG = Path(os.environ.get("E04_CONFIG", ROOT / "configs" / "e04" / "vpd_qwen3_0.6b_L2-3.yaml"))
GATE = 0.10
GATE_KEY = "ce_kl/ce_unrecovered_ci_masked"


def log(msg):
    print(f"{time.strftime('%m-%d %H:%M')} {msg}", flush=True)


def kill_tree(pid):
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)


def watched(stage, cmd, gpu, stall_minutes, attempts, poll=30):
    """Run cmd under a vramorama lease, restarting it if its log stalls or it fails.

    The stall clock starts once the lease is admitted (waiting in the queue is not a stall)
    and ignores gaps between polls much longer than `poll` (the machine slept).
    """
    for attempt in range(1, attempts + 1):
        path = OUT / (f"{stage}.log" if attempt == 1 else f"{stage}_attempt{attempt}.log")
        full = [VRAMORAMA, "run", "--vram", gpu, "--owner", f"e04-{stage}", "--"] + cmd if gpu else cmd
        log(f"[{stage}] attempt {attempt}: {' '.join(full)}")
        with open(path, "w", encoding="utf-8") as f:
            proc = subprocess.Popen(full, cwd=ROOT, env=dict(os.environ), stdout=f, stderr=subprocess.STDOUT)
            size, quiet, t_prev = -1, 0.0, time.time()
            while proc.poll() is None:
                time.sleep(poll)
                now = time.time()
                text = path.read_text(encoding="utf-8", errors="replace")
                admitted = gpu is None or "admitted" in text
                if now - t_prev > 4 * poll or not admitted or path.stat().st_size != size:
                    quiet = 0.0
                else:
                    quiet += now - t_prev
                size, t_prev = path.stat().st_size, now
                if quiet > 60 * stall_minutes:
                    log(f"[{stage}] no output for {quiet / 60:.0f} minutes: stopping it")
                    kill_tree(proc.pid)
                    proc.wait()
                    break
        if proc.returncode == 0:
            log(f"[{stage}] done")
            return
        log(f"[{stage}] attempt {attempt} ended with code {proc.returncode}")
    raise RuntimeError(f"{stage} failed after {attempts} attempts")


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


def finished() -> Path | None:
    d = find_decomposition()
    return d if d is not None and (d / f"model_{STEPS}.pth").exists() else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if finished() is None:
        watched("vpd_L2-3", [PD_PY, "-u", str(PD / "param_decomp" / "experiments" / "lm" / "lm_decomposition.py"),
                        str(CONFIG)], "12G", stall_minutes=20, attempts=3)
        time.sleep(30)
    run_dir = finished()
    assert run_dir is not None, "the decomposition finished but its final checkpoint is missing"
    g = gate(run_dir)
    log(f"gate ({run_dir.name}): {GATE_KEY} {g[GATE_KEY]:.4f} (threshold {GATE}): {'pass' if g['pass'] else 'FAIL'}")
    if not g["pass"]:
        log("gate failed: Q1-Q3 are not scored")
        return
    exp = str(ROOT / "experiments")
    if not (OUT / "stats.pt").exists():
        watched("stats", [PD_PY, "-u", f"{exp}/e04_stats.py"], "4G", stall_minutes=30, attempts=2)
    if not (OUT / "dataset_attributions.pt").exists():
        # The same 800 held-out rows as 200 batches of 4, in 1/3 of the time; peaks at 12.6 GB.
        watched("attributions", [PD_PY, "-u", f"{exp}/e04_attributions.py", "--batch", "16", "--n-batches", "50"],
                "13G", stall_minutes=30, attempts=2)
    if not (OUT / "analysis.json").exists():
        watched("analyze", [LOCAL_PY, "-u", f"{exp}/e04_analyze.py", "--n-shuffles", "100"], None,
                stall_minutes=24 * 60, attempts=1)
    if not (OUT / "causal.json").exists():
        watched("causal", [PD_PY, "-u", f"{exp}/e04_causal.py"], "4G", stall_minutes=30, attempts=2)
    log("e04: all stages done")


if __name__ == "__main__":
    main()
