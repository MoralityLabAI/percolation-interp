"""e05 driver (docs/e05_design.md): for each seed, decomposition, gate, then Q1-Q4's stages.

  python experiments/e05_pipeline.py --seeds 0 1 [--steps 40000]      (param-decomp venv; launch detached)

--steps 40000 runs the design's retrain for a seed that failed its gate at 20k, with its own
config (configs/e05/..._seed<N>_40k.yaml) and outputs (out/e05_seed<N>_40k/).

Each seed runs in its own process (e04_common reads its settings from the environment at
import), in order. Per seed, logs and outputs go to out/e05_seed<N>/. Every GPU job runs under a
vramorama lease and e04_pipeline's watchdog. A stage is skipped when its output exists, so the
driver can be rerun. If a seed fails its gate, its later stages are skipped (the design's 40k
retrain is then launched by hand) and the next seed still runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLICE = "0,1"


def seed_env(seed: int, steps: int = 20000, batch: int | None = None) -> dict:
    """--batch N (e05b): decompositions trained elsewhere at batch N, outputs in out/e05b_seed<N>/."""
    tag = "" if steps == 20000 else f"_{steps // 1000}k"
    exp_name = "e05"
    env = dict(os.environ, E04_SEED=str(seed), E04_STEPS=str(steps), E04_SLICE=SLICE)
    if batch is not None:
        tag, exp_name = f"_b{batch}", "e05b"
        env["E04_BATCH"] = str(batch)
    env["E04_OUT"] = str(ROOT / "out" / f"{exp_name}_seed{seed}{'' if batch else tag}")
    env["E04_CONFIG"] = str(ROOT / "configs" / "e05" / f"vpd_qwen3_0.6b_L0-1_seed{seed}{tag}.yaml")
    return env


def run_seed():
    import time

    from e04_common import OUT
    from e04_pipeline import CONFIG, LOCAL_PY, PD, PD_PY, finished, log, watched

    OUT.mkdir(parents=True, exist_ok=True)
    exp = str(ROOT / "experiments")
    if finished() is None:
        watched("vpd", [PD_PY, "-u", str(PD / "param_decomp" / "experiments" / "lm" / "lm_decomposition.py"),
                        str(CONFIG)], "12G", stall_minutes=20, attempts=3)
        time.sleep(30)
    assert finished() is not None, "the decomposition finished but its final checkpoint is missing"
    if not (OUT / "gate.json").exists():
        watched("gate", [PD_PY, "-u", f"{exp}/e05_gate.py"], "5G", stall_minutes=30, attempts=2)
    g = json.loads((OUT / "gate.json").read_text())
    log(f"gate ({g['run_id']}): KL(CI-masked) {g['kl_ci']:.4f} vs 0.25 x KL_PAMA {0.25 * g['kl_pama']:.4f}: "
        f"{'pass' if g['pass'] else 'FAIL'}")
    if not g["pass"]:
        log("gate failed: Q1-Q4 are not scored for this seed")
        return
    if not (OUT / "stats.pt").exists():
        watched("stats", [PD_PY, "-u", f"{exp}/e04_stats.py"], "4G", stall_minutes=30, attempts=2)
    if not (OUT / "dataset_attributions.pt").exists():
        watched("attributions", [PD_PY, "-u", f"{exp}/e04_attributions.py", "--batch", "16", "--n-batches", "50"],
                "13G", stall_minutes=30, attempts=2)
    if not (OUT / "analysis.json").exists():
        watched("analyze", [LOCAL_PY, "-u", f"{exp}/e05_analyze.py", "--n-shuffles", "1000"], None,
                stall_minutes=24 * 60, attempts=1)
    if not (OUT / "causal.json").exists():
        watched("causal", [PD_PY, "-u", f"{exp}/e05_causal.py"], "6G", stall_minutes=30, attempts=2)
    log("all stages done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=None, help="e05b: batch size of the (remotely trained) decomposition")
    ap.add_argument("--one-seed", action="store_true", help="internal: run the seed given by E04_SEED")
    args = ap.parse_args()
    if args.one_seed:
        run_seed()
        return
    for s in args.seeds:
        print(f"=== seed {s}", flush=True)
        r = subprocess.run([sys.executable, "-u", __file__, "--one-seed"], env=seed_env(s, args.steps, args.batch), cwd=ROOT)
        print(f"=== seed {s} exited with code {r.returncode}", flush=True)


if __name__ == "__main__":
    main()
