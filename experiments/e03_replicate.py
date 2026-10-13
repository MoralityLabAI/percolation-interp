"""e03 replication driver: rerun every decomposition stage with a new seed on the same target.

Stages run in order, each GPU stage under its own vramorama lease, logging to
out/e03_seed<N>/<stage>.log. A stage is skipped when its output already exists, so the driver
can be rerun after an interruption. Pre-registered predictions: docs/e03_replication.md.

  python experiments/e03_replicate.py --seed 1 2
(run with the param-decomp venv's python; the analysis stage uses this repo's venv)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PD_OUT = Path.home() / "param_decomp_out"
VRAMORAMA = r"C:\Users\patri\AppData\Local\Programs\vramorama\vramorama.exe"
PD_PY = sys.executable  # param-decomp venv
LOCAL_PY = str(ROOT / ".venv" / "Scripts" / "python.exe")  # analysis (numpy/scipy/torch CPU)


def run(stage, cmd, out, gpu_mib=None):
    log = out / f"{stage}.log"
    if gpu_mib:
        cmd = [VRAMORAMA, "run", "--vram", f"{gpu_mib}M", "--owner", f"e03-rep-{stage}", "--"] + cmd
    print(f"[{stage}] {' '.join(cmd)}", flush=True)
    env = dict(os.environ, E03_OUT=str(out))
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


VPD_STEPS = 40000


def find_decomposition(seed: int, target_dir: str) -> Path | None:
    """The decomposition of this target with this seed, identified by its final_config.yaml.

    Several seeds may train at once, so a run is never identified by which folder appeared last.
    """
    for d in sorted((PD_OUT / "decompositions").glob("*")):
        cfg = d / "final_config.yaml"
        if cfg.exists():
            text = cfg.read_text(encoding="utf-8")
            if re.search(rf"(?m)^seed: {seed}$", text) and target_dir in text:
                return d
    return None


def wait_until_complete(run_dir: Path, stall_minutes: int = 45):
    """Block until the run's final checkpoint exists (it may be training in another process)."""
    final = run_dir / f"model_{VPD_STEPS}.pth"
    metrics = run_dir / "metrics.jsonl"
    while not final.exists():
        age = time.time() - max(p.stat().st_mtime for p in [run_dir, *run_dir.iterdir()])
        if age > 60 * stall_minutes:
            raise RuntimeError(f"{run_dir.name} has not progressed for {age / 60:.0f} minutes")
        print(f"waiting for {run_dir.name} ({metrics.stat().st_size if metrics.exists() else 0} bytes of metrics)",
              flush=True)
        time.sleep(300)
    time.sleep(30)  # let the final save finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, nargs="+", required=True, help="one or more seeds, run in order")
    args = ap.parse_args()
    for s in args.seed:
        run_seed(s)


def run_seed(s):
    out = ROOT / "out" / f"e03_seed{s}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "out" / "e03" / "target.txt", out / "target.txt")
    exp = str(ROOT / "experiments")

    if not (out / "sae_tower.pt").exists():
        run("sae_tower", [PD_PY, "-u", f"{exp}/e03_sae_tower.py", "--device", "cuda", "--seed", str(s)], out, 2048)
    if not (out / "sae_graph.pt").exists():
        run("sae_graph", [PD_PY, "-u", f"{exp}/e03_sae_graph.py", "--device", "cuda", "--n-batches", "16"], out, 2048)
    if not (out / "vpd_run.txt").exists():
        target_dir = Path((out / "target.txt").read_text().strip()).parent.parent.name
        found = find_decomposition(s, target_dir)
        if found is None:
            run("vpd", [PD_PY, "-u", f"{exp}/e03_vpd.py", "decompose", "--seed", str(s)], out, 6144)
            found = find_decomposition(s, target_dir)
            assert found is not None, f"no decomposition with seed {s} after the VPD stage"
        wait_until_complete(found)
        run_id = found.name
        cache = PD_OUT / "runs" / f"param-decomp-{run_id}"
        if not cache.exists():
            shutil.copytree(PD_OUT / "decompositions" / run_id, cache)
        (out / "vpd_run.txt").write_text(run_id)
    if not (out / "harvest_subrun.txt").exists():
        run("harvest", [PD_PY, "-u", f"{exp}/e03_vpd.py", "harvest", "--n-batches", "300", "--batch-size", "16"],
            out, 4096)
    run_id = (out / "vpd_run.txt").read_text().strip()
    if not list((PD_OUT / "dataset_attributions" / run_id).glob("da-*/dataset_attributions.pt")):
        run("attributions", [PD_PY, "-u", f"{exp}/e03_vpd.py", "attributions", "--n-batches", "100",
                             "--batch-size", "16", "--subrun", "da-e03b"], out, 6144)
    if not (out / "crossref.pt").exists():
        run("crossref", [PD_PY, "-u", f"{exp}/e03_crossref.py", "--device", "cuda", "--n-batches", "64"], out, 3072)
    if not (out / "analysis.json").exists():
        run("analyze", [LOCAL_PY, "-u", f"{exp}/e03_analyze.py", "--n-shuffles", "100", "--n-draws", "1000",
                        "--strength-null"], out)
    if not (out / "causal.json").exists():
        run("causal", [PD_PY, "-u", f"{exp}/e03_causal.py", "--device", "cuda"], out, 3072)
    if not (out / "causal_in_strength.json").exists():
        run("causal_in_strength", [PD_PY, "-u", f"{exp}/e03_causal.py", "--device", "cuda",
                                   "--sae-backbone", "sae_sieve_in_strength", "--out-name", "causal_in_strength.json"],
            out, 3072)
    print(f"seed {s}: all stages done", flush=True)


if __name__ == "__main__":
    main()
