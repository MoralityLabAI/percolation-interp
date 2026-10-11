"""e03 stages around param-decomp: record the target, run VPD, harvest, dataset attributions.

Runs in param-decomp's venv. Each subcommand prints what it ran; GPU stages are
meant to be launched under a vramorama lease.

  python experiments/e03_vpd.py target            # record the newest pretrained target
  python experiments/e03_vpd.py decompose         # VPD with configs/e03/vpd_ss_llama_2L128.yaml
  python experiments/e03_vpd.py register          # copy the run into the local runs cache
  python experiments/e03_vpd.py harvest --n-batches 2000
  python experiments/e03_vpd.py attributions --n-batches 2000
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from e03_common import OUT, ROOT

PD_OUT = Path.home() / "param_decomp_out"
VPD_TEMPLATE = ROOT / "configs" / "e03" / "vpd_ss_llama_2L128.yaml"


def newest(paths):
    paths = sorted(paths, key=lambda p: p.stat().st_mtime)
    if not paths:
        raise SystemExit("nothing found")
    return paths[-1]


def cmd_target(_):
    ckpt = newest((PD_OUT / "target_models").glob("*/checkpoints/model_step_*.pt"))
    (OUT / "target.txt").write_text(str(ckpt))
    print(f"target: {ckpt}")


def cmd_decompose(_):
    target = (OUT / "target.txt").read_text().strip()
    cfg = VPD_TEMPLATE.read_text(encoding="utf-8").replace("TARGET_CHECKPOINT", target.replace("\\", "/"))
    cfg_path = OUT / "vpd_config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    import param_decomp

    script = Path(param_decomp.__file__).parent / "experiments" / "lm" / "lm_decomposition.py"
    cmd = [sys.executable, "-u", str(script), str(cfg_path)]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def cmd_register(_):
    run_dir = newest([p for p in (PD_OUT / "decompositions").glob("*") if p.is_dir()])
    run_id = run_dir.name
    cache = PD_OUT / "runs" / f"param-decomp-{run_id}"
    if not cache.exists():
        shutil.copytree(run_dir, cache)
    (OUT / "vpd_run.txt").write_text(run_id)
    print(f"registered {run_dir} as {cache} (run id {run_id})")


def _run_id() -> str:
    return (OUT / "vpd_run.txt").read_text().strip()


def cmd_harvest(args):
    cfg = {"method_config": {"type": "ParamDecompHarvestConfig", "wandb_path": _run_id(),
                             "activation_threshold": args.ci_threshold},
           "n_batches": args.n_batches, "batch_size": args.batch_size}
    subrun = "h-e03"
    cmd = [sys.executable, "-u", "-m", "param_decomp.harvest.scripts.run_worker",
           "--config_json", json.dumps(cfg), "--subrun_id", subrun]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    (OUT / "harvest_subrun.txt").write_text(subrun)


def cmd_attributions(args):
    cfg = {"wandb_path": _run_id(), "n_batches": args.n_batches, "batch_size": args.batch_size,
           "ci_threshold": args.ci_threshold}
    harvest = (OUT / "harvest_subrun.txt").read_text().strip()
    subrun = "da-e03"
    worker = [sys.executable, "-u", "-m", "param_decomp.dataset_attributions.scripts.run_worker",
              "--wandb_path", _run_id(), "--config_json", json.dumps(cfg), "--harvest_subrun_id", harvest,
              "--rank", "0", "--world_size", "1", "--subrun_id", subrun]
    print("running:", " ".join(worker), flush=True)
    subprocess.run(worker, check=True)
    merge = [sys.executable, "-u", "-m", "param_decomp.dataset_attributions.scripts.run_merge",
             "--wandb_path", _run_id(), "--subrun_id", subrun]
    print("running:", " ".join(merge), flush=True)
    subprocess.run(merge, check=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("target")
    sub.add_parser("decompose")
    sub.add_parser("register")
    for name in ("harvest", "attributions"):
        p = sub.add_parser(name)
        p.add_argument("--n-batches", type=int, default=2000)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--ci-threshold", type=float, default=0.0)
    args = ap.parse_args()
    {"target": cmd_target, "decompose": cmd_decompose, "register": cmd_register,
     "harvest": cmd_harvest, "attributions": cmd_attributions}[args.cmd](args)


if __name__ == "__main__":
    main()
