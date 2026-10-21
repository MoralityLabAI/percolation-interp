"""e05b: wait for the pod's decompositions, copy them home, delete the pod, start the local analysis.

  python cloud/e05b_fetch.py POD_ID HOST      (prime CLI venv's python; launch detached)

Every 10 minutes it asks the pod which e05b decompositions (layers 0-1, batch 32, 20k steps)
have their final checkpoint, and whether training is still running. Once both seeds are
finished, or nothing is training any more, it:
1. copies every finished decomposition into ~/param_decomp_out/decompositions/ and checks
   the checkpoint sizes;
2. deletes the pod;
3. starts e05_pipeline.py --seeds <finished> --batch 32 detached under vramorama.
The deadline watchdog (cloud/pod_watchdog.py) still guards the budget if this script dies.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from prime_cli.api.pods import PodsClient
from prime_cli.client import APIClient

ROOT = Path(__file__).resolve().parents[1]
SSH_DIR = Path(os.environ["WINDIR"]) / "System32" / "OpenSSH"
KEY = Path.home() / ".ssh" / "id_ed25519_prime"
LOCAL = Path.home() / "param_decomp_out" / "decompositions"
VRAMORAMA = r"C:\Users\patri\AppData\Local\Programs\vramorama\vramorama.exe"
PD_PY = r"C:\Users\patri\Documents\GitHub\MoralityLabAI\param-decomp\.venv\Scripts\python.exe"
OPTS = ["-i", str(KEY), "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=30", "-o", "BatchMode=yes"]

# Prints one JSON object: finished seeds -> run dir (with its checkpoint size), and whether training runs.
PROBE = r"""
import glob, json, os, re, subprocess
out = {"done": {}, "training": bool(subprocess.run(["pgrep", "-f", "lm_decomposition.py"], capture_output=True).stdout)}
for d in glob.glob(os.path.expanduser("~/param_decomp_out/decompositions/*")):
    try: t = open(d + "/final_config.yaml").read()
    except OSError: continue
    if re.search(r"(?m)^batch_size: 32$", t) and re.search(r"(?m)^steps: 20000$", t) and "model.layers.0." in t:
        ck = d + "/model_20000.pth"
        if os.path.exists(ck):
            out["done"][re.search(r"(?m)^seed: (\d+)$", t).group(1)] = [d, os.path.getsize(ck)]
print(json.dumps(out))
"""


def log(msg):
    print(f"{time.strftime('%m-%d %H:%M')} {msg}", flush=True)


def ssh(host, cmd, **kw):
    return subprocess.run([str(SSH_DIR / "ssh.exe"), *OPTS, host, cmd], capture_output=True, text=True, **kw)


def main():
    pod_id, host = sys.argv[1], sys.argv[2]
    while True:
        r = ssh(host, "~/param-decomp/.venv/bin/python -", input=PROBE, timeout=120)
        try:
            state = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            log(f"probe failed: {r.stderr.strip()[:200]}")
            time.sleep(600)
            continue
        log(f"finished seeds {sorted(state['done'])}; training running: {state['training']}")
        if len(state["done"]) >= 2 or (state["done"] and not state["training"]):
            break
        if not state["done"] and not state["training"]:
            log("nothing finished and nothing training: leaving the pod to the deadline watchdog")
            return
        time.sleep(600)

    LOCAL.mkdir(parents=True, exist_ok=True)
    seeds = []
    for seed, (remote_dir, size) in sorted(state["done"].items()):
        name = remote_dir.rstrip("/").split("/")[-1]
        if (LOCAL / name / "model_20000.pth").exists():
            log(f"seed {seed}: {name} already here")
        else:
            c = subprocess.run([str(SSH_DIR / "scp.exe"), *OPTS, "-r", f"{host}:{remote_dir}", str(LOCAL)],
                               capture_output=True, text=True, timeout=3600)
            got = (LOCAL / name / "model_20000.pth").stat().st_size if (LOCAL / name / "model_20000.pth").exists() else -1
            log(f"seed {seed}: copied {name} (checkpoint {got} of {size} bytes) {c.stderr.strip()[:200]}")
            if got != size:
                log("copy incomplete: not deleting the pod; the deadline watchdog still applies")
                return
        seeds.append(seed)

    PodsClient(APIClient()).delete(pod_id)
    log(f"deleted pod {pod_id}")
    out = ROOT / "out" / "e05b" / "pipeline.log"
    subprocess.run([VRAMORAMA, "run", "--vram", "64M", "--owner", "e05b-driver", "--detach", "--out", str(out), "--",
                    PD_PY, "-u", str(ROOT / "experiments" / "e05_pipeline.py"), "--seeds", *seeds, "--batch", "32"],
                   capture_output=True, text=True)
    log(f"started the local e05b analysis for seeds {seeds} (log {out})")


if __name__ == "__main__":
    main()
