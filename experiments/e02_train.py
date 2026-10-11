"""Step 2a: train deep ReLU MLPs on modular addition, saving checkpoints.

Trains every (depth, width, seed) combination sequentially in one process, so
one GPU lease covers the whole grid. Skips runs whose checkpoint file exists.

Run under a vramorama lease, e.g.
  vramorama run --vram 1500M --owner percolation-interp -- \
      .venv/Scripts/python experiments/e02_train.py --depths 2 4 --widths 128 256 --device cuda
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import torch

from percinterp.modadd import Config, train

OUT = Path(__file__).resolve().parents[1] / "out" / "e02" / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[2])
    ap.add_argument("--widths", type=int, nargs="+", default=[128])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--P", type=int, default=113)
    ap.add_argument("--frac-train", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--init-scale", type=float, default=1.0)
    ap.add_argument("--init", default="torch", choices=["torch", "kaiming"])
    ap.add_argument("--progress", type=int, default=0, help="print a status line every N steps")
    ap.add_argument("--export", type=Path, default=None,
                    help="also copy each run's history JSON (not checkpoints) here, e.g. a tracked folder")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tag", default="", help="subdirectory for a hyperparameter variant")
    args = ap.parse_args()
    out = args.out / args.tag if args.tag else args.out
    torch.set_num_threads(max(1, torch.get_num_threads()))
    for depth, width, seed in itertools.product(args.depths, args.widths, args.seeds):
        cfg = Config(P=args.P, depth=depth, width=width, frac_train=args.frac_train, lr=args.lr,
                     weight_decay=args.wd, steps=args.steps, seed=seed, init_scale=args.init_scale,
                     init=args.init)
        if (out / f"{cfg.name}.pt").exists():
            print(f"skip {cfg.name}", flush=True)
            continue
        r = train(cfg, out, device=args.device, progress=args.progress)
        h = r["history"]
        grok = next((rec["step"] for rec in h if rec["test_acc"] >= 0.95), None)
        mem = next((rec["step"] for rec in h if rec["train_acc"] >= 0.99), None)
        print(
            f"{cfg.name}: {r['seconds']:.0f}s, train acc 99% at {mem}, test acc 95% at {grok}, "
            f"final test acc {r['final']['test_acc']:.3f}, peak cuda {r['peak_cuda_mib']:.0f} MiB",
            flush=True,
        )
        if args.export:
            args.export.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / f"{cfg.name}.json", args.export / f"{cfg.name}.json")
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
