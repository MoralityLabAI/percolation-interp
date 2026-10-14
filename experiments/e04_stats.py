"""e04: per-component firing rates (CI > 0.1) and mean CI on held-out text.

  python experiments/e04_stats.py [--step N]   (N: test on an intermediate checkpoint)
Writes out/e04/stats.pt.
"""

from __future__ import annotations

import argparse
import time

import torch

from e04_common import CI_FIRE, OUT, ci_lower, find_run, heldout_part, load_component_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-batches", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--step", type=int, default=None)
    args = ap.parse_args()
    run_id = find_run(args.step)
    cm, run_info = load_component_model(run_id, args.device)
    modules = list(cm.target_module_paths)
    n_fire = {m: torch.zeros(cm.module_to_c[m], dtype=torch.float64, device=args.device) for m in modules}
    ci_sum = {m: torch.zeros(cm.module_to_c[m], dtype=torch.float64, device=args.device) for m in modules}
    tokens, t0 = 0, time.time()
    with torch.no_grad():
        for b, idx in enumerate(heldout_part(args.batch, "stats")):
            if b >= args.n_batches:
                break
            ci = ci_lower(cm, run_info, idx.to(args.device))
            for m in modules:
                c = ci[m].flatten(0, 1).double()
                n_fire[m] += (c > CI_FIRE).sum(0)
                ci_sum[m] += c.sum(0)
            tokens += idx.numel()
            if (b + 1) % 50 == 0:
                print(f"batch {b + 1}: {tokens} tokens, {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"run_id": run_id, "modules": modules, "tokens": tokens,
                "n_fire": {m: v.cpu() for m, v in n_fire.items()},
                "ci_sum": {m: v.cpu() for m, v in ci_sum.items()}}, OUT / "stats.pt")
    alive = sum(int((v / tokens >= 1e-5).sum()) for v in n_fire.values())
    total = sum(cm.module_to_c[m] for m in modules)
    print(f"run {run_id}: {tokens} tokens; alive components {alive}/{total}")


if __name__ == "__main__":
    main()
