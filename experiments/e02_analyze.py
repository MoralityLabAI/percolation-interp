"""Step 2b: percolation measurements across a training run's checkpoints.

For a fixed sample of test inputs, at every checkpoint:
  - EAP-IG edge scores (mean-ablation edges; graph: 2 token nodes -> D x N -> sink)
  - entry keys three ways: ranked per layer ("rank"), the within-layer shuffle
    null ("shuf"), and absolute thresholds against the final checkpoint's
    pooled scores ("abs")
  - all in degree units: c = mean kept out-degree per live source node
  - spanning point c*, backbone node fraction over a c grid and at c_ref = 2,
    backbone edge fraction (rank)
At step 0 and the final checkpoint, also subgraph fidelity over a c grid, with
all kept edges and with only the backbone's kept edges (H1a).

Run under a vramorama lease, e.g.
  vramorama run --vram 3G --owner percolation-interp -- \
      .venv/Scripts/python experiments/e02_analyze.py out/e02/runs/P113_D4_N256_s0.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from percinterp import dense
from percinterp.modadd import (
    all_pairs,
    edge_scores,
    fidelity,
    load_checkpoints,
    masked_logits,
    mean_activations,
    metric_direction,
    one_hot_pairs,
    split,
)

OUT = Path(__file__).resolve().parents[1] / "out" / "e02" / "analysis"
MODES = ("rank", "shuf", "abs")


def batches(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(n, i + bs))


@torch.no_grad()
def reference_pools(model, x, c, means, bs, ig_steps, max_per_layer=2_000_000, seed=0):
    """Pooled nonzero scores per layer, and each layer's mean nonzero edges per live source."""
    pools, per_src = None, []
    for sl in batches(len(x), bs):
        S = edge_scores(model, x[sl], c[sl], means, steps=ig_steps)
        nz = [s[s > 0] for s in S]
        pools = nz if pools is None else [torch.cat([p, n]) for p, n in zip(pools, nz)]
        per_src.append(torch.stack([
            (s > 0).flatten(1).sum(1).float() / (s > 0).any(dim=1).sum(1).clamp(min=1) for s in S
        ], dim=1))
    g = torch.Generator(device=x.device).manual_seed(seed)
    pools = [p[torch.randperm(len(p), generator=g, device=x.device)[:max_per_layer]] for p in pools]
    return pools, torch.cat(per_src).mean(0).tolist()


@torch.no_grad()
def checkpoint_stats(model, x, c, cfg, bs, ig_steps, ps, p_ref, ref, ref_scale, gen, ps_fid=None):
    means = mean_activations(model, cfg.P, x.device)
    acc = defaultdict(list)
    p_ref_t = torch.tensor([p_ref], device=x.device)
    for sl in batches(len(x), bs):
        xb, cb = x[sl], c[sl]
        S = edge_scores(model, xb, cb, means, steps=ig_steps)
        acc["n_edges"].append(sum((s > 0).flatten(1).sum(1) for s in S).float())
        for mode in MODES:
            if mode == "rank":
                E = dense.rank_entry(S, unit="degree")
            elif mode == "shuf":
                E = dense.shuffled_entry(S, gen, unit="degree")
            else:
                E = dense.reference_entry(S, ref, ref_scale)
            prof = dense.DenseProfile(E)
            acc[f"{mode}_pspan"].append(prof.p_span)
            acc[f"{mode}_bb_node"].append(prof.backbone_node_fraction(ps))
            acc[f"{mode}_bb_node_ref"].append(prof.backbone_node_fraction(p_ref_t)[:, 0])
            if mode == "rank":
                acc["rank_bb_edge"].append(prof.backbone_edge_fraction(ps))
                if ps_fid is not None:
                    fid_batch(model, xb, cb, means, prof, ps_fid, acc)
            del prof, E
        del S
    out = {}
    for k, v in acc.items():
        if k.startswith("fid_"):
            continue
        t = torch.cat(v)
        out[k] = t.median(0).values.tolist() if t.dim() == 1 else t.mean(0).tolist()
        if t.dim() == 1:
            out[k + "_q25"] = t.quantile(0.25).item()
            out[k + "_q75"] = t.quantile(0.75).item()
    if ps_fid is not None:
        out["fidelity"] = fid_summary(acc)
    return out


@torch.no_grad()
def fid_batch(model, xb, cb, means, prof, ps_fid, acc):
    mdir = metric_direction(model, cb)
    full = (model(xb) * mdir).sum(1)
    shapes = [E.shape for E in prof.E]
    zero = [torch.zeros(s, device=xb.device) for s in shapes]
    abl = (masked_logits(model, xb, zero, means) * mdir).sum(1)
    acc["fid_full"].append(full)
    acc["fid_abl"].append(abl)
    for k, p in enumerate(ps_fid.tolist()):
        for kind, bb in (("all", False), ("backbone", True)):
            logits = masked_logits(model, xb, prof.masks(p, backbone_only=bb), means)
            f = fidelity(logits, cb, full, abl)
            acc[f"fid_{kind}_metric_{k}"].append((logits * mdir).sum(1))
            acc[f"fid_{kind}_acc_{k}"].append(f["acc"])
            acc[f"fid_{kind}_loss_{k}"].append(f["loss"])
        acc[f"fid_bb_edge_{k}"].append(prof.backbone_edge_fraction(torch.tensor([p], device=xb.device))[:, 0])


def fid_summary(acc):
    full = torch.cat(acc["fid_full"]).mean()
    abl = torch.cat(acc["fid_abl"]).mean()
    ks = sorted({int(k.rsplit("_", 1)[1]) for k in acc if k.startswith("fid_bb_edge_")})
    out = {"full_metric": full.item(), "ablated_metric": abl.item()}
    for kind in ("all", "backbone"):
        out[kind] = {
            "recovered": [((torch.cat(acc[f"fid_{kind}_metric_{k}"]).mean() - abl) / (full - abl)).item() for k in ks],
            "acc": [torch.cat(acc[f"fid_{kind}_acc_{k}"]).mean().item() for k in ks],
            "loss": [torch.cat(acc[f"fid_{kind}_loss_{k}"]).mean().item() for k in ks],
        }
    out["bb_edge"] = [torch.cat(acc[f"fid_bb_edge_{k}"]).mean().item() for k in ks]
    return out


def analyze(run_path: Path, device: str, n_inputs: int, ig_steps: int, budget: float, max_ckpts: int | None):
    t0 = time.time()
    cfg, models = load_checkpoints(run_path, device)
    steps = sorted(models)
    if max_ckpts and len(steps) > max_ckpts:
        idx = torch.linspace(0, len(steps) - 1, max_ckpts).round().long().tolist()
        steps = sorted({steps[i] for i in idx})
    a, b, c = all_pairs(cfg.P, device)
    _, te = split(cfg.P, cfg.frac_train, cfg.seed)
    pick = te[torch.randperm(len(te), generator=torch.Generator().manual_seed(1234))[:n_inputs]].to(device)
    x, cc = one_hot_pairs(a[pick], b[pick], cfg.P), c[pick]
    edges_per_input = 2 * cfg.P * cfg.width + (cfg.depth - 1) * cfg.width**2 + cfg.width
    bs = int(max(2, min(n_inputs, budget // edges_per_input)))
    top = math.log10(cfg.width)
    ps = torch.logspace(-2, top, 33, device=device)  # degree units c
    ps_fid = torch.logspace(-1.5, top, 15, device=device)
    p_ref = 2.0
    final = steps[-1]
    means_final = mean_activations(models[final], cfg.P, device)
    ref, ref_scale = reference_pools(models[final], x, cc, means_final, bs, ig_steps)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    history = {h["step"]: h for h in json.loads(run_path.with_suffix(".json").read_text())["history"]}
    records = []
    for step in steps:
        rec = {"step": step, **{k: v for k, v in history.get(step, {}).items() if k != "step"}}
        do_fid = step in (steps[0], final)
        rec.update(checkpoint_stats(models[step], x, cc, cfg, bs, ig_steps, ps, p_ref, ref, ref_scale,
                                    gen, ps_fid if do_fid else None))
        records.append(rec)
        print(f"  step {step:>6}: test acc {rec.get('test_acc', float('nan')):.3f}  "
              f"c* rank {rec['rank_pspan']:.3f} shuf {rec['shuf_pspan']:.3f}  "
              f"backbone@c=2 rank {rec['rank_bb_node_ref']:.3f} shuf {rec['shuf_bb_node_ref']:.3f} "
              f"abs {rec['abs_bb_node_ref']:.3f}", flush=True)
    result = {
        "config": asdict(cfg),
        "name": cfg.name,
        "n_inputs": n_inputs,
        "ig_steps": ig_steps,
        "batch": bs,
        "units": "degree (c = mean kept out-degree per live source); keys named p* / ps hold c",
        "ps": ps.tolist(),
        "ps_fid": ps_fid.tolist(),
        "p_ref": p_ref,
        "ref_scale": ref_scale,
        "full_history": list(history.values()),
        "checkpoints": records,
        "seconds": time.time() - t0,
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else 0,
    }
    out_dir = OUT / run_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg.name}.json").write_text(json.dumps(result))
    return result


def finished_runs(path: Path) -> list[Path]:
    """Checkpoint files whose training finished (the .json is written after the .pt)."""
    if path.is_dir():
        return sorted(p for p in path.glob("*.pt") if p.with_suffix(".json").exists())
    return [path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+", help="checkpoint files or run directories")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-inputs", type=int, default=256)
    ap.add_argument("--ig-steps", type=int, default=8)
    ap.add_argument("--budget", type=float, default=2e7, help="edge elements per batch tensor")
    ap.add_argument("--max-ckpts", type=int, default=None)
    ap.add_argument("--watch", type=int, default=0,
                    help="keep polling the given directories until this many runs are analysed")
    ap.add_argument("--timeout-hours", type=float, default=6.0)
    args = ap.parse_args()
    deadline = time.time() + 3600 * args.timeout_hours
    while True:
        done = 0
        for run in [r for p in args.runs for r in finished_runs(p)]:
            if (OUT / run.parent.name / f"{run.stem}.json").exists():
                done += 1
                continue
            print(f"{run.stem}", flush=True)
            r = analyze(run, args.device, args.n_inputs, args.ig_steps, args.budget, args.max_ckpts)
            print(f"  done in {r['seconds']:.0f}s, peak cuda {r['peak_cuda_mib']:.0f} MiB", flush=True)
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            done += 1
        if done >= args.watch or time.time() > deadline:
            print(f"{done} runs analysed", flush=True)
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
