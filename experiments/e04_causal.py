"""e04 causal test (Q2, Q3): keep only the sieved components of the Qwen slice.

Restriction: mask = CI * 1[c in S] for every decomposed module (delta term excluded, as in
param-decomp's CI-masked metric); the rest of Qwen3-0.6B runs unchanged. Measure: mean over
held-out tokens of KL(target || restricted), full vocabulary.
Sets: the within-layer sieve's backbone; 100 random sets matched per module in size and
firing-rate decile; the same-size top-mean-CI set; all components; none.

  python experiments/e04_causal.py [--step N]
Writes out/e04/causal.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from e04_common import MIN_RATE, OUT, ci_lower, find_run, heldout_part, load_component_model, load_stats


def decile_sampler(rate, pool, rng):
    pool = np.asarray(pool)
    edges = np.quantile(rate[pool], np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, rate[pool], side="right") - 1, 0, 9)

    def draw(chosen):
        want = np.clip(np.searchsorted(edges, rate[np.asarray(chosen)], side="right") - 1, 0, 9)
        out, used = [], set()
        for b in want:
            cand = [c for c in pool[bins == b] if c not in used] or list(pool[bins == b])
            pick = int(rng.choice(cand))
            used.add(pick)
            out.append(pick)
        return out
    return draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-batches", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--step", type=int, default=None)
    args = ap.parse_args()
    dev, rng = args.device, np.random.default_rng(12)
    from param_decomp.models.components import make_mask_infos

    run_id = find_run(args.step)
    cm, run_info = load_component_model(run_id, dev)
    stats = load_stats()
    analysis = json.loads((OUT / "analysis.json").read_text())
    modules = list(cm.target_module_paths)
    rate = {m: stats["n_fire"][m].numpy() / stats["tokens"] for m in modules}
    ci_mean = {m: stats["ci_sum"][m].numpy() / stats["tokens"] for m in modules}
    bb = {m: [] for m in modules}
    for m, c in analysis["within_layer"]["backbone"]:
        bb[m].append(int(c))
    alive = {m: np.flatnonzero(rate[m] >= MIN_RATE) for m in modules}
    samplers = {m: decile_sampler(rate[m], alive[m], rng) for m in modules}
    sets = {"backbone": bb, "all": None, "none": {m: [] for m in modules},
            "top_ci": {m: list(np.argsort(-ci_mean[m])[:len(bb[m])]) for m in modules}}
    for k in range(args.n_random):
        sets[f"random_{k}"] = {m: samplers[m](bb[m]) if bb[m] else [] for m in modules}

    def mask_of(S):
        if S is None:
            return None
        out = {}
        for m in modules:
            v = torch.zeros(cm.module_to_c[m], device=dev)
            if S[m]:
                v[torch.tensor(S[m], device=dev)] = 1.0
            out[m] = v
        return out

    masks = {name: mask_of(S) for name, S in sets.items()}
    kl = {name: 0.0 for name in sets}
    t0 = time.time()
    with torch.no_grad():
        for b, idx in enumerate(heldout_part(args.batch, "causal")):
            if b >= args.n_batches:
                break
            idx = idx.to(dev)
            target_logp = F.log_softmax(cm(idx).float(), -1)
            ci = ci_lower(cm, run_info, idx)
            for name, mk in masks.items():
                mi = make_mask_infos(ci if mk is None else {m: ci[m] * mk[m] for m in modules})
                logp = F.log_softmax(cm(idx, mask_infos=mi).float(), -1)
                kl[name] += (target_logp.exp() * (target_logp - logp)).sum(-1).mean().item() / args.n_batches
            if (b + 1) % 8 == 0:
                print(f"batch {b + 1}/{args.n_batches}: {time.time() - t0:.0f}s", flush=True)
    rand = np.array([v for k, v in kl.items() if k.startswith("random_")])
    res = {"run_id": run_id, "backbone": kl["backbone"], "top_ci": kl["top_ci"], "all": kl["all"], "none": kl["none"],
           "random": rand.tolist(), "random_median": float(np.median(rand)), "random_min": float(rand.min()),
           "p_value": float((1 + (rand <= kl["backbone"]).sum()) / (1 + len(rand))),
           "Q2_pass": bool(kl["backbone"] < rand.min()), "Q3_pass": bool(kl["backbone"] < kl["top_ci"]),
           "backbone_size": {m: len(v) for m, v in bb.items()}, "tokens": args.n_batches * args.batch * 128}
    (OUT / "causal.json").write_text(json.dumps(res, indent=1))
    print(f"backbone KL {kl['backbone']:.4f} vs random median {res['random_median']:.4f} (min {res['random_min']:.4f}), "
          f"p {res['p_value']:.3f}; top-CI {kl['top_ci']:.4f}; all {kl['all']:.4f}; none {kl['none']:.4f}")


if __name__ == "__main__":
    main()
