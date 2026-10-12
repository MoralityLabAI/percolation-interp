"""e03 causal validation (design addendum P4, P5): does the sieved structure carry the behaviour?

P4: restrict the VPD decomposition to a component set S (mask = CI * 1[c in S]; delta term
    excluded, as in param-decomp's CI-masked metric) and measure KL(target || restricted).
    Compare the backbone with 100 random sets matched per module in size and firing-rate decile,
    plus reference sets: all components (full CI mask), none, and the top-mean-CI set of the
    same size per module.
P5: at each residual level, subtract every non-backbone SAE feature's contribution (keeping the
    backbone features, error term and bias), run the rest of the model, measure KL; compare with
    100 size- and rate-matched random feature sets.

  python experiments/e03_causal.py --device cuda
Writes out/e03/causal.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from e03_common import OUT, ROOT, load_target, logits_from_level, residuals, token_loader
from percinterp.sae import TopKSAE

MIN_RATE = 1e-5


def kl_per_token(target_logp, logits):
    """Mean over tokens of KL(target || model)."""
    logp = F.log_softmax(logits.float(), -1)
    return (target_logp.exp() * (target_logp - logp)).sum(-1).mean().item()


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
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-random", type=int, default=100)
    args = ap.parse_args()
    dev = args.device
    rng = np.random.default_rng(12)

    from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
    from param_decomp.models.components import make_mask_infos

    analysis = json.loads((ROOT / "results" / "e03" / "analysis.json").read_text())
    X = torch.load(OUT / "crossref.pt", weights_only=False)
    modules, off, foff = X["modules"], X["comp_offsets"], X["feat_offsets"]
    crate = X["n_comp"].numpy() / X["tokens"]
    frate = X["n_feat"].numpy() / X["tokens"]
    ci_mean = X["ci_sum"].numpy() / X["tokens"]

    # --- component sets (local indices per module) ---
    bb = {m: [] for m in modules}
    for m, c in analysis["vpd_sieve"]["backbone"]:
        if m in bb:
            bb[m].append(int(c))
    alive = {m: np.flatnonzero(crate[off[i]:off[i + 1]] >= MIN_RATE) for i, m in enumerate(modules)}
    samplers = {m: decile_sampler(crate[off[i]:off[i + 1]], alive[m], rng) for i, m in enumerate(modules)}
    comp_sets = {"backbone": bb, "all": None, "none": {m: [] for m in modules},
                 "top_ci": {m: list(np.argsort(-ci_mean[off[i]:off[i + 1]])[:len(bb[m])])
                            for i, m in enumerate(modules)}}
    for k in range(args.n_random):
        comp_sets[f"random_{k}"] = {m: samplers[m](bb[m]) if bb[m] else [] for m in modules}

    # --- feature sets per level (local indices) ---
    fbb = {l: [] for l in range(3)}
    for l, i in analysis["sae_sieve"]["backbone"]:
        fbb[int(l)].append(int(i))
    falive = {l: np.flatnonzero(frate[foff[l]:foff[l + 1]] >= MIN_RATE) for l in range(3)}
    fsamplers = {l: decile_sampler(frate[foff[l]:foff[l + 1]], falive[l], rng) for l in range(3)}
    feat_sets = {l: {"backbone": fbb[l], "all_ablated": []} for l in range(3)}
    for l in range(3):
        for k in range(args.n_random):
            feat_sets[l][f"random_{k}"] = fsamplers[l](fbb[l]) if fbb[l] else []

    run_id = (OUT / "vpd_run.txt").read_text().strip()
    run_info = ParamDecompRunInfo.from_path(run_id)
    cm = ComponentModel.from_run_info(run_info).to(dev).eval()
    sampling = run_info.config.sampling
    target = load_target(dev)
    saes = [TopKSAE.from_state(s).to(dev).eval() for s in torch.load(OUT / "sae_tower.pt", map_location=dev)["levels"]]
    C = {m: cm.components[m].V.shape[1] for m in modules}
    set_masks = {name: (None if S is None else
                        {m: torch.zeros(C[m], device=dev).index_fill_(0, torch.tensor(S[m], dtype=torch.long, device=dev), 1.0)
                         if len(S[m]) else torch.zeros(C[m], device=dev) for m in modules})
                 for name, S in comp_sets.items()}

    comp_kl = {name: 0.0 for name in comp_sets}
    feat_kl = {l: {name: 0.0 for name in feat_sets[l]} for l in range(3)}
    t0 = time.time()
    with torch.no_grad():
        for b, idx in enumerate(token_loader("test", args.batch, seed=4)):
            if b >= args.n_batches:
                break
            idx = idx.to(dev)[:, :512]
            out = cm(idx, cache_type="input")
            target_logp = F.log_softmax(target(idx)[0].float(), -1)
            ci = cm.calc_causal_importances(pre_weight_acts=out.cache, detach_inputs=True, sampling=sampling).lower_leaky
            for name, masks in set_masks.items():
                mi = make_mask_infos(ci if masks is None else {m: ci[m] * masks[m] for m in modules})
                comp_kl[name] += kl_per_token(target_logp, cm(idx, mask_infos=mi)) / args.n_batches
            levels = residuals(target, idx)
            for l, sae in enumerate(saes):
                x = levels[l].float()
                a = sae.encode(x)
                for name, keep in feat_sets[l].items():
                    keep_mask = torch.zeros(sae.n, device=dev)
                    if keep:
                        keep_mask[torch.tensor(keep, device=dev)] = 1.0
                    removed = (a * (1 - keep_mask)) @ sae.W_dec / sae.scale
                    logits = logits_from_level(target, (x - removed).to(levels[l].dtype), l)
                    feat_kl[l][name] += kl_per_token(target_logp, logits) / args.n_batches
            print(f"batch {b + 1}/{args.n_batches} {time.time() - t0:.0f}s", flush=True)

    def summarise(kls):
        rand = np.array([v for k, v in kls.items() if k.startswith("random_")])
        return {"backbone": kls["backbone"], "random_median": float(np.median(rand)),
                "random_min": float(rand.min()), "p_value": float((1 + (rand <= kls["backbone"]).sum()) / (1 + len(rand))),
                "random": rand.tolist(), **{k: v for k, v in kls.items() if not k.startswith("random_") and k != "backbone"}}

    res = {"P4": summarise(comp_kl), "P5": {str(l): summarise(feat_kl[l]) for l in range(3)},
           "sizes": {"components": {m: len(v) for m, v in bb.items()}, "features": {l: len(v) for l, v in fbb.items()}},
           "tokens": args.n_batches * args.batch * 512}
    res["P5"]["levels_passing"] = int(sum(res["P5"][str(l)]["p_value"] < 0.01 for l in range(3) if fbb[l]))
    (OUT / "causal.json").write_text(json.dumps(res, indent=1))
    p4 = res["P4"]
    print(f"P4: backbone KL {p4['backbone']:.3f} vs random median {p4['random_median']:.3f} (min {p4['random_min']:.3f}), "
          f"p {p4['p_value']:.3f}; top-CI {p4['top_ci']:.3f}; all {p4['all']:.3f}; none {p4['none']:.3f}")
    for l in range(3):
        r = res["P5"][str(l)]
        print(f"P5 level {l} ({len(fbb[l])} features): backbone KL {r['backbone']:.3f} vs random median "
              f"{r['random_median']:.3f} (min {r['random_min']:.3f}), p {r['p_value']:.3f}; all ablated {r['all_ablated']:.3f}")


if __name__ == "__main__":
    main()
