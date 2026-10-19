"""e05 causal test (Q2-Q4, docs/e05_design.md): keep a set S, mean-ablate every other component.

In each decomposed module the output is sum_c ci_c a_c U_c. Components in S run as in e04
(CI x activation). Every other component contributes its position-class mean of ci_c a_c
(position 0; positions 1-127) times U_c. The means are measured under the full CI-masked
decomposition on the held-out "stats" rows. S empty is "none", S = all is the CI-masked model.
KL(target || restricted) on the "causal" rows, full vocabulary, all positions.

Sets: the within-layer backbone; 100 random sets matched per module in size and firing-rate
decile (as e04); the same-size top-mean-CI set; Q4's 100 shuffled-graph backbones
(e05_analyze.py); all; none.

  python experiments/e05_causal.py      (param-decomp venv; E04_SEED / E04_SLICE / E04_OUT set)
Writes <E04_OUT>/causal.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from e04_causal import decile_sampler
from e04_common import MIN_RATE, OUT, ci_lower, find_run, heldout_part, load_component_model, load_stats


def install_bias(comp):
    """Let a component module add a per-position-class bias (2, d_out) to its output."""
    original = comp.forward

    def forward(x, *args, **kwargs):
        out = original(x, *args, **kwargs)
        b = comp.e05_bias
        if b is None:
            return out
        bias = b[1].to(out.dtype).expand_as(out).clone()
        bias[:, 0] = b[0].to(out.dtype)
        return out + bias

    comp.e05_bias = None
    comp.forward = forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-batches", type=int, default=32)
    ap.add_argument("--n-mean-batches", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, rng = args.device, np.random.default_rng(12)
    from param_decomp.models.components import make_mask_infos

    run_id = find_run()
    cm, run_info = load_component_model(run_id, dev)
    stats = load_stats()
    analysis = json.loads((OUT / "analysis.json").read_text())
    modules = list(cm.target_module_paths)
    comps = {m: cm.components[m] for m in modules}
    for c in comps.values():
        install_bias(c)

    # Position-class means of ci * act under the full CI-masked decomposition, and CI mass at position 0.
    mu0 = {m: torch.zeros(cm.module_to_c[m], device=dev, dtype=torch.float64) for m in modules}
    mur = {m: torch.zeros_like(mu0[m]) for m in modules}
    ci0 = {m: torch.zeros_like(mu0[m]) for m in modules}
    ci_all = {m: torch.zeros_like(mu0[m]) for m in modules}
    rows = rest = 0
    with torch.no_grad():
        for _, idx in zip(range(args.n_mean_batches), heldout_part(args.batch, "stats")):
            idx = idx.to(dev)
            ci = ci_lower(cm, run_info, idx)
            cache = cm(idx, mask_infos=make_mask_infos(ci), cache_type="input").cache
            for m in modules:
                v = (ci[m] * comps[m].get_component_acts(cache[m])).double()
                mu0[m] += v[:, 0].sum(0)
                mur[m] += v[:, 1:].sum((0, 1))
                ci0[m] += ci[m][:, 0].double().sum(0)
                ci_all[m] += ci[m].double().sum((0, 1))
            rows += idx.shape[0]
            rest += idx.shape[0] * (idx.shape[1] - 1)
    mu = {m: torch.stack([mu0[m] / rows, mur[m] / rest]).float() for m in modules}  # (2, C)

    rate = {m: stats["n_fire"][m].numpy() / stats["tokens"] for m in modules}
    ci_mean = {m: stats["ci_sum"][m].numpy() / stats["tokens"] for m in modules}
    bb = {m: [] for m in modules}
    for m, c in analysis["within_layer"]["backbone"]:
        bb[m].append(int(c))
    alive = {m: np.flatnonzero(rate[m] >= MIN_RATE) for m in modules}
    samplers = {m: decile_sampler(rate[m], alive[m], rng) for m in modules}
    sets = {"backbone": bb, "all": None, "none": {m: [] for m in modules},
            "top_ci": {m: [int(i) for i in np.argsort(-ci_mean[m])[:len(bb[m])]] for m in modules}}
    for k in range(args.n_random):
        sets[f"random_{k}"] = {m: samplers[m](bb[m]) if bb[m] else [] for m in modules}
    for k, null in enumerate(analysis["q4_null_backbones"]):
        S = {m: [] for m in modules}
        for m, c in null:
            S[m].append(int(c))
        sets[f"q4null_{k}"] = S

    def member(S):
        out = {}
        for m in modules:
            v = torch.zeros(cm.module_to_c[m], device=dev)
            if S[m]:
                v[torch.tensor(S[m], device=dev)] = 1.0
            out[m] = v
        return out

    members = {name: (None if S is None else member(S)) for name, S in sets.items()}
    biases = {name: (None if mk is None else {m: (mu[m] * (1 - mk[m])) @ comps[m].U.float() for m in modules})
              for name, mk in members.items()}
    kl = {name: 0.0 for name in sets}
    t0 = time.time()
    with torch.no_grad():
        for b, idx in enumerate(heldout_part(args.batch, "causal")):
            if b >= args.n_batches:
                break
            idx = idx.to(dev)
            target = F.log_softmax(cm(idx).float(), -1)
            ci = ci_lower(cm, run_info, idx)
            for name, mk in members.items():
                for m in modules:
                    comps[m].e05_bias = None if mk is None else biases[name][m]
                mi = make_mask_infos(ci if mk is None else {m: ci[m] * mk[m] for m in modules})
                logp = F.log_softmax(cm(idx, mask_infos=mi).float(), -1)
                kl[name] += (target.exp() * (target - logp)).sum(-1).mean().item() / args.n_batches
            if (b + 1) % 4 == 0:
                print(f"batch {b + 1}/{args.n_batches}: {time.time() - t0:.0f}s", flush=True)
    for c in comps.values():
        c.e05_bias = None
    rand = np.array([v for k, v in kl.items() if k.startswith("random_")])
    q4 = np.array([v for k, v in kl.items() if k.startswith("q4null_")])
    gap = kl["none"] - kl["all"]
    share = lambda x: (kl["none"] - x) / gap  # noqa: E731
    pos0_frac = {m: (ci0[m] / ci_all[m].clamp(min=1e-12)).cpu().numpy() for m in modules}
    res = {"run_id": run_id, "backbone": kl["backbone"], "top_ci": kl["top_ci"], "all": kl["all"], "none": kl["none"],
           "random": rand.tolist(), "random_min": float(rand.min()), "random_median": float(np.median(rand)),
           "q4_null": q4.tolist(), "q4_null_min": float(q4.min()), "q4_null_median": float(np.median(q4)),
           "share": {"backbone": share(kl["backbone"]), "top_ci": share(kl["top_ci"]),
                     "random_median": share(float(np.median(rand))), "random_best": share(float(rand.min())),
                     "q4_null_median": share(float(np.median(q4))), "q4_null_best": share(float(q4.min()))},
           "p_q2": float((1 + (rand <= kl["backbone"]).sum()) / (1 + len(rand))),
           "p_q4": float((1 + (q4 <= kl["backbone"]).sum()) / (1 + len(q4))),
           "Q2_pass": bool(kl["backbone"] < rand.min()), "Q3_pass": bool(kl["backbone"] < kl["top_ci"]),
           "Q4_pass": bool(kl["backbone"] < q4.min()),
           "backbone_size": {m: len(v) for m, v in bb.items()},
           "backbone_pos0_concentrated": [[m, c, float(pos0_frac[m][c])] for m, cs in bb.items() for c in cs
                                          if pos0_frac[m][c] > 0.9],
           "tokens": args.n_batches * args.batch * 128}
    (OUT / "causal.json").write_text(json.dumps(res, indent=1))
    print(f"backbone KL {kl['backbone']:.4f} (share {res['share']['backbone']:.2%}); random best {rand.min():.4f}, "
          f"median {np.median(rand):.4f}; top-CI {kl['top_ci']:.4f}; Q4 null best {q4.min():.4f}, median "
          f"{np.median(q4):.4f}; all {kl['all']:.4f}; none {kl['none']:.4f}")
    print(f"Q2 {'pass' if res['Q2_pass'] else 'fail'}, Q3 {'pass' if res['Q3_pass'] else 'fail'}, "
          f"Q4 {'pass' if res['Q4_pass'] else 'fail'}")


if __name__ == "__main__":
    main()
