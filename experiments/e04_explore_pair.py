"""e04, exploratory (not pre-registered): how much of the backbone's effect is one attention pair?

The seven matched random sets that close ~4% of the KL gap all contain both L2 v[187] and
L2 o[335] (a value and an output direction of the same attention layer); sets with only one
stay near 0%. This measures the same restriction as e04_causal.py, on the same held-out rows,
for the pair, each half, the backbone without the pair, and the top-CI set without the pair.

  python experiments/e04_explore_pair.py
Writes out/e04/explore_pair.json.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from e04_common import OUT, ci_lower, find_run, heldout_part, load_component_model, load_stats

V, O = ("model.layers.2.self_attn.v_proj", 187), ("model.layers.2.self_attn.o_proj", 335)


def main(n_batches: int = 32, batch: int = 4, dev: str = "cuda"):
    from param_decomp.models.components import make_mask_infos

    run_id = find_run()
    cm, run_info = load_component_model(run_id, dev)
    stats = load_stats()
    modules = list(cm.target_module_paths)
    analysis = json.loads((OUT / "analysis.json").read_text())
    bb = {tuple(x) for x in analysis["within_layer"]["backbone"]}
    ci_mean = {m: stats["ci_sum"][m].numpy() / stats["tokens"] for m in modules}
    size = {m: sum(1 for mm, _ in bb if mm == m) for m in modules}
    top = {(m, int(i)) for m in modules for i in ci_mean[m].argsort()[::-1][:size[m]]}
    sets = {"pair": {V, O}, "v_only": {V}, "o_only": {O}, "backbone": bb, "backbone_minus_pair": bb - {V, O},
            "top_ci": top, "top_ci_minus_pair": top - {V, O}, "none": set()}

    def mask(S):
        out = {m: torch.zeros(cm.module_to_c[m], device=dev) for m in modules}
        for m, i in S:
            out[m][i] = 1.0
        return out

    masks = {k: mask(S) for k, S in sets.items()}
    kl = {k: 0.0 for k in sets}
    kl["all"] = 0.0
    with torch.no_grad():
        for b, idx in enumerate(heldout_part(batch, "causal")):
            if b >= n_batches:
                break
            idx = idx.to(dev)
            target = F.log_softmax(cm(idx).float(), -1)
            ci = ci_lower(cm, run_info, idx)
            for k in kl:
                mi = make_mask_infos(ci if k == "all" else {m: ci[m] * masks[k][m] for m in modules})
                logp = F.log_softmax(cm(idx, mask_infos=mi).float(), -1)
                kl[k] += (target.exp() * (target - logp)).sum(-1).mean().item() / n_batches
    gap = kl["none"] - kl["all"]
    res = {"run_id": run_id, "kl": kl, "share_of_gap": {k: (kl["none"] - v) / gap for k, v in kl.items()},
           "sizes": {k: len(S) for k, S in sets.items()}, "tokens": n_batches * batch * 128}
    (OUT / "explore_pair.json").write_text(json.dumps(res, indent=1))
    for k, v in res["share_of_gap"].items():
        print(f"{k:22s} KL {kl[k]:.4f}  share of gap {v:.3%}")


if __name__ == "__main__":
    main()
