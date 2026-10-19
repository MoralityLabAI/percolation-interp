"""e04, exploratory (not pre-registered): the massive activation in the decomposed slice.

Layer 2's down_proj writes a vector of norm ~7,000 at position 0 (~5 elsewhere): Qwen's massive
activation, the anchor of its attention sink. This finds the components that write it and
measures, on the held-out "causal" rows, KL to the target when only they are kept (mask = CI x
membership, as in e04_causal.py), alone and together with e04's sieved backbone, and whether
the backbone, the top mean-CI set or e04's sets contain them.

Also splits the zero-ablation gap on the target model alone, by editing only what the slice
writes to the residual stream (o_proj and down_proj outputs): position 0 kept as is or at its
mean with the rest zeroed, and the rest at its mean with position 0 zeroed. And it measures the
norm of the position-0 down_proj write under each restriction.

  python experiments/e04_massive.py
  (seed 1: E04_SEED=1 E04_OUT=out/e04_seed1)
Writes <E04_OUT>/massive.json.
"""

from __future__ import annotations

import json
from collections import Counter

import torch
import torch.nn.functional as F

from e04_common import OUT, SLICE, ci_lower, find_run, heldout_part, load_component_model, load_stats

DOWN = f"model.layers.{SLICE[0]}.mlp.down_proj"


def main(n_batches: int = 32, batch: int = 4, dev: str = "cuda"):
    from param_decomp.models.components import make_mask_infos

    run_id = find_run()
    cm, run_info = load_component_model(run_id, dev)
    stats = load_stats()
    modules = list(cm.target_module_paths)
    comp = cm.components[DOWN]
    unorm = comp.U.float().norm(dim=1)  # (C,)
    batches = [b for _, b in zip(range(n_batches), heldout_part(batch, "causal"))]
    inputs = {}

    def grab_in(_, args, __):
        inputs["x"] = args[0]

    # Per-component write at position 0: |ci * act| * ||U||, averaged over rows.
    contrib = torch.zeros(comp.U.shape[0], device=dev)
    with torch.no_grad():
        h = cm.target_model.get_submodule(DOWN).register_forward_hook(grab_in)
        for idx in batches:
            idx = idx.to(dev)
            cm(idx)
            ci = ci_lower(cm, run_info, idx)
            acts = comp.get_component_acts(inputs["x"][:, 0].float())  # (B, C)
            contrib += (ci[DOWN][:, 0].float() * acts).abs().mul(unorm).sum(0)
        h.remove()
    contrib /= sum(len(b) for b in batches)
    order = contrib.argsort(descending=True)
    top = [int(i) for i in order[:5]]
    analysis = json.loads((OUT / "analysis.json").read_text())
    bb = {(m, int(i)) for m, i in analysis["within_layer"]["backbone"]}
    ci_mean = stats["ci_sum"][DOWN].numpy() / stats["tokens"]
    n_bb_down = sum(1 for m, _ in bb if m == DOWN)
    top_ci_down = {int(i) for i in ci_mean.argsort()[::-1][:n_bb_down]}

    def mask_of(S, ci):
        out = {m: torch.zeros_like(ci[m]) for m in modules}
        for m, i in S:
            out[m][..., i] = ci[m][..., i]
        return out

    sets = {f"massive_top{k}": {(DOWN, i) for i in top[:k]} for k in (1, 2, 3, 5)}
    sets["backbone"] = bb
    sets["backbone_plus_massive_top1"] = bb | {(DOWN, top[0])}
    writers = [m for m in modules if m.endswith(("o_proj", "down_proj"))]

    # Means of each writer's output at position 0 and elsewhere, on the target model.
    s0 = {m: 0.0 for m in writers}
    sr = {m: 0.0 for m in writers}
    n_rows = n_rest = 0

    def acc(name):
        def hook(_, __, out):
            s0[name] = s0[name] + out[:, 0].float().sum(0)
            sr[name] = sr[name] + out[:, 1:].float().sum((0, 1))
        return hook

    with torch.no_grad():
        for idx in batches:
            hs = [cm.target_model.get_submodule(m).register_forward_hook(acc(m)) for m in writers]
            cm(idx.to(dev))
            for h in hs:
                h.remove()
            n_rows += idx.shape[0]
            n_rest += idx.shape[0] * (idx.shape[1] - 1)
    m0 = {m: s0[m] / n_rows for m in writers}
    mr = {m: sr[m] / n_rest for m in writers}

    def edit(name, pos0, rest):  # pos0/rest: "actual", "mean" or "zero"
        def hook(_, __, out):
            new = out.clone()
            for sl, how, mean in ((slice(0, 1), pos0, m0[name]), (slice(1, None), rest, mr[name])):
                if how == "zero":
                    new[:, sl] = 0
                elif how == "mean":
                    new[:, sl] = mean.to(out.dtype)
            return new
        return hook

    edits = {"writers_pos0_actual_rest_zero": ("actual", "zero"), "writers_pos0_mean_rest_zero": ("mean", "zero"),
             "writers_pos0_zero_rest_mean": ("zero", "mean"), "writers_pos0_mean_rest_mean": ("mean", "mean"),
             "writers_pos0_actual_rest_mean": ("actual", "mean")}
    write0 = Counter()

    def norm0(_, __, out):
        inputs["n0"] = out[:, 0].float().norm(dim=-1).mean().item()

    kl = Counter()
    with torch.no_grad():
        for idx in batches:
            idx = idx.to(dev)
            target = F.log_softmax(cm(idx).float(), -1)
            ci = ci_lower(cm, run_info, idx)
            runs = {"all": ci, "none": {m: torch.zeros_like(ci[m]) for m in modules}}
            runs.update({k: mask_of(S, ci) for k, S in sets.items()})
            for k, mk in runs.items():
                h = cm.target_model.get_submodule(DOWN).register_forward_hook(norm0)
                logp = F.log_softmax(cm(idx, mask_infos=make_mask_infos(mk)).float(), -1)
                h.remove()
                write0[k] += inputs["n0"] / len(batches)
                kl[k] += (target.exp() * (target - logp)).sum(-1).mean().item() / len(batches)
            for k, (a, b) in edits.items():
                hs = [cm.target_model.get_submodule(m).register_forward_hook(edit(m, a, b)) for m in writers]
                logp = F.log_softmax(cm(idx).float(), -1)
                for h in hs:
                    h.remove()
                kl[k] += (target.exp() * (target - logp)).sum(-1).mean().item() / len(batches)
            h = cm.target_model.get_submodule(DOWN).register_forward_hook(norm0)
            cm(idx)
            h.remove()
            write0["target"] += inputs["n0"] / len(batches)
    gap = kl["none"] - kl["all"]
    res = {"run_id": run_id, "module": DOWN,
           "top_writers": [{"c": i, "write_norm_pos0": contrib[i].item(), "ci_mean": float(ci_mean[i]),
                            "in_backbone": (DOWN, i) in bb, "in_top_ci": i in top_ci_down} for i in top],
           "total_write_norm_pos0": contrib.sum().item(),
           "down_write_norm_pos0": dict(write0), "kl": dict(kl), "share_of_gap": {k: (kl["none"] - v) / gap for k, v in kl.items()}}
    (OUT / "massive.json").write_text(json.dumps(res, indent=1))
    for w in res["top_writers"]:
        print(w)
    for k, v in kl.items():
        w = f"  pos-0 down write {write0[k]:8.1f}" if k in write0 else ""
        print(f"{k:32s} KL {v:.4f}  share of gap {res['share_of_gap'][k]:7.2%}{w}")
    print(f"target pos-0 down write {write0['target']:.1f}")


if __name__ == "__main__":
    main()
