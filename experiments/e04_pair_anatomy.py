"""e04, exploratory (not pre-registered): what is the always-on layer-2 attention pair?

In both seeds one value direction and one output direction of layer 2's attention close ~4%
of the KL gap together and nothing apart (docs/e04_results.md). Suspicion: the pair writes a
nearly constant vector, i.e. it acts as a bias that later layers depend on. Checks, on the
held-out "causal" rows:

1. Where the pair fires: CI > 0.1 by position (0 vs the rest) and the tokens on which the
   value direction fires most.
2. How constant its output is. With only the pair on, o_proj writes s_t * U[o] (rank one), so
   constancy is about the scalar s_t: mean(s)^2 / mean(s^2) is the share of its second moment
   carried by a constant.
3. Causal: the slice with every component off, plus the pair's mean output added back as a
   constant, against the pair itself (KL to the target, share of the none->all gap).
4. The whole slice zero-ablated (as e04's "none" and the slice-selection rule) against
   mean-ablated, two ways: only the residual writers (o_proj, down_proj) replaced by their mean
   outputs, which is the slice's contribution to the residual stream at its mean; and all 14
   projections (q and k at their means also flatten every attention pattern).

  python experiments/e04_pair_anatomy.py --v 187 --o 335
  (seed 1: E04_SEED=1 E04_OUT=out/e04_seed1, --v 182 --o 59)
Writes <E04_OUT>/pair_anatomy.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import torch
import torch.nn.functional as F

from e04_common import MODEL, OUT, SLICE, ci_lower, find_run, heldout_part, load_component_model

PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def kl_to(target_logp, logits):
    logp = F.log_softmax(logits.float(), -1)
    return (target_logp.exp() * (target_logp - logp)).sum(-1).mean().item()


def main(v_idx: int, o_idx: int, n_batches: int = 32, batch: int = 4, dev: str = "cuda"):
    from param_decomp.models.components import make_mask_infos
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    V = (f"model.layers.{SLICE[0]}.self_attn.v_proj", v_idx)
    O = (f"model.layers.{SLICE[0]}.self_attn.o_proj", o_idx)
    run_id = find_run()
    cm, run_info = load_component_model(run_id, dev)
    modules = list(cm.target_module_paths)
    attn = cm.target_model.get_submodule(f"model.layers.{SLICE[0]}.self_attn")
    o_mod = cm.target_model.get_submodule(O[0])
    batches = [b for _, b in zip(range(n_batches), heldout_part(batch, "causal"))]

    def pair_masks(ci):
        out = {m: torch.zeros_like(ci[m]) for m in modules}
        for m, i in (V, O):
            out[m][..., i] = ci[m][..., i]
        return out

    # Pass 1: firing, the pair's output scalar, and per-projection mean outputs of the target.
    n_tok = 0
    fire_v_pos0 = fire_v_rest = fire_o_pos0 = fire_o_rest = 0.0
    v_tokens, s_all = Counter(), []
    proj_sum = {m: None for m in modules}
    pos0_sum = {m: None for m in modules}  # position 0 kept apart: early layers write massive activations there
    norm_pos0 = {m: 0.0 for m in modules}
    norm_rest = {m: 0.0 for m in modules}
    captured = {}

    def grab_o(_, __, out):
        captured["o"] = out

    def sum_out(name):
        def hook(_, __, out):
            o = out.detach().float()
            s, s0 = o[:, 1:].sum((0, 1)), o[:, 0].sum(0)
            proj_sum[name] = s if proj_sum[name] is None else proj_sum[name] + s
            pos0_sum[name] = s0 if pos0_sum[name] is None else pos0_sum[name] + s0
            norm_pos0[name] += o[:, 0].norm(dim=-1).sum().item()
            norm_rest[name] += o[:, 1:].norm(dim=-1).sum().item()
        return hook

    with torch.no_grad():
        for idx in batches:
            idx = idx.to(dev)
            hs = [cm.target_model.get_submodule(m).register_forward_hook(sum_out(m)) for m in modules]
            cm(idx)  # target model, unmodified
            for h in hs:
                h.remove()
            ci = ci_lower(cm, run_info, idx)
            cv, co = ci[V[0]][..., V[1]], ci[O[0]][..., O[1]]
            fire_v_pos0 += (cv[:, 0] > 0.1).float().sum().item()
            fire_v_rest += (cv[:, 1:] > 0.1).float().sum().item()
            fire_o_pos0 += (co[:, 0] > 0.1).float().sum().item()
            fire_o_rest += (co[:, 1:] > 0.1).float().sum().item()
            v_tokens.update(idx[cv > 0.1].tolist())
            h = o_mod.register_forward_hook(grab_o)
            cm(idx, mask_infos=make_mask_infos(pair_masks(ci)))
            h.remove()
            u = cm.components[O[0]].U[O[1]].float()
            s_all.append((captured["o"].float() @ u / u.dot(u)).flatten())  # o_proj output = s * U[o]
            n_tok += idx.numel()
    rows = n_tok // 128
    s = torch.cat(s_all)
    u = cm.components[O[0]].U[O[1]].float()
    mean_pair_vec = s.mean() * u
    proj_mean = {m: (proj_sum[m] + pos0_sum[m]) / n_tok for m in modules}  # over all positions
    rest_mean = {m: proj_sum[m] / (n_tok - rows) for m in modules}
    pos0_mean = {m: pos0_sum[m] / rows for m in modules}

    # Pass 2: KL to the target under each intervention.
    add = {}

    def add_hook(_, __, out):
        if "vec" not in add:
            return out
        if isinstance(out, tuple):
            return (out[0] + add["vec"].to(out[0].dtype),) + tuple(out[1:])
        return out + add["vec"].to(out.dtype)

    def mean_hook(name):
        def hook(_, __, out):
            return proj_mean[name].to(out.dtype).expand_as(out).clone()
        return hook

    def pos_mean_hook(name):
        def hook(_, __, out):
            new = rest_mean[name].to(out.dtype).expand_as(out).clone()
            new[:, 0] = pos0_mean[name].to(out.dtype)
            return new
        return hook

    kl = Counter()
    with torch.no_grad():
        h_add = attn.register_forward_hook(add_hook)
        for idx in batches:
            idx = idx.to(dev)
            target = F.log_softmax(cm(idx).float(), -1)
            ci = ci_lower(cm, run_info, idx)
            zero = {m: torch.zeros_like(ci[m]) for m in modules}
            kl["all"] += kl_to(target, cm(idx, mask_infos=make_mask_infos(ci)))
            kl["none"] += kl_to(target, cm(idx, mask_infos=make_mask_infos(zero)))
            kl["pair"] += kl_to(target, cm(idx, mask_infos=make_mask_infos(pair_masks(ci))))
            add["vec"] = mean_pair_vec
            kl["none_plus_pair_mean"] += kl_to(target, cm(idx, mask_infos=make_mask_infos(zero)))
            del add["vec"]
            for name, mods in [("slice_writers_mean_ablated", [m for m in modules if m.endswith(("o_proj", "down_proj"))]),
                               ("slice_all_proj_mean_ablated", modules)]:
                hs = [cm.target_model.get_submodule(m).register_forward_hook(mean_hook(m)) for m in mods]
                kl[name] += kl_to(target, cm(idx))
                for hh in hs:
                    hh.remove()
            writers = [m for m in modules if m.endswith(("o_proj", "down_proj"))]
            hs = [cm.target_model.get_submodule(m).register_forward_hook(pos_mean_hook(m)) for m in writers]
            kl["slice_writers_mean_ablated_pos0_apart"] += kl_to(target, cm(idx))
            for hh in hs:
                hh.remove()
        h_add.remove()
    kl = {k: v / len(batches) for k, v in kl.items()}
    gap = kl["none"] - kl["all"]
    share = {k: (kl["none"] - v) / gap for k, v in kl.items()}
    res = {
        "run_id": run_id, "pair": [list(V), list(O)], "tokens": n_tok,
        "fire_rate": {"v_pos0": fire_v_pos0 / rows, "v_rest": fire_v_rest / (n_tok - rows),
                      "o_pos0": fire_o_pos0 / rows, "o_rest": fire_o_rest / (n_tok - rows)},
        "v_top_tokens": [(tok.decode([t]), n) for t, n in v_tokens.most_common(15)],
        "v_distinct_tokens": len(v_tokens),
        "pair_scalar": {"mean": s.mean().item(), "sd": s.std().item(),
                        "share_of_second_moment_in_mean": (s.mean() ** 2 / (s ** 2).mean()).item(),
                        "frac_positive": (s > 0).float().mean().item()},
        "pair_mean_vec_norm": mean_pair_vec.norm().item(),
        "writer_output_norm": {m: {"pos0": norm_pos0[m] / rows, "rest": norm_rest[m] / (n_tok - rows)}
                               for m in modules if m.endswith(("o_proj", "down_proj"))},
        "kl": kl, "share_of_gap": share,
    }
    (OUT / "pair_anatomy.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("fire_rate", "pair_scalar", "v_distinct_tokens")}, indent=1))
    print("v top tokens:", res["v_top_tokens"])
    print("writer output norms (pos 0 / rest):", {m.split("layers.")[1]: f"{v['pos0']:.1f} / {v['rest']:.1f}"
                                                  for m, v in res["writer_output_norm"].items()})
    for k in kl:
        print(f"{k:22s} KL {kl[k]:.4f}  share of gap {share[k]:.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=int, required=True)
    ap.add_argument("--o", type=int, required=True)
    args = ap.parse_args()
    main(args.v, args.o)
