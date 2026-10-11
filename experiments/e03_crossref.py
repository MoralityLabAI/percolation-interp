"""e03 stage: cross-reference VPD components against SAE-tower features.

Co-firing: on the same tokens, a component fires when its causal importance
(lower_leaky, as in param-decomp's harvest) exceeds --ci-threshold, and a feature
fires when it is in the token's top k. Counts are accumulated for every
(component, feature) pair.

Weight space, in residual coordinates:
  read  modules (q, k, v, MLP up) of layer l:  gamma * V[:, c]   vs level-l SAE encoder columns
  write modules (o, MLP down) of layer l:      U[c, :]           vs level-(l+1) SAE decoder rows
gamma is the gain of the RMSNorm in front of the module (rms_1 for attention, rms_2 for the MLP).

  python experiments/e03_crossref.py --device cuda --n-batches 64
Writes out/e03/crossref.pt.
"""

from __future__ import annotations

import argparse
import time

import torch

from e03_common import OUT, load_target, residuals, token_loader
from percinterp.sae import TopKSAE

READ = {"attn.q_proj": "rms_1", "attn.k_proj": "rms_1", "attn.v_proj": "rms_1", "mlp.c_fc": "rms_2"}
WRITE = {"attn.o_proj", "mlp.down_proj"}


def module_parts(name: str) -> tuple[int, str]:
    """'h.0.attn.q_proj' -> (0, 'attn.q_proj')."""
    parts = name.split(".")
    return int(parts[1]), ".".join(parts[2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-batches", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ci-threshold", type=float, default=0.1)
    args = ap.parse_args()
    dev = args.device

    from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo

    run_id = (OUT / "vpd_run.txt").read_text().strip()
    run_info = ParamDecompRunInfo.from_path(run_id)
    cm = ComponentModel.from_run_info(run_info).to(dev).eval()
    sampling = run_info.config.sampling
    target = load_target(dev)
    blob = torch.load(OUT / "sae_tower.pt", map_location=dev)
    saes = [TopKSAE.from_state(s).to(dev).eval() for s in blob["levels"]]

    modules = list(cm.components.keys())
    sizes = [cm.components[m].V.shape[1] for m in modules]
    offsets = torch.tensor([0] + sizes).cumsum(0).tolist()
    n_comp, n_feat = offsets[-1], sum(s.n for s in saes)
    feat_off = torch.tensor([0] + [s.n for s in saes]).cumsum(0).tolist()

    co = torch.zeros(n_comp, n_feat, device=dev, dtype=torch.float64)
    n_c = torch.zeros(n_comp, device=dev, dtype=torch.float64)
    n_f = torch.zeros(n_feat, device=dev, dtype=torch.float64)
    ci_sum = torch.zeros(n_comp, device=dev, dtype=torch.float64)
    tokens = 0
    t0 = time.time()
    with torch.no_grad():
        for b, idx in enumerate(token_loader("test", args.batch, seed=3)):
            if b >= args.n_batches:
                break
            idx = idx.to(dev)[:, :512]
            out = cm(idx, cache_type="input")
            ci = cm.calc_causal_importances(pre_weight_acts=out.cache, detach_inputs=True,
                                            sampling=sampling).lower_leaky
            Fc = torch.cat([(ci[m] > args.ci_threshold).flatten(0, 1) for m in modules], dim=1).float()
            ci_sum += torch.cat([ci[m].flatten(0, 1) for m in modules], dim=1).sum(0).double()
            Ff = torch.cat([(s.encode(r.float()) > 0).flatten(0, 1)
                            for s, r in zip(saes, residuals(target, idx))], dim=1).float()
            co += (Fc.T @ Ff).double()
            n_c += Fc.sum(0).double()
            n_f += Ff.sum(0).double()
            tokens += Fc.shape[0]
            if (b + 1) % 8 == 0:
                print(f"batch {b + 1}/{args.n_batches}: {tokens} tokens, {time.time() - t0:.0f}s", flush=True)

        # weight-space cosines, in residual coordinates
        cos_read, cos_write = {}, {}
        for m in modules:
            layer, kind = module_parts(m)
            comp = cm.components[m]
            if kind in READ:
                gamma = getattr(target.h[layer], READ[kind]).weight
                r = gamma[:, None] * comp.V  # (d, C)
                enc = saes[layer].W_enc  # (d, n)
                cos_read[m] = (torch.nn.functional.normalize(r, dim=0).T
                               @ torch.nn.functional.normalize(enc, dim=0)).cpu()
            if kind in WRITE:
                w = comp.U  # (C, d)
                dec = saes[layer + 1].W_dec  # (n, d)
                cos_write[m] = (torch.nn.functional.normalize(w, dim=1)
                                @ torch.nn.functional.normalize(dec, dim=1).T).cpu()

    torch.save({
        "modules": modules, "comp_offsets": offsets, "feat_offsets": feat_off,
        "co": co.cpu(), "n_comp": n_c.cpu(), "n_feat": n_f.cpu(), "ci_sum": ci_sum.cpu(), "tokens": tokens,
        "ci_threshold": args.ci_threshold, "cos_read": cos_read, "cos_write": cos_write,
    }, OUT / "crossref.pt")
    alive_c = int((n_c / tokens >= 1e-5).sum())
    alive_f = int((n_f / tokens >= 1e-5).sum())
    print(f"{tokens} tokens; components firing >= 1e-5: {alive_c}/{n_comp}; features: {alive_f}/{n_feat}")


if __name__ == "__main__":
    main()
