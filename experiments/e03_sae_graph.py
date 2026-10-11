"""e03 stage: feature-to-feature attribution graph across the SAE tower.

Edge (i at level l -> j at level l+1): for each target feature j, backpropagate
the batch sum of its activation to the level-l residual stream, then read
|a_i(t) * (d_i . grad_j(t))| per token, where a_i is the source feature's
activation and d_i its decoder direction in residual units. Summed over tokens
(and therefore over positions, so attention's cross-position effects count).

Level 1 is the whole residual stream between blocks, a complete cut, so every
level-0 -> level-2 path passes through it: only adjacent edges are needed.

  python experiments/e03_sae_graph.py --device cuda --n-batches 16
Writes out/e03/sae_graph.pt: {"abs": [A01, A12], "signed": [...], "tokens": n},
A_l with shape (n_latents at level l+1, n_latents at level l).
"""

from __future__ import annotations

import argparse
import time

import torch

from e03_common import OUT, load_target, residuals, token_loader
from percinterp.sae import TopKSAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-batches", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    dev = args.device
    model = load_target(dev)
    blob = torch.load(OUT / "sae_tower.pt", map_location=dev)
    saes = [TopKSAE.from_state(s).to(dev).eval() for s in blob["levels"]]
    for s in saes:
        for p in s.parameters():
            p.requires_grad_(False)
    n = [s.n for s in saes]
    A_abs = [torch.zeros(n[l + 1], n[l], device=dev, dtype=torch.float64) for l in range(2)]
    A_sgn = [torch.zeros_like(a) for a in A_abs]
    tokens = 0
    t0 = time.time()
    for b, idx in enumerate(token_loader("test", args.batch, seed=2)):
        if b >= args.n_batches:
            break
        idx = idx.to(dev)[:, :512]
        with torch.no_grad():
            levels = [r.float() for r in residuals(model, idx)]
        tokens += idx.numel()
        for l in range(2):
            src = saes[l]
            with torch.no_grad():
                a_src = src.encode(levels[l])  # (B, T, n_src)
                D = src.W_dec / src.scale  # residual units, (n_src, d)
            x = levels[l].detach().requires_grad_(True)
            y = model.h[l](x)
            a_tgt = saes[l + 1].encode(y)  # differentiable through the kept pre-activations
            active = (a_tgt.detach() > 0).flatten(0, 1).any(0).nonzero().flatten().tolist()
            for j in active:
                (g,) = torch.autograd.grad(a_tgt[..., j].sum(), x, retain_graph=True)
                contrib = a_src * (g @ D.T)  # (B, T, n_src)
                A_abs[l][j] += contrib.abs().sum((0, 1)).double()
                A_sgn[l][j] += contrib.sum((0, 1)).double()
            del y, a_tgt
        print(f"batch {b + 1}/{args.n_batches}: {tokens} tokens, {time.time() - t0:.0f}s", flush=True)
    torch.save({"abs": [a.cpu() for a in A_abs], "signed": [a.cpu() for a in A_sgn], "tokens": tokens},
               OUT / "sae_graph.pt")
    for l in range(2):
        nz = (A_abs[l] > 0).float().mean().item()
        print(f"level {l} -> {l + 1}: nonzero edge fraction {nz:.3f}")


if __name__ == "__main__":
    main()
