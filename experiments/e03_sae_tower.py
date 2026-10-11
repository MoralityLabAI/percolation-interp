"""e03 stage: train the SAE tower (one TopK SAE per residual level) on the target.

  python experiments/e03_sae_tower.py --device cuda
Writes out/e03/sae_tower.pt and out/e03/sae_tower_eval.json.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from e03_common import OUT, load_target, logits_from_level, residuals, token_loader
from percinterp.sae import TopKSAE


def train_tower(args):
    device = args.device
    model = load_target(device)
    d = model.config.n_embd
    saes = [TopKSAE(d, args.n_latents, args.k).to(device) for _ in range(3)]
    opts = [torch.optim.Adam(s.parameters(), lr=args.lr, betas=(0.9, 0.999)) for s in saes]
    t0 = time.time()
    step = 0
    while step < args.steps:
        for idx in token_loader("train", args.batch, seed=args.seed):
            if step >= args.steps:
                break
            idx = idx.to(device)[:, :512]
            with torch.no_grad():
                levels = [r.reshape(-1, d).float() for r in residuals(model, idx)]
            logs = []
            for sae, opt, x in zip(saes, opts, levels):
                if step == 0:
                    sae.set_scale(x)
                loss, stats = sae.loss(x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                sae.remove_parallel_grad()
                opt.step()
                sae.normalize_decoder()
                logs.append(stats)
            if step % args.log_every == 0:
                print(f"step {step}: " + "  ".join(
                    f"L{l} fvu {s['fvu']:.3f} dead {s['dead_frac']:.2f}" for l, s in enumerate(logs)
                ) + f"  {time.time() - t0:.0f}s", flush=True)
            step += 1
    return model, saes


@torch.no_grad()
def evaluate(model, saes, args):
    """FVU and spliced-in CE per level, and per-latent firing counts, on the test split."""
    d = model.config.n_embd
    n_tok = 0
    sse = [0.0] * 3
    sst = [0.0] * 3
    ce_clean, ce_splice = 0.0, [0.0] * 3
    fire = [torch.zeros(s.n, device=args.device) for s in saes]
    for b, idx in enumerate(token_loader("test", args.batch, n_ctx=513, seed=1)):
        if b >= args.eval_batches:
            break
        idx = idx.to(args.device)
        inp, tgt = idx[:, :-1], idx[:, 1:]
        levels = residuals(model, inp)
        clean = logits_from_level(model, levels[0], 0)
        ce_clean += F.cross_entropy(clean.flatten(0, 1), tgt.flatten(), reduction="sum").item()
        for l, (sae, x) in enumerate(zip(saes, levels)):
            recon, a, _ = sae(x.float())
            sse[l] += (x.float() - recon).pow(2).sum().item()
            sst[l] += (x.float() - x.float().reshape(-1, d).mean(0)).pow(2).sum().item()
            fire[l] += (a > 0).reshape(-1, sae.n).sum(0)
            spliced = logits_from_level(model, recon.to(x.dtype), l)
            ce_splice[l] += F.cross_entropy(spliced.flatten(0, 1), tgt.flatten(), reduction="sum").item()
        n_tok += tgt.numel()
    return {
        "tokens": n_tok,
        "ce_clean": ce_clean / n_tok,
        "levels": [
            {
                "fvu": sse[l] / sst[l],
                "ce_spliced": ce_splice[l] / n_tok,
                "alive": int((fire[l] > 0).sum()),
                "alive_1e5": int((fire[l] / n_tok >= 1e-5).sum()),
                "firing_rate": (fire[l] / n_tok).tolist(),
            }
            for l in range(3)
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-latents", type=int, default=1024)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=40)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model, saes = train_tower(args)
    torch.save({"levels": [s.state() for s in saes], "args": vars(args)}, OUT / "sae_tower.pt")
    ev = evaluate(model, saes, args)
    (OUT / "sae_tower_eval.json").write_text(json.dumps(ev))
    print(f"CE clean {ev['ce_clean']:.3f}")
    for l, lv in enumerate(ev["levels"]):
        print(f"level {l}: FVU {lv['fvu']:.3f}, CE spliced {lv['ce_spliced']:.3f}, "
              f"alive {lv['alive']} (>=1e-5: {lv['alive_1e5']})")


if __name__ == "__main__":
    main()
