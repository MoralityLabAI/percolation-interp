"""e05 quality gate (docs/e05_design.md): KL(target || CI-masked) <= 0.25 x KL_PAMA.

KL_PAMA: the slice's residual writers (o_proj, down_proj) replaced by their position-class means
(position 0; positions 1-127), measured on the held-out "stats" rows of the target model. Both
KLs are measured on the "causal" rows, full vocabulary, all positions.

  python experiments/e05_gate.py      (param-decomp venv; E04_SEED / E04_SLICE / E04_OUT set)
Writes <E04_OUT>/gate.json.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from e04_common import OUT, SLICE, ci_lower, find_run, heldout_part, load_component_model

WRITERS = ["self_attn.o_proj", "mlp.down_proj"]
RATIO = 0.25


def kl_to(target, logits):
    logp = F.log_softmax(logits.float(), -1)
    return (target.exp() * (target - logp)).sum(-1).mean().item()


def writer_means(model, names, dev, n_batches=64, batch=4):
    s0 = {n: 0.0 for n in names}
    sr = {n: 0.0 for n in names}
    rows = rest = 0

    def acc(name):
        def hook(_, __, out):
            s0[name] = s0[name] + out[:, 0].float().sum(0)
            sr[name] = sr[name] + out[:, 1:].float().sum((0, 1))
        return hook

    hs = [model.get_submodule(n).register_forward_hook(acc(n)) for n in names]
    with torch.no_grad():
        for _, idx in zip(range(n_batches), heldout_part(batch, "stats")):
            model(idx.to(dev))
            rows += idx.shape[0]
            rest += idx.shape[0] * (idx.shape[1] - 1)
    for h in hs:
        h.remove()
    return {n: s0[n] / rows for n in names}, {n: sr[n] / rest for n in names}


def pama_hooks(model, names, m0, mr):
    def make(name):
        def hook(_, __, out):
            new = mr[name].to(out.dtype).expand_as(out).clone()
            new[:, 0] = m0[name].to(out.dtype)
            return new
        return hook
    return [model.get_submodule(n).register_forward_hook(make(n)) for n in names]


def main(n_batches: int = 32, n_mean_batches: int = 64, batch: int = 4, dev: str = "cuda"):
    from param_decomp.models.components import make_mask_infos

    run_id = find_run()
    cm, run_info = load_component_model(run_id, dev)
    names = [f"model.layers.{l}.{w}" for l in SLICE for w in WRITERS]
    m0, mr = writer_means(cm.target_model, names, dev, n_batches=n_mean_batches)
    kl_ci = kl_pama = 0.0
    with torch.no_grad():
        for _, idx in zip(range(n_batches), heldout_part(batch, "causal")):
            idx = idx.to(dev)
            target = F.log_softmax(cm(idx).float(), -1)
            ci = ci_lower(cm, run_info, idx)
            kl_ci += kl_to(target, cm(idx, mask_infos=make_mask_infos(ci))) / n_batches
            hs = pama_hooks(cm.target_model, names, m0, mr)
            kl_pama += kl_to(target, cm(idx)) / n_batches
            for h in hs:
                h.remove()
    res = {"run_id": run_id, "slice": list(SLICE), "kl_ci": kl_ci, "kl_pama": kl_pama, "ratio": kl_ci / kl_pama,
           "threshold_ratio": RATIO, "pass": bool(kl_ci <= RATIO * kl_pama), "tokens": n_batches * batch * 128}
    (OUT / "gate.json").write_text(json.dumps(res, indent=1))
    print(f"run {run_id}: KL(CI-masked) {kl_ci:.4f}, KL_PAMA {kl_pama:.4f}, ratio {res['ratio']:.3f}: "
          f"{'pass' if res['pass'] else 'FAIL'}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-batches", type=int, default=32)
    ap.add_argument("--n-mean-batches", type=int, default=64)
    a = ap.parse_args()
    main(a.n_batches, a.n_mean_batches, dev=a.device)
