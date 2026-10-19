"""e05 slice selection (docs/e05_design.md), target model only, held-out "stats" rows.

1. For every layer's residual writers (o_proj, down_proj): mean output norm at position 0 and at
   positions 1-127. A layer is excluded if a writer's position-0 norm exceeds 100x its norm
   elsewhere (a massive-activation writer).
2. Position-class means of every writer's output, then KL_PAMA for every consecutive layer pair:
   KL(target || both layers' writers replaced by their position-class means).
3. Selected: the non-excluded pair with the largest KL_PAMA.

  python experiments/e05_select.py
Writes out/e05/select.json (and results/e05/select.json).
"""

from __future__ import annotations

import json
import shutil

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from e04_common import MODEL, ROOT, heldout_part

OUT = ROOT / "out" / "e05"
RES = ROOT / "results" / "e05"
WRITERS = ["self_attn.o_proj", "mlp.down_proj"]
MASSIVE = 100.0


def main(n_batches: int = 64, batch: int = 4):
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda().eval()
    L = model.config.num_hidden_layers
    batches = [b.cuda() for _, b in zip(range(n_batches), heldout_part(batch, "stats"))]
    names = [f"model.layers.{l}.{w}" for l in range(L) for w in WRITERS]
    s0 = {n: 0.0 for n in names}
    sr = {n: 0.0 for n in names}
    n0 = {n: 0.0 for n in names}
    nr = {n: 0.0 for n in names}

    def acc(name):
        def hook(_, __, out):
            o = out.float()
            s0[name] = s0[name] + o[:, 0].sum(0)
            sr[name] = sr[name] + o[:, 1:].sum((0, 1))
            n0[name] += o[:, 0].norm(dim=-1).sum().item()
            nr[name] += o[:, 1:].norm(dim=-1).sum().item()
        return hook

    rows = sum(b.shape[0] for b in batches)
    rest = sum(b.shape[0] * (b.shape[1] - 1) for b in batches)
    with torch.no_grad():
        hs = [model.get_submodule(n).register_forward_hook(acc(n)) for n in names]
        targets = [F.log_softmax(model(b).logits.float(), -1) for b in batches]
        for h in hs:
            h.remove()
        m0 = {n: s0[n] / rows for n in names}
        mr = {n: sr[n] / rest for n in names}
        norms = {n: {"pos0": n0[n] / rows, "rest": nr[n] / rest} for n in names}
        massive = sorted({l for l in range(L) for w in WRITERS
                          if norms[f"model.layers.{l}.{w}"]["pos0"] > MASSIVE * norms[f"model.layers.{l}.{w}"]["rest"]})

        def pama(name):
            def hook(_, __, out):
                new = mr[name].to(out.dtype).expand_as(out).clone()
                new[:, 0] = m0[name].to(out.dtype)
                return new
            return hook

        kl = {}
        for a in range(L - 1):
            hs = [model.get_submodule(f"model.layers.{l}.{w}").register_forward_hook(pama(f"model.layers.{l}.{w}"))
                  for l in (a, a + 1) for w in WRITERS]
            tot = 0.0
            for b, t in zip(batches, targets):
                logp = F.log_softmax(model(b).logits.float(), -1)
                tot += (t.exp() * (t - logp)).sum(-1).mean().item() / len(batches)
            for h in hs:
                h.remove()
            kl[f"{a}-{a + 1}"] = tot
            print(f"layers {a}-{a + 1}: KL_PAMA {tot:.4f}{'  (excluded)' if a in massive or a + 1 in massive else ''}",
                  flush=True)
    eligible = {k: v for k, v in kl.items() if not any(int(x) in massive for x in k.split("-"))}
    chosen = max(eligible, key=eligible.get)
    res = {"rows": rows, "massive_threshold": MASSIVE, "excluded_layers": massive,
           "writer_norms": {n.replace("model.layers.", ""): v for n, v in norms.items()},
           "kl_pama": kl, "selected": chosen, "selected_kl_pama": kl[chosen]}
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    (OUT / "select.json").write_text(json.dumps(res, indent=1))
    shutil.copyfile(OUT / "select.json", RES / "select.json")
    print(f"excluded layers {massive}; selected {chosen} (KL_PAMA {kl[chosen]:.4f})")


if __name__ == "__main__":
    main()
