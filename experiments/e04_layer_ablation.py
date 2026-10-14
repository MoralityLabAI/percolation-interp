"""e04 slice selection: KL from zeroing all seven projections of each consecutive layer pair.

Uses only the target model (no decomposition, no Q1-Q3 quantity), on held-out pile-10k rows.

  python experiments/e04_layer_ablation.py
Writes out/e04/layer_ablation.json.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from e04_common import MODEL, OUT, heldout_loader

PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def main(n_batches: int = 16, batch: int = 4):
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    n_layers = model.config.num_hidden_layers
    batches = [b.cuda() for _, b in zip(range(n_batches), heldout_loader(batch, seed=7))]
    with torch.no_grad():
        target = [F.log_softmax(model(b).logits.float(), -1) for b in batches]
        results = {}
        for L in range(n_layers - 1):
            saved = {}
            for l in (L, L + 1):
                for p in PROJ:
                    mod = model.model.layers[l].get_submodule(p)
                    saved[(l, p)] = mod.weight.data.clone()
                    mod.weight.data.zero_()
            kl = 0.0
            for b, t in zip(batches, target):
                logp = F.log_softmax(model(b).logits.float(), -1)
                kl += (t.exp() * (t - logp)).sum(-1).mean().item() / len(batches)
            for (l, p), w in saved.items():
                model.model.layers[l].get_submodule(p).weight.data.copy_(w)
            results[f"{L}-{L + 1}"] = kl
            print(f"layers {L}-{L + 1}: KL {kl:.3f}", flush=True)
    (OUT / "layer_ablation.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
