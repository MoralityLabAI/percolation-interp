"""e04: percolation sieve on the Qwen slice's VPD component graph (Q1).

Graph: alive components of layers 2-3 in residual-stream order -> one output node.
  - edges: |regular_attr_abs| between components; component -> output with the norm of the
    component's effect in logit space, sqrt(m^T (W W^T) m) for its residual-space attribution m;
  - sources: layer-2 components that read the residual stream (q, k, v, gate, up);
  - sink: the output node.
Sieved twice: against within-layer shuffles (the backbone used by Q2/Q3, as in e03) and against
in-strength-preserving shuffles.

  python experiments/e04_analyze.py      (this repo's venv)
Writes out/e04/analysis.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from percinterp import LayeredDAG
from percinterp.sieve import sieve

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("E04_OUT", ROOT / "out" / "e04"))
MIN_RATE = 1e-5
STAGE = {"attn.q": 0, "attn.k": 0, "attn.v": 0, "attn.o": 1, "glu.gate": 2, "glu.up": 2, "glu.down": 3}
CONCRETE = {"attn.q": "self_attn.q_proj", "attn.k": "self_attn.k_proj", "attn.v": "self_attn.v_proj",
            "attn.o": "self_attn.o_proj", "glu.gate": "mlp.gate_proj", "glu.up": "mlp.up_proj",
            "glu.down": "mlp.down_proj"}


def concrete(canon: str) -> str:
    layer, rest = canon.split(".", 1)
    return f"model.layers.{layer}.{CONCRETE[rest]}"


def key(canon: str):
    layer, rest = canon.split(".", 1)
    return int(layer), STAGE[rest], rest


def build_graph():
    A = torch.load(OUT / "dataset_attributions.pt", weights_only=True)
    stats = torch.load(OUT / "stats.pt", weights_only=False)
    reg = A["regular_attr_abs"]
    names = set(reg) | {s for t in reg for s in reg[t]} | set(A["unembed_attr"])
    layers = sorted((n for n in names if n.split(".", 1)[1] in STAGE), key=key)
    alive = {k: np.flatnonzero(stats["n_fire"][concrete(k)].numpy() / stats["tokens"] >= MIN_RATE) for k in layers}
    blocks = [(k, alive[k]) for k in layers] + [("output", np.array([0]))]
    sizes = [len(ix) for _, ix in blocks]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    pos = {name: i for i, (name, _) in enumerate(blocks)}
    src, dst, w = [], [], []

    def add(tgt, s, M):
        ti, si = pos[tgt], pos[s]
        M = np.abs(M[np.ix_(blocks[ti][1], blocks[si][1])])
        r, c = np.nonzero(M)
        dst.append(r + offsets[ti])
        src.append(c + offsets[si])
        w.append(M[r, c])

    for t in layers:
        for s, M in reg.get(t, {}).items():
            if s in pos and pos[s] < pos[t]:
                add(t, s, M.float().numpy())
    W = A["w_unembed"].float()  # (d_model, vocab)
    G = W @ W.T if W.shape[0] < W.shape[1] else W.T @ W  # (d_model, d_model)
    for s, M in A["unembed_attr"].items():
        if s in pos:
            M = M.float()
            norms = torch.sqrt(torch.einsum("ic,ij,jc->c", M, G, M).clamp(min=0)).numpy()
            add("output", s, norms[None, :])
    first = min(int(k.split(".")[0]) for k in layers)
    sources = np.zeros(offsets[-1], dtype=bool)
    for i, (name, ix) in enumerate(blocks):
        if name != "output" and int(name.split(".")[0]) == first and STAGE[name.split(".", 1)[1]] in (0, 2):
            sources[offsets[i]:offsets[i + 1]] = True
    sinks = np.zeros(offsets[-1], dtype=bool)
    sinks[-1] = True
    g = LayeredDAG(sizes, np.concatenate(src), np.concatenate(dst), np.concatenate(w), sources=sources, sinks=sinks)
    ids = [(name if name == "output" else concrete(name), int(i)) for name, ix in blocks for i in ix]
    return g, g.weight, ids, {k: len(v) for k, v in alive.items()}


def record(res, ids):
    return {"z_peak": res.z_peak, "null_peaks": res.null_peaks.tolist(), "p": res.p_value, "c_star": res.c_star,
            "learned_peak": res.peak, "backbone": [ids[i] for i in np.flatnonzero(res.backbone) if ids[i][0] != "output"],
            "cs": res.cs.tolist(), "real": res.real.tolist(), "shuffle_mean": res.shuffles.mean(0).tolist(),
            "z": res.z.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shuffles", type=int, default=100)
    args = ap.parse_args()
    g, w, ids, n_alive = build_graph()
    print(f"graph: {g.n_nodes} nodes, {g.n_edges} edges, {int(g.sources.sum())} sources; alive per module {n_alive}",
          flush=True)
    report = {"n_alive": n_alive, "n_nodes": g.n_nodes, "n_edges": g.n_edges}
    for name, kw in [("within_layer", {}), ("in_strength", {"null_groups": g.dst})]:
        res = sieve(g, w, n_shuffles=args.n_shuffles, rng=np.random.default_rng(len(name)), **kw)
        print(f"{name}: z_peak {res.z_peak:.1f} (shuffle max {res.null_peaks.max():.1f}), p {res.p_value:.3f}, "
              f"c* {res.c_star:.3g}, learned {res.peak:+.3f}, backbone {int(res.backbone.sum())} nodes", flush=True)
        report[name] = record(res, ids)
    report["Q1_pass"] = all(report[k]["z_peak"] > max(report[k]["null_peaks"]) for k in ("within_layer", "in_strength"))
    (OUT / "analysis.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
