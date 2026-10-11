"""e03 analysis: percolation sieve on both graphs (P1), cross-method enrichment (P2, P3).

SAE edge statistic (pre-registered, docs/e03_design.md): sum over tokens of the per-token
|a_i (d_i . grad_j)|, the "abs" entry of out/e03/sae_graph.pt. The "signed" entry
(|sum of signed contributions|, the analogue of param-decomp's attr_abs used for the VPD
graph) is reported as a secondary variant with --sae-edge signed.

  python experiments/e03_analyze.py
Writes out/e03/analysis.json and figures under out/e03/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from percinterp import LayeredDAG
from percinterp.sieve import sieve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "e03"
MIN_RATE = 1e-5  # alive: fires on at least this fraction of tokens
MIN_CO = 5  # PMI only for pairs that co-fire at least this often


# --- graphs -----------------------------------------------------------------------------


def sae_graph(edge: str = "abs"):
    """LayeredDAG over alive SAE features: level 0 -> 1 -> 2. Returns (g, weight, node ids)."""
    ev = json.loads((OUT / "sae_tower_eval.json").read_text())
    G = torch.load(OUT / "sae_graph.pt")
    alive = [np.flatnonzero(np.array(lv["firing_rate"]) >= MIN_RATE) for lv in ev["levels"]]
    mats = [np.abs(G[edge][l].numpy())[np.ix_(alive[l + 1], alive[l])] for l in range(2)]
    g = LayeredDAG.from_dense(mats)
    ids = [(lvl, int(i)) for lvl in range(3) for i in alive[lvl]]  # node order = layer order
    return g, g.weight, ids


CANON_TO_MODULE = {"attn.q": "attn.q_proj", "attn.k": "attn.k_proj", "attn.v": "attn.v_proj",
                   "attn.o": "attn.o_proj", "mlp.up": "mlp.c_fc", "mlp.down": "mlp.down_proj"}
STAGE = {"attn.q": 0, "attn.k": 0, "attn.v": 0, "attn.o": 1, "mlp.up": 2, "mlp.down": 3}


def canon_to_module(key: str) -> str:
    """'0.attn.q' -> 'h.0.attn.q_proj' (param-decomp's LlamaSimpleMLP path schema)."""
    layer, rest = key.split(".", 1)
    return f"h.{layer}.{CANON_TO_MODULE[rest]}"


def attributions_path() -> Path:
    run_id = (OUT / "vpd_run.txt").read_text().strip()
    root = Path.home() / "param_decomp_out" / "dataset_attributions" / run_id
    found = sorted(root.glob("da-*/dataset_attributions.pt"))
    if not found:
        raise SystemExit(f"no dataset_attributions.pt under {root}")
    return found[-1]


def vpd_graph(comp_rate: dict[str, np.ndarray]):
    """LayeredDAG: embed tokens -> components (residual-stream order) -> output.

    comp_rate: per module path, each component's firing rate (CI > threshold), from crossref.pt.
    Alive components fire on >= MIN_RATE of tokens. Edges: |regular_attr_abs| between
    components, |embed_attr_abs| from tokens seen in the data, and component -> output with the
    norm of the component's effect in logit space (w_unembed @ unembed_attr).
    Returns (g, weight, ids) with ids[i] = ("embed", token) | (module path, component) | ("output", 0).
    """
    A = torch.load(attributions_path(), weights_only=True)
    layers = sorted({t for t in A["regular_attr_abs"]} | {t for t in A["embed_attr_abs"]},
                    key=lambda k: (int(k.split(".")[0]), STAGE[k.split(".", 1)[1]], k))
    alive = {k: np.flatnonzero(comp_rate[canon_to_module(k)] >= MIN_RATE) for k in layers}
    tokens = np.flatnonzero(A["embed_token_count"].numpy() > 0)
    blocks = [("embed", tokens)] + [(k, alive[k]) for k in layers] + [("output", np.array([0]))]
    sizes = [len(ix) for _, ix in blocks]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    pos = {name: i for i, (name, _) in enumerate(blocks)}
    src, dst, w = [], [], []

    def add(tgt_name, src_name, M):
        ti, si = pos[tgt_name], pos[src_name]
        M = np.abs(M[np.ix_(blocks[ti][1], blocks[si][1])])
        r, c = np.nonzero(M)
        dst.append(r + offsets[ti])
        src.append(c + offsets[si])
        w.append(M[r, c])

    for t in layers:
        for s, M in A["regular_attr_abs"].get(t, {}).items():
            if s in pos and pos[s] < pos[t]:
                add(t, s, M.float().numpy())
        if t in A["embed_attr_abs"]:
            add(t, "embed", A["embed_attr_abs"][t].float().numpy())
    W_U = A["w_unembed"].float()
    for s, M in A["unembed_attr"].items():
        if s in pos:
            W = W_U if W_U.shape[1] == M.shape[0] else W_U.T  # orient as (vocab, d_model)
            logit_effect = (W @ M.float()).norm(dim=0).numpy()  # (src_c,)
            add("output", s, logit_effect[None, :])
    g = LayeredDAG(sizes, np.concatenate(src), np.concatenate(dst), np.concatenate(w))
    ids = [(name if name in ("embed", "output") else canon_to_module(name), int(i))
           for name, ix in blocks for i in ix]
    return g, g.weight, ids


# --- cross-reference enrichment -------------------------------------------------------------


def pmi_matrix(X):
    co, nc, nf, N = (X["co"].numpy(), X["n_comp"].numpy(), X["n_feat"].numpy(), X["tokens"])
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((co * N) / (nc[:, None] * nf[None, :]))
    pmi[(co < MIN_CO) | ~np.isfinite(pmi)] = np.nan
    return pmi


def best_match(score, comps, feats_by_comp):
    """Mean over components of their best score against the allowed features (nan -> 0)."""
    vals = []
    for c in comps:
        f = feats_by_comp(c)
        s = score[c, f] if len(f) else np.array([])
        s = s[np.isfinite(s)]
        vals.append(s.max() if len(s) else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def enrichment(score, comp_backbone, comp_pool_by_layer, feat_backbone, feat_pool, feat_rate, allowed_levels,
               n_draws=1000, rng=None):
    """P2/P3 statistic for the backbone sets against size-, layer- and rate-matched random sets.

    comp_backbone: {layer key: [component idx]}; comp_pool_by_layer: {layer key: alive idx}.
    feat_backbone / feat_pool: {level: [feature idx]} (global feature indices).
    allowed_levels(c) -> levels whose features component c is compared against.
    """
    rng = rng or np.random.default_rng(0)

    def feats_for(sets):
        return lambda c: np.array([f for lvl in allowed_levels(c) for f in sets.get(lvl, [])], dtype=int)

    comps = [c for v in comp_backbone.values() for c in v]
    obs = best_match(score, comps, feats_for(feat_backbone))
    deciles = {lvl: np.quantile(feat_rate[pool], np.linspace(0, 1, 11)) for lvl, pool in feat_pool.items()}

    def rate_matched(lvl, chosen):
        pool = np.asarray(feat_pool[lvl])
        bins = np.clip(np.searchsorted(deciles[lvl], feat_rate[pool], side="right") - 1, 0, 9)
        want = np.clip(np.searchsorted(deciles[lvl], feat_rate[np.asarray(chosen)], side="right") - 1, 0, 9)
        out = []
        for b in want:
            cand = pool[bins == b]
            out.append(rng.choice(cand))
        return out

    null = []
    for _ in range(n_draws):
        rc = [c for key, v in comp_backbone.items() for c in rng.choice(comp_pool_by_layer[key], len(v), replace=False)]
        rf = {lvl: rate_matched(lvl, v) for lvl, v in feat_backbone.items() if len(v)}
        null.append(best_match(score, rc, feats_for(rf)))
    null = np.array(null)
    return {"observed": obs, "null_median": float(np.median(null)), "ratio": obs / float(np.median(null)),
            "p_value": float((1 + (null >= obs).sum()) / (1 + n_draws))}


# --- main -------------------------------------------------------------------------------------


def run_sieve(name, g, w, n_shuffles, seed, **kw):
    res = sieve(g, w, n_shuffles=n_shuffles, rng=np.random.default_rng(seed), **kw)
    print(f"{name}: z_peak {res.z_peak:.1f} (shuffle peaks max {res.null_peaks.max():.1f}), "
          f"p {res.p_value:.3f}, c* {res.c_star:.3g}, learned {res.peak:+.3f}, "
          f"backbone {int(res.backbone.sum())}/{g.n_nodes} nodes", flush=True)
    return res


def sieve_record(res, ids, extra=None):
    return {**(extra or {}), "z_peak": res.z_peak, "null_peaks": res.null_peaks.tolist(), "p": res.p_value,
            "c_star": res.c_star, "learned_peak": res.peak,
            "backbone": [ids[i] for i in np.flatnonzero(res.backbone)],
            "cs": res.cs.tolist(), "real": res.real.tolist(), "shuffle_mean": res.shuffles.mean(0).tolist(),
            "z": res.z.tolist()}


def load_crossref():
    X = torch.load(OUT / "crossref.pt", weights_only=False)
    off = X["comp_offsets"]
    rate = X["n_comp"].numpy() / X["tokens"]
    comp_rate = {m: rate[off[i]:off[i + 1]] for i, m in enumerate(X["modules"])}
    return X, comp_rate


def layer_of(module: str) -> int:
    return int(module.split(".")[1])


def enrichment_tests(X, vpd_backbone_ids, sae_backbone_ids, comp_rate, n_draws, seed):
    """P2 (co-firing PMI) and P3 (weight-space |cos|) for the sieved sets against matched random sets."""
    modules, off, foff = X["modules"], X["comp_offsets"], X["feat_offsets"]
    midx = {m: i for i, m in enumerate(modules)}
    n_comp, n_feat = off[-1], foff[-1]
    feat_rate = X["n_feat"].numpy() / X["tokens"]
    comp_module = np.empty(n_comp, dtype=object)
    for i, m in enumerate(modules):
        comp_module[off[i]:off[i + 1]] = m

    comp_bb, comp_pool = {}, {}
    for m in modules:
        comp_pool[m] = off[midx[m]] + np.flatnonzero(comp_rate[m] >= MIN_RATE)
    for m, c in vpd_backbone_ids:
        if m in midx:
            comp_bb.setdefault(m, []).append(off[midx[m]] + c)
    comp_pool = {m: comp_pool[m] for m in comp_bb}
    feat_bb, feat_pool = {}, {}
    for lvl in range(len(foff) - 1):
        feat_pool[lvl] = foff[lvl] + np.flatnonzero(feat_rate[foff[lvl]:foff[lvl + 1]] >= MIN_RATE)
    for lvl, i in sae_backbone_ids:
        feat_bb.setdefault(lvl, []).append(foff[lvl] + i)

    pmi = pmi_matrix(X)
    cos = np.full((n_comp, n_feat), np.nan)
    for m, M in X["cos_read"].items():
        l = layer_of(m)
        cos[off[midx[m]]:off[midx[m] + 1], foff[l]:foff[l + 1]] = np.abs(M.numpy())
    for m, M in X["cos_write"].items():
        l = layer_of(m) + 1
        cos[off[midx[m]]:off[midx[m] + 1], foff[l]:foff[l + 1]] = np.abs(M.numpy())
    is_write = np.array([m.endswith(("o_proj", "down_proj")) for m in comp_module])

    def levels_cofire(c):
        l = layer_of(comp_module[c])
        return [l, l + 1]

    def levels_weight(c):
        l = layer_of(comp_module[c])
        return [l + 1] if is_write[c] else [l]

    out = {
        "n_backbone_components": int(sum(len(v) for v in comp_bb.values())),
        "n_backbone_features": int(sum(len(v) for v in feat_bb.values())),
    }
    out["P2"] = enrichment(pmi, comp_bb, comp_pool, feat_bb, feat_pool, feat_rate, levels_cofire,
                           n_draws=n_draws, rng=np.random.default_rng(seed))
    out["P3"] = enrichment(cos, comp_bb, comp_pool, feat_bb, feat_pool, feat_rate, levels_weight,
                           n_draws=n_draws, rng=np.random.default_rng(seed + 1))
    for k in ("P2", "P3"):
        r = out[k]
        print(f"{k}: observed {r['observed']:.3f}, null median {r['null_median']:.3f}, "
              f"ratio {r['ratio']:.2f}, p {r['p_value']:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shuffles", type=int, default=100)
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--only", choices=["sae", "all"], default="all")
    ap.add_argument("--sae-edge", choices=["abs", "signed"], default="abs")
    args = ap.parse_args()
    report = {}
    g, w, sae_ids = sae_graph(args.sae_edge)
    sae_res = run_sieve(f"SAE graph ({args.sae_edge} edges)", g, w, args.n_shuffles, 0)
    report["sae_sieve"] = sieve_record(sae_res, sae_ids, {"edge": args.sae_edge})
    if args.only == "all":
        X, comp_rate = load_crossref()
        g, w, vpd_ids = vpd_graph(comp_rate)
        vpd_res = run_sieve("VPD graph", g, w, args.n_shuffles, 1)
        report["vpd_sieve"] = sieve_record(vpd_res, vpd_ids)
        vpd_bb = [i for i in report["vpd_sieve"]["backbone"] if i[0] not in ("embed", "output")]
        sae_bb = [tuple(i) for i in report["sae_sieve"]["backbone"]]
        report["enrichment"] = enrichment_tests(X, vpd_bb, sae_bb, comp_rate, args.n_draws, 2)
    suffix = "" if args.sae_edge == "abs" else f"_{args.sae_edge}"
    (OUT / f"analysis{suffix}.json").write_text(json.dumps(report, default=str))


if __name__ == "__main__":
    main()
