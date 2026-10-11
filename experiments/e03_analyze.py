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


def vpd_graph():
    """LayeredDAG over alive VPD components (built from param-decomp's dataset attributions)."""
    raise NotImplementedError("filled in once the attribution file's layout has been inspected")


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


def run_sieve(name, g, w, n_shuffles, seed):
    res = sieve(g, w, n_shuffles=n_shuffles, rng=np.random.default_rng(seed))
    print(f"{name}: z_peak {res.z_peak:.1f} (shuffle peaks max {res.null_peaks.max():.1f}), "
          f"p {res.p_value:.3f}, c* {res.c_star:.3g}, learned {res.peak:+.3f}, "
          f"backbone {int(res.backbone.sum())}/{g.n_nodes} nodes", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shuffles", type=int, default=100)
    ap.add_argument("--only", choices=["sae", "vpd", "all"], default="all")
    ap.add_argument("--sae-edge", choices=["abs", "signed"], default="abs")
    args = ap.parse_args()
    report = {}
    if args.only in ("sae", "all"):
        g, w, ids = sae_graph(args.sae_edge)
        res = run_sieve(f"SAE graph ({args.sae_edge} edges)", g, w, args.n_shuffles, 0)
        report["sae_sieve"] = {"edge": args.sae_edge, "z_peak": res.z_peak, "null_peaks": res.null_peaks.tolist(), "p": res.p_value,
                               "c_star": res.c_star, "learned_peak": res.peak,
                               "backbone": [ids[i] for i in np.flatnonzero(res.backbone)],
                               "cs": res.cs.tolist(), "real": res.real.tolist(),
                               "shuffle_mean": res.shuffles.mean(0).tolist(), "z": res.z.tolist()}
    suffix = "" if args.sae_edge == "abs" else f"_{args.sae_edge}"
    (OUT / f"analysis{suffix}.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
