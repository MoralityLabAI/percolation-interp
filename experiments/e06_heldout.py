"""e06 held-out run (docs/e06_design.md, pre-registered): nulls, diagnostics and P1-P5 on one graph.

CPU only. Fixed c: c_s = 0.1, c = 1, c_L = 4, plus the sieve's 48-point grid for the grid-wide
max-statistic (secondary). 1,000 draws each of N-WL, N-OS, N-IS, N-LP (skipped if the same as
N-WL), N-AB and N-DC; N-JD at c_s and c = 1 (30 rounds, skeleton edges only, (source layer,
destination layer) blocks); P5's 1,000 within-layer permutations of out-strength.

Graphs (one per run):
  --e03-dir DIR    e03 toy-model seed directory (crossref.pt, vpd_run.txt): its VPD graph, built
                   exactly as e06_explore.py built e03 seed 0's. P3 is reported only.
                   N-JD uses the parallel swap sampler plus the mixing check against exact
                   sequential swaps on --mix-n draws.
  --e05b-dir DIR   e05b output directory (dataset_attributions.pt, stats.pt, gate.json): the
                   Qwen-slice graph of e04_analyze.build_graph. All of P1-P5 count; the gate's
                   pass status is reported alongside. N-JD uses exact sequential swaps.
  --qwen-dir DIR   any other directory in that layout (e.g. e04, as a code check): scored, but
                   nothing counts.

The e05b run, once out/e05b_seed0 and out/e05b_seed1 hold their decompositions (one command):

  PYTHONPATH=src .venv/Scripts/python.exe -u experiments/e06_heldout.py \
      --e05b-dir <main checkout>/out/e05b_seed0 <main checkout>/out/e05b_seed1

writes results/e06/heldout_e05b_seed{0,1}.json (scored), and per-draw arrays to
out/e06_heldout/. Then `python experiments/e06_score.py results/e06/heldout_*.json` prints
P1-P5 per graph and whether each prediction holds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: torch is used to read the inputs, never on a GPU

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import numpy as np  # noqa: E402
from scipy.stats import ks_2samp, rankdata  # noqa: E402

from e06_score import score, summary_line  # noqa: E402
from percinterp import LayeredDAG  # noqa: E402
from percinterp.diagnostics import (backbone_anatomy, binned_groups, decompose, decompose_fast, default_grid,  # noqa: E402
                                    degree_determined_share, joint_degree_null, joint_degree_null_sequential,
                                    kept_coverage, node_strength, pair_groups, skeleton_keys,
                                    strength_correlation)
from percinterp.sieve import degree_entry  # noqa: E402

FIXED = {"c_s": 0.1, "c_1": 1.0, "c_L": 4.0}  # design, "Choice of c"
JD_CELLS = ("c_s", "c_1")  # design, P4
N_DRAWS = 1000
JD_ROUNDS = 30
Z_MIN = 3.0
# One rng seed per null, fixed before any held-out graph was analysed.
SEEDS = {"within_layer": 6101, "out_strength": 6102, "in_strength": 6103, "layer_pair": 6104,
         "activity_binned": 6105, "in_out_decoupled": 6106, "joint_degree": 6107, "joint_degree_seq": 6108,
         "strength_perm": 6109}
LABEL = {"within_layer": "N-WL", "out_strength": "N-OS", "in_strength": "N-IS", "layer_pair": "N-LP",
         "activity_binned": "N-AB", "in_out_decoupled": "N-DC"}


# --- graphs ---------------------------------------------------------------------------------


def build_e03_vpd(d: Path):
    """e03 VPD graph and firing rates, as e06_explore.build('e03_vpd') with OUT = d."""
    import torch

    import e03_analyze as ea
    ea.OUT = d
    _, comp_rate = ea.load_crossref()
    g, _, ids = ea.vpd_graph(comp_rate)
    A = torch.load(ea.attributions_path(), weights_only=True, map_location="cpu")
    tok = A["embed_token_count"].numpy().astype(float)
    tok_rate = tok / tok.sum()
    rate = np.array([tok_rate[i] if m == "embed" else 1.0 if m == "output" else comp_rate[m][i] for m, i in ids])
    return g, ids, rate


def build_qwen(d: Path):
    """Qwen-slice graph (e04_analyze.build_graph with OUT = d) and stats.pt firing rates."""
    import torch

    import e04_analyze as e4
    e4.OUT = d
    g, _, ids, _ = e4.build_graph()
    stats = torch.load(d / "stats.pt", weights_only=False, map_location="cpu")
    rate = np.array([1.0 if m == "output" else stats["n_fire"][m][i].item() / stats["tokens"] for m, i in ids])
    return g, ids, rate


def load_graph(kind: str, d: Path, cache: Path):
    path = cache / "graph.npz"
    if not path.exists():
        g, ids, rate = build_e03_vpd(d) if kind == "e03" else build_qwen(d)
        names = [ids[int(g.offsets[l])][0] for l in range(g.n_layers)]
        cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, src=g.src, dst=g.dst, w=g.weight, sizes=g.layer_sizes, sources=g.sources,
                            sinks=g.sinks, rate=rate, names=np.array(names))
    z = np.load(path)
    g = LayeredDAG(list(z["sizes"]), z["src"], z["dst"], z["w"], sources=z["sources"], sinks=z["sinks"])
    return g, [str(x) for x in z["names"]], z["rate"]


# --- weight nulls and N-DC: decomposition at the fixed c and on the grid ----------------------


def _z(x, ref, floor):
    return (x - ref.mean(0)) / np.maximum(ref.std(0), floor)


def null_record(dn, n_grid: int, floor: float) -> tuple[dict, dict]:
    """Summary of one null (fixed-c cells and the grid's max-statistic) and its per-draw arrays."""
    cs = dn.real.cs
    real = dn.real.total("b")
    sh = np.array([d.total("b") for d in dn.shuffles])
    n = len(sh)
    w = dn.real.weights

    # grid: the sieve's statistics on the 48-point grid only
    gr, gs = real[:n_grid], sh[:, :n_grid]
    z = _z(gr, gs, floor)
    learned = gr - gs.mean(0)
    peaks = np.array([_z(gs[i], np.delete(gs, i, axis=0), floor).max() for i in range(n)])
    ok = z >= Z_MIN
    k = int(np.argmax(np.where(ok, learned, -np.inf))) if ok.any() else int(np.argmax(learned))
    grid = {"cs": cs[:n_grid].tolist(), "z_curve": z.tolist(), "learned_curve": learned.tolist(),
            "real_curve": gr.tolist(), "null_mean_curve": gs.mean(0).tolist(),
            "excess_coverage_curve": dn.excess("coverage")[:n_grid].tolist(),
            "excess_cov_curve": dn.excess("cov")[:n_grid].tolist(),
            "z_peak": float(z.max()), "c_peak": float(cs[int(np.argmax(z))]),
            "null_peaks_max": float(peaks.max()), "p": float((1 + (peaks >= z.max()).sum()) / (1 + n)),
            "c_star": float(cs[k]), "z_min": float(z.min()), "c_zmin": float(cs[int(np.argmin(z))])}

    fixed = {}
    for j, (label, c) in enumerate(FIXED.items()):
        kk = n_grid + j
        assert np.isclose(cs[kk], c)
        sd = float(sh[:, kk].std())
        rec = {"c": c, "real": float(real[kk]), "null_mean": float(sh[:, kk].mean()), "null_sd": sd,
               "z": float((real[kk] - sh[:, kk].mean()) / max(sd, floor)),
               "p_high": float((1 + (sh[:, kk] >= real[kk]).sum()) / (1 + n)),
               "p_low": float((1 + (sh[:, kk] <= real[kk]).sum()) / (1 + n)), "excess": {}, "per_layer_excess": {}}
        for t in ("b", "coverage", "cov", "rF", "rB"):
            real_l = getattr(dn.real, t)[:, kk]
            null_l = dn.stack(t)[:, :, kk].mean(0)
            rec["excess"][t] = float((w * (real_l - null_l)).sum())
            rec[f"real_{t}"] = float((w * real_l).sum())
            rec["per_layer_excess"][t] = (w * (real_l - null_l)).tolist()
        fixed[label] = rec
    arrays = {"b_fixed": sh[:, n_grid:], "null_peaks": peaks}
    return {"grid": grid, "fixed": fixed, "layers_used": dn.real.layers.tolist(), "n_draws": n}, arrays


def weight_nulls(g, rate) -> dict:
    groups = g.edge_layer("dst")
    nulls = {"within_layer": groups, "out_strength": g.src, "in_strength": g.dst}
    if len(np.unique(pair_groups(g))) > len(np.unique(groups)):
        nulls["layer_pair"] = pair_groups(g)
    nulls["activity_binned"] = binned_groups(g, groups, rate, rate, n_bins=5)
    nulls["in_out_decoupled"] = "decouple"
    return nulls


# --- P5: strength correlation ---------------------------------------------------------------


def strength_test(g, rate, n_perm: int, rng) -> tuple[dict, np.ndarray]:
    """Pooled Spearman rho(in, out) over interior nodes with both strengths > 0 (ranks standardised
    within each layer, as e06_report.rate_partial), its rate-partialled version, and a one-sided
    permutation p against n_perm within-layer permutations of out-strength."""
    s_in, s_out = node_strength(g)
    ok = ~(g.sources | g.sinks) & (s_in > 0) & (s_out > 0)
    xs, ys, rs, bounds = [], [], [], [0]
    for l in range(g.n_layers):
        m = ok & (g.node_layer == l)
        if m.sum() < 3:
            continue
        for arr, out in ((s_in, xs), (s_out, ys), (rate, rs)):
            r = rankdata(arr[m])
            out.append((r - r.mean()) / max(r.std(), 1e-12))
        bounds.append(bounds[-1] + int(m.sum()))
    x, y, r = map(np.concatenate, (xs, ys, rs))

    def resid(v):
        return v - r * (v @ r) / (r @ r) if (r @ r) > 0 else v

    rho = float(np.corrcoef(x, y)[0, 1])
    perm = np.empty(n_perm)
    for k in range(n_perm):
        yp = np.concatenate([rng.permutation(y[a:b]) for a, b in zip(bounds[:-1], bounds[1:])])
        perm[k] = np.corrcoef(x, yp)[0, 1]
    sc = strength_correlation(g)
    return {"rho": rho, "rho_given_rate": float(np.corrcoef(resid(x), resid(y))[0, 1]),
            "rho_in_rate": float(np.corrcoef(x, r)[0, 1]), "rho_out_rate": float(np.corrcoef(y, r)[0, 1]),
            "p_perm": float((1 + (perm >= rho).sum()) / (1 + n_perm)), "perm_mean": float(perm.mean()),
            "perm_sd": float(perm.std()), "perm_max": float(perm.max()), "n_nodes": int(len(x)),
            "n_perm": n_perm, "pooled_scaled_rank_rho": sc["pooled_within_group_rho"],
            "per_layer_rho": {str(k): v["rho"] for k, v in sc["per_group"].items()}}, perm


# --- N-JD -------------------------------------------------------------------------------------


def jd_summary(r) -> dict:
    nl = r.null
    return {"real": r.real, "null_mean": float(nl.mean()), "null_sd": float(nl.std()), "null_min": float(nl.min()),
            "null_max": float(nl.max()), "testable": bool(nl.max() > nl.min()), "z": r.z, "p_high": r.p_high,
            "p_low": r.p_low, "n_edges": r.n_edges, "acceptance": r.acceptance, "n_draws": len(nl)}


def joint_degree(g, E, sampler: str, n: int, mix_n: int) -> tuple[dict, dict]:
    out, arrays = {"sampler": sampler, "rounds": JD_ROUNDS, "blocks": "pair", "skeleton": True, "cells": {}}, {}
    allowed = None
    if sampler == "sequential" or mix_n > 0:
        allowed = skeleton_keys(g, E)
    rng = np.random.default_rng(SEEDS["joint_degree"])
    rng_seq = np.random.default_rng(SEEDS["joint_degree_seq"])
    for label in JD_CELLS:
        c = FIXED[label]
        t = time.time()
        dd = degree_determined_share(g, E, c)
        if sampler == "sequential":
            r = joint_degree_null_sequential(g, E, c, n_null=n, rng=rng, n_rounds=JD_ROUNDS, allowed=allowed)
        else:
            r = joint_degree_null(g, E, c, n_null=n, rng=rng, n_rounds=JD_ROUNDS)
        rec = {"c": c, **jd_summary(r), "degree_determined": dd, "seconds": time.time() - t}
        arrays[f"jd_{label}"] = r.null
        print(f"  N-JD {label} ({sampler}): real {r.real:.4f}, null {r.null.mean():.4f} +- {r.null.std():.4f} "
              f"[{r.null.min():.4f}, {r.null.max():.4f}], p_high {r.p_high:.4f}, {r.n_edges} edges, acceptance "
              f"{r.acceptance:.3f}, S {dd['share_equal']:.4f}; {time.time() - t:.0f}s", flush=True)
        if sampler == "parallel" and mix_n > 0:
            t = time.time()
            q = joint_degree_null_sequential(g, E, c, n_null=mix_n, rng=rng_seq, n_rounds=JD_ROUNDS, allowed=allowed)
            a, b = r.null, q.null
            spread = a.max() > a.min() or b.max() > b.min()
            ks = ks_2samp(a, b) if spread else None
            se = np.sqrt(a.var() / len(a) + b.var() / len(b))
            rec["mixing_check"] = {
                "n_sequential": mix_n, "sequential": jd_summary(q),
                "mean_diff": float(b.mean() - a.mean()),
                "mean_diff_z": float((b.mean() - a.mean()) / se) if se > 0 else None,
                "sd_ratio": float(b.std() / a.std()) if a.std() > 0 else None,
                "ks_stat": float(ks.statistic) if ks is not None else 0.0,
                "ks_p": float(ks.pvalue) if ks is not None else 1.0,
                "p_high_sequential": q.p_high, "seconds": time.time() - t}
            arrays[f"jd_{label}_sequential"] = b
            m = rec["mixing_check"]
            print(f"  mixing check {label}: sequential {b.mean():.4f} +- {b.std():.4f} [{b.min():.4f}, {b.max():.4f}] "
                  f"(acceptance {q.acceptance:.3f}) vs parallel {a.mean():.4f} +- {a.std():.4f}; KS p {m['ks_p']:.3f}; "
                  f"{m['seconds']:.0f}s", flush=True)
        out["cells"][label] = rec
    return out, arrays


# --- reported, not tested ---------------------------------------------------------------------


def coverage_summary(g, E, c) -> dict:
    interior = ~(g.sources | g.sinks)
    kc = kept_coverage(g, E, c)["layers"]
    nint = [int(interior[g.layer_nodes(r["layer"])].sum()) for r in kc]
    tot = sum(nint)

    def f(key):
        return sum(r[key] * ni for r, ni in zip(kc, nint)) / tot

    return {"c": c, "in_cov": f("in_cov"), "out_cov": f("out_cov"), "in_cov_uniform": f("in_cov_uniform"),
            "out_cov_uniform": f("out_cov_uniform"), "layers": kc}


def anatomy(g, E, c, names) -> dict:
    an = backbone_anatomy(g, E, c)
    an["nodes_per_layer"] = {names[l]: v for l, v in enumerate(an["nodes_per_layer"]) if v}
    an["edge_layer_pairs"] = {f"{names[int(k.split('->')[0])]} -> {names[int(k.split('->')[1])]}": v
                              for k, v in an["edge_layer_pairs"].items()}
    return an


# --- one graph ----------------------------------------------------------------------------------


def run(name: str, kind: str, d: Path, n: int, mix_n: int, res_dir: Path, out_dir: Path) -> dict:
    t0 = time.time()
    work = out_dir / name
    g, names, rate = load_graph(kind, d, work)
    interior = ~(g.sources | g.sinks)
    n_int = int(interior.sum())
    floor = 1.0 / n_int
    groups = g.edge_layer("dst")
    E = degree_entry(g, g.weight, groups)
    grid = default_grid(E)
    cs = np.concatenate([grid, list(FIXED.values())])
    print(f"[{name}] {kind}: {g.n_nodes} nodes, {g.n_edges} edges, {g.n_layers} layers, {int(g.sources.sum())} "
          f"sources, {n_int} interior; load {time.time() - t0:.0f}s", flush=True)
    rec = {"name": name, "kind": kind, "input_dir": str(d), "design": "docs/e06_design.md (pre-registered, 4d49a98)",
           "n_nodes": g.n_nodes, "n_edges": g.n_edges, "n_interior": n_int, "layer_names": names,
           "layer_sizes": g.layer_sizes, "fixed_c": FIXED, "n_draws": n, "seeds": SEEDS, "nulls": {}}
    gate = d / "gate.json"
    rec["gate"] = json.loads(gate.read_text()) if gate.exists() else None
    if rec["gate"] is not None:
        rec["gate"] = {k: v for k, v in rec["gate"].items() if k != "trajectory"}
        print(f"[{name}] gate.json: pass = {rec['gate'].get('pass')}", flush=True)
    elif kind == "e05b":
        print(f"[{name}] WARNING: no gate.json in {d}", flush=True)

    arrays = {}
    for key, ng in weight_nulls(g, rate).items():
        part = work / f"null_{key}_{n}.json"
        if part.exists():  # resume: same seed, same draws
            nr = json.loads(part.read_text())
            arrays.update({k: np.array(v) for k, v in nr.pop("_arrays").items()})
            rec["nulls"][key] = nr
            print(f"[{name}] {LABEL[key]}: loaded from {part.name}", flush=True)
            continue
        t = time.time()
        rng = np.random.default_rng(SEEDS[key])
        if isinstance(ng, str):  # N-DC
            dn = decompose(g, g.weight, cs=cs, n_shuffles=n, rng=rng, null="decouple")
        else:  # weight shuffles: same draws as decompose(null="shuffle"), bit-identical, faster
            dn = decompose_fast(g, g.weight, groups=groups, cs=cs, n_shuffles=n, rng=rng, null_groups=ng)
        nr, arr = null_record(dn, len(grid), floor)
        nr["label"] = LABEL[key]
        nr["seconds"] = time.time() - t
        rec["nulls"][key] = nr
        arr = {f"{key}_{k}": v for k, v in arr.items()}
        arrays.update(arr)
        part.write_text(json.dumps({**nr, "_arrays": {k: v.tolist() for k, v in arr.items()}}))
        fx = nr["fixed"]
        print(f"[{name}] {LABEL[key]}: " + " | ".join(
            f"{lb}: db {v['excess']['b']:+.4f} = cov {v['excess']['coverage']:+.4f} + align {v['excess']['cov']:+.4f}, "
            f"z {v['z']:.1f}, p_high {v['p_high']:.4f}" for lb, v in fx.items())
            + f" | grid z_peak {nr['grid']['z_peak']:.2f} (null max {nr['grid']['null_peaks_max']:.2f}, "
              f"p {nr['grid']['p']:.4f}); {nr['seconds']:.0f}s", flush=True)
        del dn

    t = time.time()
    rec["strength"], arrays["strength_perm"] = strength_test(g, rate, n, np.random.default_rng(SEEDS["strength_perm"]))
    s = rec["strength"]
    print(f"[{name}] P5 strength: rho {s['rho']:+.3f} (p {s['p_perm']:.4f}, perm max {s['perm_max']:+.3f}), "
          f"given rate {s['rho_given_rate']:+.3f}; {time.time() - t:.0f}s", flush=True)

    sampler = "parallel" if kind == "e03" else "sequential"
    rec["joint_degree"], arr = joint_degree(g, E, sampler, n, mix_n if sampler == "parallel" else 0)
    arrays.update(arr)

    c_star = rec["nulls"]["within_layer"]["grid"]["c_star"]
    rec["kept_coverage"] = {lb: coverage_summary(g, E, c) for lb, c in FIXED.items()}
    rec["anatomy"] = {"c_star": anatomy(g, E, c_star, names), "c_1": anatomy(g, E, 1.0, names)}
    rec["degree_determined"] = {lb: degree_determined_share(g, E, c) for lb, c in FIXED.items()}
    rec["seconds"] = time.time() - t0
    rec["scores"] = score(rec)
    print(summary_line(name, kind, rec["scores"]), flush=True)

    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"heldout_{name}.json").write_text(json.dumps(rec, indent=1, default=float))
    np.savez_compressed(work / "draws.npz", **arrays)
    print(f"[{name}] wrote {res_dir / f'heldout_{name}.json'} and {work / 'draws.npz'}; total {rec['seconds']:.0f}s",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--e03-dir", type=Path, nargs="+", help="e03 seed directories (VPD graph; P3 reported only)")
    src.add_argument("--e05b-dir", type=Path, nargs="+", help="e05b output directories (all predictions count)")
    src.add_argument("--qwen-dir", type=Path, nargs="+", help="other e04-layout directories (code check)")
    ap.add_argument("--name", nargs="*", default=None, help="graph names (default: directory names)")
    ap.add_argument("--n", type=int, default=N_DRAWS, help="draws per null (the design: 1000)")
    ap.add_argument("--mix-n", type=int, default=200, help="e03: sequential N-JD draws for the mixing check")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "e06")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out" / "e06_heldout")
    args = ap.parse_args()
    kind, dirs = (("e03", args.e03_dir) if args.e03_dir else ("e05b", args.e05b_dir) if args.e05b_dir
                  else ("check", args.qwen_dir))
    names = args.name or [d.name for d in dirs]
    if len(names) != len(dirs):
        raise SystemExit("--name needs one name per directory")
    for d, name in zip(dirs, names):  # sequentially
        if kind == "e03" and not name.startswith("e03"):
            name = f"e03_{name}"
        run(name, kind, d.resolve(), args.n, args.mix_n, args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
