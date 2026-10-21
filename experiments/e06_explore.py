"""e06 exploratory diagnostics on e03/e04 graphs (discovery data; nothing here is pre-registered).

For each graph (e03 seed 0 VPD and SAE; e04 seeds 0 and 1), CPU only:
  1. sieve-equivalent curves under five weight nulls (within-layer, out-strength, in-strength,
     layer-pair, activity-binned), each with the coverage/alignment decomposition
     b_l = rF rB + Cov (percinterp.diagnostics.decompose);
  2. in/out strength rank correlation per layer and per module type, hub shares;
  3. kept-degree coverage at c* and at the most negative z, against shuffles and the
     uniform-placement formula;
  4. backbone anatomy at c* (nodes per module, edge spans, path hop lengths);
  5. joint-degree null (double-edge swaps within layer-pair blocks) at c*, 1 and 4.

  python experiments/e06_explore.py --data <main checkout>/out [--only e04_s0] [--n 100]
Graphs are cached in out/e06/graphs/; full results in out/e06/; the summary in results/e06/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import numpy as np  # noqa: E402

from percinterp import LayeredDAG  # noqa: E402
from percinterp.controls import init_mlp, mlp_graph  # noqa: E402
from percinterp.controls import permute_within  # noqa: E402
from percinterp.diagnostics import (backbone_anatomy, binned_groups, decompose, degree_determined_share,  # noqa: E402
                                    joint_degree_null, kept_coverage, pair_groups, strength_correlation)
from percinterp.sieve import degree_entry, sieve  # noqa: E402

OUT = ROOT / "out" / "e06"
RES = ROOT / "results" / "e06"
JD_CS = (1.0, 4.0)  # fixed before any joint-degree null was run, besides each graph's c*


# --- graphs ---------------------------------------------------------------------------------


def build(name: str, data: Path):
    """(g, ids, layer_names, rate) for a named graph. rate: firing rate per node (output = 1)."""
    import torch

    if name.startswith("e03"):
        import e03_analyze as ea
        ea.OUT = data / "e03"
        if name == "e03_vpd":
            X, comp_rate = ea.load_crossref()
            g, _, ids = ea.vpd_graph(comp_rate)
            A = torch.load(ea.attributions_path(), weights_only=True)
            tok = A["embed_token_count"].numpy().astype(float)
            tok_rate = tok / tok.sum()
            rate = np.array([tok_rate[i] if m == "embed" else 1.0 if m == "output" else comp_rate[m][i]
                             for m, i in ids])
        else:
            g, _, ids = ea.sae_graph("abs")
            ev = json.loads((ea.OUT / "sae_tower_eval.json").read_text())
            rate = np.array([ev["levels"][lvl]["firing_rate"][i] for lvl, i in ids])
            ids = [(f"level{lvl}", i) for lvl, i in ids]
    else:
        import e04_analyze as e4
        e4.OUT = data / ("e04" if name == "e04_s0" else "e04_seed1")
        g, _, ids, _ = e4.build_graph()
        stats = torch.load(e4.OUT / "stats.pt", weights_only=False)
        rate = np.array([1.0 if m == "output" else stats["n_fire"][m][i].item() / stats["tokens"] for m, i in ids])
    names = [ids[int(g.offsets[l])][0] for l in range(g.n_layers)]
    return g, ids, names, rate


def load(name: str, data: Path):
    path = OUT / "graphs" / f"{name}.npz"
    if not path.exists():
        g, ids, names, rate = build(name, data)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, src=g.src, dst=g.dst, w=g.weight, sizes=g.layer_sizes, sources=g.sources,
                            sinks=g.sinks, rate=rate, names=np.array(names),
                            ids_m=np.array([str(m) for m, _ in ids]), ids_i=np.array([i for _, i in ids]))
    z = np.load(path)
    g = LayeredDAG(list(z["sizes"]), z["src"], z["dst"], z["w"], sources=z["sources"], sinks=z["sinks"])
    return g, list(z["names"]), z["rate"], z["ids_m"]


# --- sieve statistics from a decomposition (same formulas as sieve()) ------------------------


def sieve_stats(dn, n_interior: int, z_min: float = 3.0) -> dict:
    real = dn.real.total("b")
    sh = np.array([d.total("b") for d in dn.shuffles])
    floor = 1.0 / max(1, n_interior)

    def zf(x, ref):
        return (x - ref.mean(0)) / np.maximum(ref.std(0), floor)

    learned = real - sh.mean(0)
    z = zf(real, sh)
    peaks = np.array([zf(sh[i], np.delete(sh, i, axis=0)).max() for i in range(len(sh))])
    ok = z >= z_min
    k = int(np.argmax(np.where(ok, learned, -np.inf))) if ok.any() else int(np.argmax(learned))
    kmin = int(np.argmin(z))
    return {"z": z, "learned": learned, "real": real, "shuffle_mean": sh.mean(0), "z_peak": float(z.max()),
            "null_peaks_max": float(peaks.max()), "p": float((1 + (peaks >= z.max()).sum()) / (1 + len(peaks))),
            "c_star": float(dn.real.cs[k]), "k_star": k, "z_min": float(z[kmin]), "c_zmin": float(dn.real.cs[kmin]),
            "k_zmin": kmin, "learned_min": float(learned.min()), "c_learned_min": float(dn.real.cs[np.argmin(learned)])}


def excess_r(dn, t: str) -> np.ndarray:
    """Node-weighted mean of rF or rB over layers, real minus mean shuffle."""
    w = dn.real.weights[:, None]
    return (w * getattr(dn.real, t)).sum(0) - (w[None] * dn.stack(t)).sum(1).mean(0)


def terms_at(dn, k: int) -> dict:
    """Excess of b, coverage (rF rB), Cov, rF and rB at grid index k: total and per layer."""
    w = dn.real.weights
    out = {"c": float(dn.real.cs[k])}
    for t in ("b", "coverage", "cov", "rF", "rB"):
        real_l = getattr(dn.real, t)[:, k]
        sh_l = dn.stack(t)[:, :, k].mean(0)
        out[t] = {"real": float((w * real_l).sum()), "null": float((w * sh_l).sum()),
                  "excess": float((w * (real_l - sh_l)).sum()),
                  "per_layer_excess_weighted": (w * (real_l - sh_l)).tolist()}
    return out


# --- per graph ------------------------------------------------------------------------------


MODULE_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "c_fc", "down_proj"]


def module_type(name: str) -> str:
    for t in MODULE_TYPES:
        if name.endswith(t):
            return t
    return name


def analyse(name: str, data: Path, n: int, seeds: dict, only_nulls: list[str] | None = None) -> dict:
    t0 = time.time()
    g, names, rate, ids_m = load(name, data)
    interior = ~(g.sources | g.sinks)
    groups = g.edge_layer("dst")
    E = degree_entry(g, g.weight, groups)
    print(f"[{name}] {g.n_nodes} nodes, {g.n_edges} edges, {g.n_layers} layers, "
          f"{int(g.sources.sum())} sources, {int(interior.sum())} interior; load {time.time() - t0:.0f}s", flush=True)
    rec = {"n_nodes": g.n_nodes, "n_edges": g.n_edges, "layer_names": names, "layer_sizes": g.layer_sizes,
           "n_interior": int(interior.sum())}

    nulls = {"within_layer": None, "out_strength": g.src, "in_strength": g.dst}
    if len(np.unique(pair_groups(g))) > len(np.unique(groups)):
        nulls["layer_pair"] = pair_groups(g)
    nulls["activity_binned"] = binned_groups(g, groups, rate, rate, n_bins=5)
    nulls["in_out_decoupled"] = "decouple"

    cs = None
    rec["nulls"] = {}
    if only_nulls is not None:
        rec = json.loads((OUT / f"explore_{name}.json").read_text())
        cs = np.array(rec["nulls"]["within_layer"]["cs"])
        nulls = {k: v for k, v in nulls.items() if k in only_nulls}
    dns = {}
    for k, ng in nulls.items():
        t = time.time()
        dec = isinstance(ng, str)
        dn = decompose(g, g.weight, cs=cs, n_shuffles=n, rng=np.random.default_rng(seeds.get(k, 100 + len(k))),
                       null_groups=None if dec else ng, null="decouple" if dec else "shuffle")
        cs = dn.real.cs
        st = sieve_stats(dn, int(interior.sum()))
        dns[k] = dn
        pos = int(np.argmax(st["learned"]))
        rec["nulls"][k] = {
            **{kk: v for kk, v in st.items() if not isinstance(v, np.ndarray)},
            "cs": cs.tolist(), "z_curve": st["z"].tolist(), "learned_curve": st["learned"].tolist(),
            "real_curve": st["real"].tolist(), "shuffle_mean_curve": st["shuffle_mean"].tolist(),
            "excess_coverage_curve": dn.excess("coverage").tolist(), "excess_cov_curve": dn.excess("cov").tolist(),
            "excess_rF_curve": excess_r(dn, "rF").tolist(), "excess_rB_curve": excess_r(dn, "rB").tolist(),
            "at_c_star": terms_at(dn, st["k_star"]), "at_z_min": terms_at(dn, st["k_zmin"]),
            "at_max_learned": terms_at(dn, pos),
            "layers_used": dn.real.layers.tolist(),
        }
        print(f"[{name}] {k}: z_peak {st['z_peak']:.2f} (null max {st['null_peaks_max']:.2f}, p {st['p']:.3f}), "
              f"c* {st['c_star']:.3g}, z_min {st['z_min']:.1f} at c {st['c_zmin']:.3g}, "
              f"learned min {st['learned_min']:+.3f}; {time.time() - t:.0f}s", flush=True)
    if only_nulls is not None:
        return rec
    wl = rec["nulls"]["within_layer"]
    c_star, c_neg = wl["c_star"], wl["c_zmin"]

    # strength correlation: per layer, per module type, and references
    sc = strength_correlation(g)
    mt = np.array([module_type(names[l]) for l in g.node_layer])
    _, mt_id = np.unique(mt, return_inverse=True)
    sc_type = strength_correlation(g, node_groups=mt_id)
    rng = np.random.default_rng(7)
    w_sh = np.zeros_like(g.weight)
    live = g.weight > 0
    w_sh[live] = permute_within(g.weight[live], groups[live], rng)
    sc_sh = strength_correlation(g, w_sh)
    rec["strength"] = {
        "per_layer": {names[k]: v for k, v in sc["per_group"].items()},
        "pooled_within_layer_rho": sc["pooled_within_group_rho"],
        "per_module_type": {np.unique(mt)[k]: v for k, v in sc_type["per_group"].items()},
        "within_layer_shuffle_pooled_rho": sc_sh["pooled_within_group_rho"],
        "within_layer_shuffle_per_layer": {names[k]: v for k, v in sc_sh["per_group"].items()},
    }
    print(f"[{name}] strength rho pooled {sc['pooled_within_group_rho']:+.3f} "
          f"(one within-layer shuffle {sc_sh['pooled_within_group_rho']:+.3f})", flush=True)

    # kept-degree coverage at c* and at the most negative z, real vs 20 within-layer shuffles
    def cov_summary(entry, c):
        kc = kept_coverage(g, entry, c)["layers"]
        nint = [int(interior[g.layer_nodes(r["layer"])].sum()) for r in kc]
        tot = sum(nint)
        f = lambda key: sum(r[key] * ni for r, ni in zip(kc, nint)) / tot  # noqa: E731
        return {"in_cov": f("in_cov"), "out_cov": f("out_cov"), "in_cov_uniform": f("in_cov_uniform"),
                "out_cov_uniform": f("out_cov_uniform"), "layers": kc}

    rec["kept_coverage"] = {}
    for label, c in (("c_star", c_star), ("z_min", c_neg)):
        real = cov_summary(E, c)
        shs = []
        rng = np.random.default_rng(8)
        for _ in range(20):
            w = np.zeros_like(g.weight)
            w[live] = permute_within(g.weight[live], groups[live], rng)
            shs.append(cov_summary(degree_entry(g, w, groups), c))
        rec["kept_coverage"][label] = {
            "c": c, "real": {k: v for k, v in real.items() if k != "layers"}, "real_layers": real["layers"],
            "shuffle": {k: float(np.mean([s[k] for s in shs])) for k in ("in_cov", "out_cov")}}
        r = rec["kept_coverage"][label]
        print(f"[{name}] kept coverage at {label} c={c:.3g}: in {r['real']['in_cov']:.3f} vs shuffle "
              f"{r['shuffle']['in_cov']:.3f} (uniform {r['real']['in_cov_uniform']:.3f}); out "
              f"{r['real']['out_cov']:.3f} vs {r['shuffle']['out_cov']:.3f} "
              f"(uniform {r['real']['out_cov_uniform']:.3f})", flush=True)

    # backbone anatomy at c* (within-layer sieve) and at c = 1
    rec["anatomy"] = {}
    for label, c in (("c_star", c_star), ("c1", 1.0)):
        an = backbone_anatomy(g, E, c)
        an["nodes_per_layer"] = {names[l]: v for l, v in enumerate(an["nodes_per_layer"]) if v}
        an["edge_layer_pairs"] = {f"{names[int(k.split('->')[0])]} -> {names[int(k.split('->')[1])]}": v
                                  for k, v in an["edge_layer_pairs"].items()}
        rec["anatomy"][label] = an
    print(f"[{name}] backbone at c*: {rec['anatomy']['c_star']['n_nodes']} nodes, hops "
          f"{rec['anatomy']['c_star']['min_hops']}-{rec['anatomy']['c_star']['max_hops']}, "
          f"per layer {rec['anatomy']['c_star']['nodes_per_layer']}", flush=True)

    rec["degree_determined"] = [degree_determined_share(g, E, c) for c in (c_star,) + JD_CS]
    print(f"[{name}] degree-determined share: " + ", ".join(f"c={d['c']:.3g}: {d['share_equal']:.4f} "
          f"(backbone {d['backbone']:.4f}, local {d['local']:.4f})" for d in rec["degree_determined"]), flush=True)

    # joint-degree null
    rec["joint_degree"] = {}
    runs = [(c, "pair", True) for c in (c_star,) + JD_CS] + [(c_star, "pair", False)]
    runs += [(c, blk, True) for blk in ("dst", "none") for c in (c_star,) + JD_CS]
    for c, blk, skel in runs:
        t = time.time()
        r = joint_degree_null(g, E, c, n_null=n, rng=np.random.default_rng(int(1000 * c) % 2**31), skeleton=skel,
                              blocks=blk)
        key = f"{blk} c={c:.4g}{'' if skel else ' (no skeleton)'}"
        rec["joint_degree"][key] = {"c": c, "blocks": blk, "skeleton": skel, "real": r.real, "null_mean": float(r.null.mean()),
                                    "null_sd": float(r.null.std()), "null_min": float(r.null.min()),
                                    "null_max": float(r.null.max()), "z": r.z, "p_high": r.p_high, "p_low": r.p_low,
                                    "n_edges": r.n_edges, "acceptance": r.acceptance, "null": r.null.tolist()}
        print(f"[{name}] joint-degree {key}: real {r.real:.4f}, null {r.null.mean():.4f} +- {r.null.std():.4f} "
              f"[{r.null.min():.4f}, {r.null.max():.4f}], z {r.z:+.1f}, p_high {r.p_high:.3f}, p_low {r.p_low:.3f}, "
              f"{r.n_edges} edges, acceptance {r.acceptance:.2f}; {time.time() - t:.0f}s", flush=True)
    rec["seconds"] = time.time() - t0
    return rec


def init_reference(n_samples: int = 16) -> dict:
    """Hypothesis 3 without training: e01's Kaiming ReLU MLP (9 layers of 128), EAP edges."""
    rhos, top10o, top10i = [], [], []
    rhos_w = []
    for s in range(n_samples):
        rng = np.random.default_rng(s)
        Ws = init_mlp([128] * 9, rng)
        g = mlp_graph(Ws, rng.normal(size=128), kind="eap", readout=rng.normal(size=128))
        sc = strength_correlation(g)
        rhos.append(sc["pooled_within_group_rho"])
        top10o.append(np.mean([v["top10_out"] for v in sc["per_group"].values()]))
        top10i.append(np.mean([v["top10_in"] for v in sc["per_group"].values()]))
        gw = mlp_graph(Ws, kind="weight")
        rhos_w.append(strength_correlation(gw)["pooled_within_group_rho"])
    return {"eap_pooled_rho_mean": float(np.mean(rhos)), "eap_pooled_rho_sd": float(np.std(rhos)),
            "eap_top10_out": float(np.mean(top10o)), "eap_top10_in": float(np.mean(top10i)),
            "weight_pooled_rho_mean": float(np.mean(rhos_w)), "n_samples": n_samples}


# Seeds of the original analyses, so the within-layer (and e04 in-strength) curves reproduce them.
SEEDS = {"e03_vpd": {"within_layer": 1}, "e03_sae": {"within_layer": 0, "in_strength": 4},
         "e04_s0": {"within_layer": 12, "in_strength": 11}, "e04_s1": {"within_layer": 12, "in_strength": 11}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "out", help="folder holding e03/, e04/, e04_seed1/")
    ap.add_argument("--only", nargs="*", default=["e04_s0", "e04_s1", "e03_vpd", "e03_sae"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--merge-nulls", nargs="*", default=None,
                    help="only run these weight nulls and merge them into an existing explore_<name>.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in [x for x in args.only if x != "init"]:
        try:
            if args.build_only:
                g, names, _, _ = load(name, args.data)
                print(name, g.n_nodes, g.n_edges, names, flush=True)
                continue
            rec = analyse(name, args.data, args.n, SEEDS.get(name, {}), args.merge_nulls)
        except FileNotFoundError as e:
            print(f"[{name}] missing input, skipped: {e}", flush=True)
            continue
        (OUT / f"explore_{name}.json").write_text(json.dumps(rec, default=float))
    if not args.build_only and args.merge_nulls is None:
        ref = init_reference()
        print(f"init MLP EAP reference: {ref}", flush=True)
        (OUT / "explore_init.json").write_text(json.dumps(ref))


if __name__ == "__main__":
    main()
