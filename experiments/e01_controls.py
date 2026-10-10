"""Step 1: the control pipeline.

Part A checks the measurement code against exact answers for random layered
graphs (Bernoulli bond percolation, where theory.reach_chain is exact).

Part B measures randomly initialised ReLU MLPs against their own
within-layer-shuffled skeletons. Shuffling turns any ranking into plain
percolation on that skeleton, so any gap between the two is structure that
exists before training. Step 2 must subtract it before crediting training.

Run:  python experiments/e01_controls.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from percinterp import components, profile, rank_entry
from percinterp import plotstyle
from percinterp.controls import complete_layered, init_mlp, mlp_edge_matrices, permute_within
from percinterp.graph import LayeredDAG
from percinterp.measure import forward_entry
from percinterp.theory import giant_fraction, mean_field_reach, reach_chain

plotstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out" / "e01"


def exact_median_c(N: int, depth: int) -> float:
    """c = p*N at which the exact spanning probability of the complete graph is 1/2."""
    widths = [N] * (depth + 1)
    p = brentq(lambda p: reach_chain(widths, p)["span_prob"] - 0.5, 1e-7, 1.0, xtol=1e-9)
    return p * N


# --- Part A ------------------------------------------------------------------


def part_a_spanning(widths, depths, n_samples, rng):
    cs = np.geomspace(0.05, 4.0, 36)
    res = {}
    for L in depths:
        for N in widths:
            layers = [N] * (L + 1)
            g = complete_layered(layers, rng)
            p_span = np.array(
                [forward_entry(g, rng.random(g.n_edges))[g.sinks].min() for _ in range(n_samples)]
            )
            exact = np.array([reach_chain(layers, c / N)["span_prob"] for c in cs])
            mc = (p_span[None, :] <= cs[:, None] / N).mean(axis=1)
            se = np.sqrt(np.maximum(exact * (1 - exact), 1.0 / n_samples) / n_samples)
            res[f"L{L}_N{N}"] = {
                "depth": L,
                "width": N,
                "c": cs.tolist(),
                "exact": exact.tolist(),
                "mc": mc.tolist(),
                "max_abs_dev": float(np.abs(mc - exact).max()),
                "max_z": float((np.abs(mc - exact) / se).max()),
                "median_c_mc": float(np.median(p_span) * N),
                "median_c_exact": exact_median_c(N, L),
                "path_count_c": float(N ** (-1.0 / L)),
            }
    return res


def part_a_reach(widths, depth, n_samples_small, rng):
    cs = np.linspace(0.1, 4.0, 40)
    res = {"depth": depth, "c": cs.tolist(), "exact": {}, "mean_field": [], "giant": []}
    for N in widths:
        res["exact"][str(N)] = [
            float(reach_chain([N] * (depth + 1), c / N)["mean_reach_frac"][-1]) for c in cs
        ]
    res["mean_field"] = [float(mean_field_reach(c, depth)[-1]) for c in cs]
    res["giant"] = giant_fraction(cs).tolist()
    N = widths[0]
    g = complete_layered([N] * (depth + 1), rng)
    sims = np.array(
        [
            profile(g, rng.random(g.n_edges)).reach_fraction(cs / N, depth)
            for _ in range(n_samples_small)
        ]
    )
    res["mc_width"] = N
    res["mc"] = sims.mean(axis=0).tolist()
    return res


# --- Part B ------------------------------------------------------------------


def span_pair(g: LayeredDAG, rng, group_by="dst"):
    """Profiles for the graph (ranked per layer) and its within-layer shuffle."""
    groups = g.edge_layer(group_by)
    real = profile(g, rank_entry(g.weight, groups))
    shuf = profile(g, rank_entry(permute_within(g.weight, groups, rng), groups))
    return real, shuf


def part_b_mlp(widths, depth, kinds, n_samples, rng):
    """Each sample is a fresh network and a fresh Gaussian input."""
    res = {}
    for kind in kinds:
        for N in widths:
            layers = [N] * (depth + 1)
            rows = []
            for _ in range(n_samples):
                Ws = init_mlp(layers, rng)
                x = rng.normal(size=N)
                readout = rng.normal(size=N)
                g = LayeredDAG.from_dense(mlp_edge_matrices(Ws, x, kind, readout))
                real, shuf = span_pair(g, rng)
                rows.append(
                    (
                        real.p_span,
                        shuf.p_span,
                        real.mean_kept_out_degree(real.p_span),
                        shuf.mean_kept_out_degree(shuf.p_span),
                        g.n_edges / (N * N * depth),
                    )
                )
            a = np.array(rows)
            res[f"{kind}_N{N}"] = {
                "kind": kind,
                "width": N,
                "p_span_real": a[:, 0].tolist(),
                "p_span_shuf": a[:, 1].tolist(),
                "c_span_real": a[:, 2].tolist(),
                "c_span_shuf": a[:, 3].tolist(),
                "edge_density": float(a[:, 4].mean()),
                "median_ratio_real_over_shuf": float(np.median(a[:, 0] / a[:, 1])),
                "frac_real_below_shuf": float((a[:, 0] < a[:, 1]).mean()),
            }
    return res


def part_b_averaged(N, depth, kind, n_inputs, rng):
    """Per-input graphs versus the input-averaged graph of the same network.

    Compared in kept edges per node at spanning, not p: the averaged graph has
    no dead-unit zeros, so its skeleton is denser than any single input's.
    """
    layers = [N] * (depth + 1)
    Ws = init_mlp(layers, rng)
    readout = rng.normal(size=N)
    X = rng.normal(size=(n_inputs, N))
    mats = [mlp_edge_matrices(Ws, x, kind, readout) for x in X]

    def c_span(m):
        g = LayeredDAG.from_dense(m)
        prof = profile(g, rank_entry(g.weight, g.edge_layer()))
        return prof.mean_kept_out_degree(prof.p_span)

    per_input = [c_span(m) for m in mats]
    c_avg = c_span([np.mean([m[l] for m in mats], axis=0) for l in range(depth)])
    return {
        "kind": kind,
        "width": N,
        "c_span_per_input": per_input,
        "c_span_averaged": c_avg,
        "averaged_percentile_among_inputs": float((np.array(per_input) < c_avg).mean()),
    }


def part_b_scale(N, d_in, depth, n_nets, rng):
    """Global versus per-layer ranking when one layer has a different weight scale."""
    layers = [d_in] + [N] * depth
    rows = []
    for _ in range(n_nets):
        Ws = init_mlp(layers, rng)
        g = LayeredDAG.from_dense(mlp_edge_matrices(Ws, kind="weight"))
        per_layer = profile(g, rank_entry(g.weight, g.edge_layer())).p_span
        glob = profile(g, rank_entry(g.weight)).p_span
        rows.append((per_layer, glob))
    a = np.array(rows)
    return {
        "widths": layers,
        "p_span_per_layer": a[:, 0].tolist(),
        "p_span_global": a[:, 1].tolist(),
        "median_ratio_global_over_per_layer": float(np.median(a[:, 1] / a[:, 0])),
    }


def part_b_panel(N, depth, kind, n_samples, ps, rng):
    """All observables against p for one setting, graph versus shuffle."""
    layers = [N] * (depth + 1)
    keys = ["spanning", "backbone", "largest", "susceptibility"]
    acc = {"real": {k: [] for k in keys}, "shuf": {k: [] for k in keys}}
    for _ in range(n_samples):
        Ws = init_mlp(layers, rng)
        g = LayeredDAG.from_dense(mlp_edge_matrices(Ws, rng.normal(size=N), kind, rng.normal(size=N)))
        for name, prof in zip(["real", "shuf"], span_pair(g, rng)):
            comp = components(g, prof.entry, ps)
            acc[name]["spanning"].append(prof.spanning(ps))
            acc[name]["backbone"].append(prof.backbone_node_fraction(ps))
            acc[name]["largest"].append(comp["largest"])
            acc[name]["susceptibility"].append(comp["susceptibility"])
    return {
        "kind": kind,
        "width": N,
        "depth": depth,
        "p": ps.tolist(),
        **{
            name: {k: np.mean(v, axis=0).tolist() for k, v in d.items()} for name, d in acc.items()
        },
    }


# --- Figures -------------------------------------------------------------------


def fig_spanning(res, widths, depths, path):
    colors = dict(zip(widths, plotstyle.ordinal(len(widths))))
    fig, axes = plt.subplots(1, len(depths), figsize=(4.2 * len(depths), 3.2), sharey=True)
    for ax, L in zip(np.atleast_1d(axes), depths):
        for N in widths:
            r = res[f"L{L}_N{N}"]
            ax.plot(r["c"], r["exact"], color=colors[N], label=f"N = {N}")
            ax.plot(r["c"][::2], r["mc"][::2], "o", color=colors[N], markersize=3.5,
                    markeredgecolor=plotstyle.SURFACE, markeredgewidth=0.6)
        ax.axvline(1.0, color=plotstyle.MUTED, lw=0.8, ls="--")
        ax.text(1.04, 0.03, "c = 1", color=plotstyle.INK_2, fontsize=8)
        ax.set_xscale("log")
        ax.set_title(f"Depth {L}")
        ax.set_xlabel("kept edges per node, c = pN")
    np.atleast_1d(axes)[0].set_ylabel("spanning probability")
    np.atleast_1d(axes)[-1].legend(loc="upper left", title="width")
    fig.suptitle(
        "Complete layered graphs: simulation (dots) on the exact chain (lines)",
        x=0.01, ha="left", fontsize=10, color=plotstyle.INK,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_reach(res, path):
    widths = list(res["exact"].keys())
    colors = dict(zip(widths, plotstyle.ordinal(len(widths))))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for N in widths:
        ax.plot(res["c"], res["exact"][N], color=colors[N], label=f"exact, N = {N}")
    ax.plot(res["c"], res["mean_field"], color=plotstyle.INK, lw=1.0, ls="--",
            label=f"mean field, depth {res['depth']}")
    ax.plot(res["c"], res["giant"], color=plotstyle.MUTED, lw=1.0, ls=":",
            label="infinite depth (giant component)")
    ax.plot(res["c"][::2], res["mc"][::2], "o", color=colors[str(res["mc_width"])], markersize=3.5,
            markeredgecolor=plotstyle.SURFACE, markeredgewidth=0.6,
            label=f"simulation, N = {res['mc_width']}")
    ax.set_xlabel("kept edges per node, c = pN")
    ax.set_ylabel(f"fraction of layer {res['depth']} reachable")
    ax.set_title("Reachable fraction converges to the mean-field transition at c = 1")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_mlp(res, widths, kinds, depth, path):
    titles = {"weight": "|W| (wiring)", "direct": "|W a| (direct effect)", "eap": "|a W g| (EAP)"}
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.6 * len(kinds), 3.2), sharey=True)
    x = np.arange(len(widths))
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        for k, (key, label) in enumerate([("c_span_real", "graph"), ("c_span_shuf", "shuffled skeleton")]):
            vals = [np.array(res[f"{kind}_N{N}"][key]) for N in widths]
            med = np.array([np.median(v) for v in vals])
            lo = med - np.array([np.percentile(v, 25) for v in vals])
            hi = np.array([np.percentile(v, 75) for v in vals]) - med
            ax.errorbar(x + (k - 0.5) * 0.18, med, yerr=[lo, hi], fmt="o", color=plotstyle.SERIES[k],
                        ecolor=plotstyle.SERIES[k], elinewidth=1.2, capsize=0, label=label)
        theory = [exact_median_c(N, depth) for N in widths]
        ax.plot(x, theory, color=plotstyle.MUTED, lw=1.0, ls="--", label="complete graph, exact")
        ax.set_xticks(x, [str(N) for N in widths])
        ax.set_xlabel("width N")
        ax.set_title(titles[kind])
    np.atleast_1d(axes)[0].set_ylabel("kept edges per node at spanning (median, IQR)")
    np.atleast_1d(axes)[-1].legend(loc="upper right")
    fig.suptitle(f"ReLU MLPs at initialisation, depth {depth}, ranked per layer",
                 x=0.01, ha="left", fontsize=10, color=plotstyle.INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_panel(res, path):
    labels = {
        "spanning": "spanning probability",
        "backbone": "fraction of nodes on a surviving path",
        "largest": "largest cluster (fraction of nodes)",
        "susceptibility": "mean finite-cluster size",
    }
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    for ax, key in zip(axes, labels):
        for k, (name, label) in enumerate([("real", "graph"), ("shuf", "shuffled skeleton")]):
            ax.plot(res["p"], res[name][key], color=plotstyle.SERIES[k], label=label)
        ax.set_xscale("log")
        ax.set_xlabel("fraction of edges kept per layer, p")
        ax.set_title(labels[key], fontsize=9)
    axes[0].legend(loc="upper left")
    fig.suptitle(
        f"All observables, {res['kind']} graph, N = {res['width']}, depth {res['depth']}, at init",
        x=0.01, ha="left", fontsize=10, color=plotstyle.INK,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --- Main ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sizes, for a smoke test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.quick:
        a_widths, a_depths, a_samples = [16, 32], [4, 16], 100
        reach_widths, b_widths, b_samples = [32, 128], [32, 64], 12
    else:
        a_widths, a_depths, a_samples = [16, 32, 64, 128], [4, 16], 400
        reach_widths, b_widths, b_samples = [64, 256, 1024], [64, 128, 256], 48
    depth_b = 8
    kinds = ["weight", "direct", "eap"]
    t0 = time.time()
    results = {"seed": args.seed, "quick": args.quick}

    print("A1 spanning vs exact chain ...", flush=True)
    results["a_spanning"] = part_a_spanning(a_widths, a_depths, a_samples, rng)
    fig_spanning(results["a_spanning"], a_widths, a_depths, OUT / "a1_spanning.png")

    print("A2 reachable fraction vs mean field ...", flush=True)
    results["a_reach"] = part_a_reach(reach_widths, 32, 60, rng)
    fig_reach(results["a_reach"], OUT / "a2_reach.png")

    print("B1 MLPs at init vs shuffled skeleton ...", flush=True)
    results["b_mlp"] = part_b_mlp(b_widths, depth_b, kinds, b_samples, rng)
    fig_mlp(results["b_mlp"], b_widths, kinds, depth_b, OUT / "b1_mlp_init.png")

    print("B2 per-input vs averaged graph ...", flush=True)
    results["b_averaged"] = {
        kind: part_b_averaged(b_widths[1], depth_b, kind, b_samples, rng) for kind in ["direct", "eap"]
    }

    print("B3 global vs per-layer ranking ...", flush=True)
    results["b_scale"] = part_b_scale(b_widths[1], b_widths[1] // 4, depth_b, b_samples, rng)

    print("B4 observable panel ...", flush=True)
    ps = np.geomspace(1e-3, 0.3, 30)
    results["b_panel"] = part_b_panel(b_widths[1], depth_b, "eap", 16 if not args.quick else 4, ps, rng)
    fig_panel(results["b_panel"], OUT / "b4_panel.png")

    results["seconds"] = time.time() - t0
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print_summary(results)


def print_summary(r):
    print(f"\n=== e01 summary ({r['seconds']:.0f}s) ===")
    print("A1  simulation vs exact chain (complete layered graphs, Bernoulli edges)")
    for key, v in r["a_spanning"].items():
        print(
            f"  depth {v['depth']:>2} N {v['width']:>4}: max |dev| {v['max_abs_dev']:.3f} "
            f"(max z {v['max_z']:.1f}); median c at spanning: sim {v['median_c_mc']:.3f}, "
            f"exact {v['median_c_exact']:.3f}, path count N^(-1/L) {v['path_count_c']:.3f}"
        )
    a2 = r["a_reach"]
    print(f"A2  reachable fraction at depth {a2['depth']}: max |exact - mean field| by width")
    for N, ex in a2["exact"].items():
        print(f"  N {N:>5}: {np.max(np.abs(np.array(ex) - np.array(a2['mean_field']))):.4f}")
    print("B1  MLP at init: median p_span(graph) / p_span(shuffled skeleton); share of samples below 1")
    for key, v in r["b_mlp"].items():
        print(
            f"  {v['kind']:>6} N {v['width']:>4}: ratio {v['median_ratio_real_over_shuf']:.3f}, "
            f"below {v['frac_real_below_shuf']:.2f}, edge density {v['edge_density']:.2f}"
        )
    print("B2  kept edges per node at spanning: input-averaged graph vs per-input graphs")
    for kind, v in r["b_averaged"].items():
        print(
            f"  {kind:>6}: averaged {v['c_span_averaged']:.3f}, per-input median "
            f"{np.median(v['c_span_per_input']):.3f}, percentile {v['averaged_percentile_among_inputs']:.2f}"
        )
    s = r["b_scale"]
    print(f"B3  widths {s['widths'][:3]}...: median p_span global / per-layer = "
          f"{s['median_ratio_global_over_per_layer']:.2f}")


if __name__ == "__main__":
    main()
