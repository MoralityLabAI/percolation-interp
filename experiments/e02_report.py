"""Step 2c: score H1-H3 (docs/plan.md, "Step 2 design") from e02_analyze outputs.

Run:  python experiments/e02_report.py [analysis_dir]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from percinterp import plotstyle

plotstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def first_step(history, key, thresh):
    return next((h["step"] for h in history if h[key] >= thresh), None)


def crossing(x, y, level):
    """First x (log-interpolated) where y rises through level; None if never."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for k in range(1, len(y)):
        if y[k - 1] < level <= y[k]:
            t = (level - y[k - 1]) / (y[k] - y[k - 1])
            return float(np.exp(np.log(x[k - 1]) + t * (np.log(x[k]) - np.log(x[k - 1]))))
    return None


def interp_log(x, xs, ys):
    return float(np.interp(np.log(x), np.log(xs), ys))


def score_run(r):
    cfg = r["config"]
    hist = r["full_history"]
    ck = r["checkpoints"]
    init, final = ck[0], ck[-1]
    t_mem = first_step(hist, "train_acc", 0.99)
    t_grok = first_step(hist, "test_acc", 0.95)
    out = {
        "name": r["name"], "depth": cfg["depth"], "width": cfg["width"], "seed": cfg["seed"],
        "final_test_acc": hist[-1]["test_acc"], "grokked": hist[-1]["test_acc"] >= 0.95,
        "t_mem": t_mem, "t_grok": t_grok,
        "c_span": {k: (init[f"{k}_pspan"], final[f"{k}_pspan"]) for k in ("rank", "shuf")},
    }
    # H2
    out["delta"] = math.log(final["rank_pspan"] / final["shuf_pspan"]) - math.log(
        init["rank_pspan"] / init["shuf_pspan"])
    # H1a, H1b from final-checkpoint fidelity
    fid = final["fidelity"]
    cs = r["ps_fid"]
    rec_all, rec_bb = np.array(fid["all"]["recovered"]), np.array(fid["backbone"]["recovered"])
    band = (rec_all >= 0.2) & (rec_all <= 0.9)
    out["h1a_max_gap"] = float(np.abs(rec_all - rec_bb)[band].max()) if band.any() else None
    out["h1a_gap_all_c"] = np.abs(rec_all - rec_bb).tolist()
    c_half = crossing(cs, rec_all, 0.5)
    out["c_half"] = c_half
    out["bb_edge_half"] = interp_log(c_half, cs, fid["bb_edge"]) if c_half else None
    # H3
    steps = np.array([c["step"] for c in ck], float)
    learned = np.array([c["rank_bb_node_ref"] - c["shuf_bb_node_ref"] for c in ck])
    ls = np.log(steps + 1)
    slope = np.abs(np.diff(learned) / np.diff(ls))
    k = int(np.argmax(slope))
    out["h3_steepest_step"] = float(np.exp((ls[k] + ls[k + 1]) / 2) - 1)
    if t_mem is not None and t_grok is not None:
        out["h3_rank_in_window"] = bool(t_mem <= out["h3_steepest_step"] <= 1.5 * t_grok)
        after = steps >= t_mem
        norms = np.array([c.get("weight_norm", np.nan) for c in ck])
        absv = np.array([c["abs_bb_node_ref"] for c in ck])
        ok = after & np.isfinite(norms)
        out["h3_abs_spearman_norm"] = float(spearmanr(absv[ok], norms[ok])[0]) if ok.sum() > 3 else None
    # H2' (depth arm): c at which the mean hidden-layer backbone node fraction first reaches 0.25
    cs_grid = r["ps"]

    def c_bb(rec, key):
        return crossing(cs_grid, rec[key], 0.25)

    cb = {k: (c_bb(init, f"{k}_bb_node"), c_bb(final, f"{k}_bb_node")) for k in ("rank", "shuf")}
    out["c_bb"] = cb
    if all(v is not None for pair in cb.values() for v in pair):
        out["delta_prime"] = math.log(cb["rank"][1] / cb["shuf"][1]) - math.log(cb["rank"][0] / cb["shuf"][0])
    else:
        out["delta_prime"] = None
    # H3-peak (depth arm): peak of learned connectivity in [0.5, 1.0] x t_grok, then >= 30% fall by the end
    if t_grok:
        kp = int(np.argmax(learned))
        out["h3_peak_step_over_tgrok"] = float(steps[kp] / t_grok)
        out["h3_peak_fall"] = float((learned[kp] - learned[-1]) / learned[kp]) if learned[kp] > 0 else None
        out["h3_peak_pass"] = bool(0.5 <= out["h3_peak_step_over_tgrok"] <= 1.0
                                   and out["h3_peak_fall"] is not None and out["h3_peak_fall"] >= 0.3)
    out["series"] = {
        "step": steps.tolist(), "learned": learned.tolist(),
        "rank": [c["rank_bb_node_ref"] for c in ck], "shuf": [c["shuf_bb_node_ref"] for c in ck],
        "abs": [c["abs_bb_node_ref"] for c in ck],
        "test_acc": [c.get("test_acc") for c in ck], "weight_norm": [c.get("weight_norm") for c in ck],
    }
    out["fidelity_final"] = {"c": cs, "rec_all": rec_all.tolist(), "rec_bb": rec_bb.tolist(),
                             "acc_all": fid["all"]["acc"], "bb_edge": fid["bb_edge"]}
    return out


def verdicts(runs):
    g = [r for r in runs if r["grokked"]]
    v = {"n_runs": len(runs), "n_grokked": len(g)}
    gaps = [r["h1a_max_gap"] for r in g if r["h1a_max_gap"] is not None]
    v["H1a"] = {"max_gap_over_runs": max(gaps) if gaps else None,
                "pass": bool(gaps) and max(gaps) < 0.05}
    by_cfg = defaultdict(list)
    for r in g:
        if r["c_half"] is not None:
            by_cfg[(r["depth"], r["width"])].append(r)
    c_half = [np.mean([r["c_half"] for r in rs]) for rs in by_cfg.values()]
    bb_half = [np.mean([r["bb_edge_half"] for r in rs]) for rs in by_cfg.values()]
    if len(c_half) >= 2:
        cv_c = float(np.std(c_half) / np.mean(c_half))
        cv_bb = float(np.std(bb_half) / np.mean(bb_half))
        v["H1b"] = {"n_configs": len(c_half), "cv_c": cv_c, "cv_backbone": cv_bb, "pass": cv_bb < cv_c}
    deltas = defaultdict(list)
    for r in g:
        deltas[(r["depth"], r["width"])].append(r["delta"])
    v["H2"] = {}
    for key, d in sorted(deltas.items()):
        d = np.array(d)
        sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
        v["H2"][f"D{key[0]}_N{key[1]}"] = {
            "deltas": d.tolist(), "mean": float(d.mean()), "sd": sd,
            "pass": bool(d.mean() < 0 and (len(d) < 2 or abs(d.mean()) > sd)),
        }
    # depth arm, stricter H1b: backbone axis must beat both nominal c and kept fraction p (~ c/N)
    p_half = [np.mean([r["c_half"] / r["width"] for r in rs]) for rs in by_cfg.values()]
    if len(c_half) >= 2:
        cv_p = float(np.std(p_half) / np.mean(p_half))
        v["H1b_strict"] = {"n_configs": len(c_half), "cv_c": v["H1b"]["cv_c"], "cv_p": cv_p,
                           "cv_backbone": v["H1b"]["cv_backbone"],
                           "pass": v["H1b"]["cv_backbone"] < min(v["H1b"]["cv_c"], cv_p)}
    dps = defaultdict(list)
    for r in g:
        if r.get("delta_prime") is not None:
            dps[(r["depth"], r["width"])].append(r["delta_prime"])
    v["H2_prime"] = {}
    for key, d in sorted(dps.items()):
        d = np.array(d)
        sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
        v["H2_prime"][f"D{key[0]}_N{key[1]}"] = {
            "deltas": d.tolist(), "mean": float(d.mean()), "sd": sd,
            "pass": bool(d.mean() < 0 and (len(d) < 2 or abs(d.mean()) > sd)),
            "confirmatory": key[0] in (4, 8)}
    peaks = [r.get("h3_peak_pass") for r in g if r.get("h3_peak_pass") is not None]
    v["H3_peak"] = {"n_runs": len(peaks), "n_pass": int(sum(peaks)),
                    "pass": bool(peaks) and (len(peaks) - sum(peaks)) <= len(peaks) / 3,
                    "per_run": {r["name"]: (r.get("h3_peak_step_over_tgrok"), r.get("h3_peak_fall")) for r in g}}
    v["H3_rank"] = {r["name"]: r.get("h3_rank_in_window") for r in g}
    v["H3_abs"] = {r["name"]: r.get("h3_abs_spearman_norm") for r in g}
    return v


def fig_h3(runs, path):
    cfgs = sorted({(r["depth"], r["width"]) for r in runs})
    seed0 = {(r["depth"], r["width"]): r for r in runs if r["seed"] == min(x["seed"] for x in runs)}
    cols = len(cfgs)
    fig, axes = plt.subplots(2, cols, figsize=(3.2 * cols, 5.0), sharex="col", squeeze=False)
    for j, key in enumerate(cfgs):
        r = seed0.get(key)
        if r is None:
            continue
        s = r["series"]
        steps = np.array(s["step"]) + 1
        ax = axes[0, j]
        ax.plot(steps, s["test_acc"], color=plotstyle.INK_2)
        ax.set_title(f"D = {key[0]}, N = {key[1]}", fontsize=9)
        ax.set_ylim(-0.03, 1.03)
        ax2 = axes[1, j]
        for k, (name, label) in enumerate([("rank", "ranked"), ("shuf", "shuffled"), ("abs", "absolute")]):
            ax2.plot(steps, s[name], color=plotstyle.SERIES[k], label=label)
        for ax_ in (ax, ax2):
            for t, ls in ((r["t_mem"], ":"), (r["t_grok"], "--")):
                if t:
                    ax_.axvline(t + 1, color=plotstyle.MUTED, lw=0.8, ls=ls)
            ax_.set_xscale("log")
        ax2.set_xlabel("training step + 1")
    axes[0, 0].set_ylabel("test accuracy")
    axes[1, 0].set_ylabel("backbone node fraction at c = 2")
    axes[1, -1].legend(loc="upper left")
    fig.suptitle("Connectivity over training (seed 0). Dotted: train acc 99%. Dashed: test acc 95%.",
                 x=0.01, ha="left", fontsize=10, color=plotstyle.INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_h1(runs, path):
    g = [r for r in runs if r["grokked"]]
    depths = sorted({r["depth"] for r in g})
    widths = sorted({r["width"] for r in g})
    colors = dict(zip(depths, plotstyle.SERIES))
    styles = dict(zip(widths, ["-", "--", ":"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    seen = set()
    for r in g:
        f = r["fidelity_final"]
        label = None
        if (r["depth"], r["width"]) not in seen:
            seen.add((r["depth"], r["width"]))
            label = f"D = {r['depth']}, N = {r['width']}"
        kw = dict(color=colors[r["depth"]], ls=styles[r["width"]], lw=1.2)
        axes[0].plot(f["c"], f["rec_all"], label=label, **kw)
        axes[1].plot(f["bb_edge"], f["rec_all"], **kw)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("kept out-degree per node, c")
    axes[1].set_xlabel("fraction of edges on the backbone")
    axes[0].set_ylabel("logit metric recovered")
    axes[0].set_title("against nominal occupation")
    axes[1].set_title("against effective sparsity")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle("H1b: do fidelity curves collapse on connectivity? (final checkpoints, all seeds)",
                 x=0.01, ha="left", fontsize=10, color=plotstyle.INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir", type=Path, nargs="?", default=ROOT / "out" / "e02" / "analysis" / "runs")
    args = ap.parse_args()
    runs = [score_run(json.loads(p.read_text())) for p in sorted(args.analysis_dir.glob("*.json"))]
    out = ROOT / "out" / "e02" / "report" / args.analysis_dir.name
    out.mkdir(parents=True, exist_ok=True)
    v = verdicts(runs)
    (out / "summary.json").write_text(json.dumps({"verdicts": v, "runs": runs}, indent=1))
    fig_h3(runs, out / "h3_over_training.png")
    if v["n_grokked"]:
        fig_h1(runs, out / "h1_fidelity.png")
    print(f"{v['n_grokked']}/{v['n_runs']} runs grokked (final test acc >= 0.95)")
    for r in runs:
        print(f"  {r['name']}: test {r['final_test_acc']:.3f} t_mem {r['t_mem']} t_grok {r['t_grok']} "
              f"c* rank {r['c_span']['rank'][0]:.2f}->{r['c_span']['rank'][1]:.2f} "
              f"shuf {r['c_span']['shuf'][0]:.2f}->{r['c_span']['shuf'][1]:.2f} delta {r['delta']:+.3f} "
              f"H1a gap {r['h1a_max_gap']} c_half {r['c_half']} steepest {r['h3_steepest_step']:.0f}")
    print(json.dumps({k: v[k] for k in v if k not in ("H3_rank", "H3_abs")}, indent=1))
    print("H3_rank", v["H3_rank"])
    print("H3_abs", v["H3_abs"])


if __name__ == "__main__":
    main()
