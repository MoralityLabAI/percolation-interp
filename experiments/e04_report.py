"""e04 report: gate, Q1-Q3 verdicts, results copied to results/e04/, figures to docs/figures/e04/.

  python experiments/e04_report.py      (this repo's venv)
Reads out/e04/{gate,analysis,causal}.json; makes whatever figures their presence allows.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from percinterp import plotstyle

plotstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("E04_OUT", ROOT / "out" / "e04"))
RES = Path(os.environ.get("E04_RESULTS", ROOT / "results" / "e04"))
FIG = Path(os.environ.get("E04_FIG", ROOT / "docs" / "figures" / "e04"))
PD_OUT = Path.home() / "param_decomp_out"
GATE_KEY = "ce_kl/ce_unrecovered_ci_masked"


def fig_gate(g, path):
    steps, vals = np.array(g["trajectory"]).T
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(steps, vals, color=plotstyle.SERIES[0], marker="o", markersize=3)
    ax.axhline(g["threshold"], color=plotstyle.MUTED, lw=0.8, ls="--", label=f"gate ({g['threshold']:.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("CE gap unrecovered, CI-masked")
    ax.legend(loc="upper right")
    ax.set_title(f"Decomposition quality: {g[GATE_KEY]:.3f} at step {g['step']} "
                 f"({'passes' if g['pass'] else 'fails'} the gate)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_sieve(r, path):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.6), sharex=True)
    for j, (key, title) in enumerate([("within_layer", "within-layer shuffles"),
                                      ("in_strength", "in-strength-preserving shuffles")]):
        s = r[key]
        cs, real, sh, z = map(np.array, (s["cs"], s["real"], s["shuffle_mean"], s["z"]))
        ax = axes[0, j]
        ax.plot(cs, real, color=plotstyle.SERIES[0], label="component graph")
        ax.plot(cs, sh, color=plotstyle.SERIES[1], label="shuffles (mean)")
        ax.axvline(s["c_star"], color=plotstyle.MUTED, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_title(f"Against {title}: peak z {s['z_peak']:.1f}, p {s['p']:.3f}", fontsize=9)
        ax.set_ylabel("backbone fraction (interior nodes)" if j == 0 else "")
        axz = axes[1, j]
        axz.plot(cs, z, color=plotstyle.INK_2)
        axz.axhline(0, color=plotstyle.BASELINE, lw=0.8)
        axz.axhline(3, color=plotstyle.MUTED, lw=0.8, ls=":")
        axz.set_yscale("symlog", linthresh=10)
        axz.axvline(s["c_star"], color=plotstyle.MUTED, lw=0.8, ls="--")
        axz.set_xlabel("kept out-degree per live source, c")
        axz.set_ylabel("learned connectivity, z vs shuffles" if j == 0 else "")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Qwen3-0.6B layers 2-3, VPD component graph: percolation sieve (dashed: c*)",
                 x=0.01, ha="left", fontsize=10, color=plotstyle.INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_causal(c, path):
    """Share of the KL gap closed, (KL_none - KL_set) / (KL_none - KL_all): 0 = no components, 1 = all."""
    gap = c["none"] - c["all"]
    share = lambda kl: (c["none"] - np.asarray(kl)) / gap  # noqa: E731
    rand = share(c["random"])
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    ax.scatter(rand, rng.uniform(-0.25, 0.25, len(rand)), s=14, color=plotstyle.MUTED, alpha=0.7, lw=0,
               label=f"{len(rand)} random sets, matched per module in size and firing-rate decile")
    for v, name, color in [(share(c["backbone"]), "sieved backbone", plotstyle.SERIES[0]),
                           (share(c["top_ci"]), "same-size top mean-CI set", plotstyle.SERIES[1])]:
        ax.plot([v], [0], "D", color=color, markersize=9, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(f"{name}\n{v:.1%}", (v, 0), xytext=(0, 18), textcoords="offset points", ha="center",
                    fontsize=8, color=plotstyle.INK)
    ax.axvline(0, color=plotstyle.BASELINE, lw=0.8)
    ax.set_ylim(-0.6, 1.0)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("share of the KL gap closed (no components = 0%; all components = 100%, off scale)")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_title(f"Keeping only the {sum(c['backbone_size'].values())} sieved components beats every matched "
                 f"random set (p {c['p_value']:.2f})", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    g = json.loads((OUT / "gate.json").read_text())
    shutil.copyfile(OUT / "gate.json", RES / "gate.json")
    run_dir = PD_OUT / "decompositions" / g["run_id"]
    if (run_dir / "final_config.yaml").exists():
        shutil.copyfile(run_dir / "final_config.yaml", RES / "vpd_final_config.yaml")
    fig_gate(g, FIG / "gate.png")
    print(f"gate: {GATE_KEY} {g[GATE_KEY]:.4f} at step {g['step']}: {'pass' if g['pass'] else 'FAIL'}")
    if not g["pass"]:
        print("Q1-Q3 not scored (gate failed)")
        return
    verdicts = {}
    if (OUT / "analysis.json").exists():
        r = json.loads((OUT / "analysis.json").read_text())
        shutil.copyfile(OUT / "analysis.json", RES / "analysis.json")
        fig_sieve(r, FIG / "sieve_curves.png")
        verdicts["Q1"] = r["Q1_pass"]
        for k in ("within_layer", "in_strength"):
            s = r[k]
            print(f"Q1 {k}: z_peak {s['z_peak']:.1f} vs shuffle max {max(s['null_peaks']):.1f}, p {s['p']:.3f}, "
                  f"c* {s['c_star']:.3g}, backbone {len(s['backbone'])} components")
    if (OUT / "causal.json").exists():
        c = json.loads((OUT / "causal.json").read_text())
        shutil.copyfile(OUT / "causal.json", RES / "causal.json")
        fig_causal(c, FIG / "causal.png")
        verdicts["Q2"], verdicts["Q3"] = c["Q2_pass"], c["Q3_pass"]
        print(f"Q2: backbone KL {c['backbone']:.4f} vs random min {c['random_min']:.4f} "
              f"(median {c['random_median']:.4f}), p {c['p_value']:.3f}")
        print(f"Q3: backbone KL {c['backbone']:.4f} vs top-CI {c['top_ci']:.4f}; "
              f"all {c['all']:.4f}, none {c['none']:.4f}")
    print("verdicts:", {k: "pass" if v else "fail" for k, v in verdicts.items()})


if __name__ == "__main__":
    main()
