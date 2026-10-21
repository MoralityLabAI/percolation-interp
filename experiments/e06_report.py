"""e06 exploratory report: figures to docs/figures/e06/, compact results to results/e06/.

  python experiments/e06_report.py
Reads out/e06/explore_*.json (from e06_explore.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from percinterp import plotstyle  # noqa: E402

plotstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "out" / "e06"
RES = ROOT / "results" / "e06"
FIG = ROOT / "docs" / "figures" / "e06"
GRAPHS = [("e03_vpd", "e03 VPD (toy, seed 0)"), ("e03_sae", "e03 SAE tower (seed 0)"),
          ("e04_s0", "e04 Qwen L2-3, seed 0"), ("e04_s1", "e04 Qwen L2-3, seed 1")]
NULLS = [("within_layer", "within-layer", plotstyle.INK), ("out_strength", "out-strength", plotstyle.SERIES[0]),
         ("in_strength", "in-strength", plotstyle.SERIES[1]), ("layer_pair", "layer-pair", plotstyle.SERIES[2]),
         ("activity_binned", "activity-binned", plotstyle.MUTED)]


def load():
    out = {}
    for name, _ in GRAPHS:
        p = OUT / f"explore_{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


def fig_decomposition(data, path, null="within_layer"):
    names = [(n, t) for n, t in GRAPHS if n in data]
    fig, axes = plt.subplots(2, len(names), figsize=(3.0 * len(names), 5.4), sharex="col", squeeze=False)
    for j, (name, title) in enumerate(names):
        r = data[name]["nulls"][null]
        cs = np.array(r["cs"])
        tot = np.array(r["learned_curve"])
        cov = np.array(r["excess_coverage_curve"])
        al = np.array(r["excess_cov_curve"])
        for i in range(2):
            ax = axes[i, j]
            ax.plot(cs, tot, color=plotstyle.INK, label="backbone excess, total")
            ax.plot(cs, cov, color=plotstyle.SERIES[0], lw=1.2, label="coverage term, Δ(rF·rB)")
            ax.plot(cs, al, color=plotstyle.SERIES[1], lw=1.2, label="alignment term, ΔCov(A, D)")
            ax.axhline(0, color=plotstyle.BASELINE, lw=0.8)
            ax.axvline(r["c_star"], color=plotstyle.MUTED, lw=0.8, ls="--")
            ax.set_xscale("log")
        small = cs <= max(1.0, 3 * r["c_star"])
        lo = min(tot[small].min(), cov[small].min(), al[small].min())
        hi = max(tot[small].max(), cov[small].max(), al[small].max())
        pad = 0.15 * (hi - lo + 1e-4)
        axes[1, j].set_ylim(lo - pad, hi + pad)
        axes[1, j].set_xlim(cs.min(), max(1.0, 3 * r["c_star"]))
        axes[0, j].set_title(title, fontsize=9)
        axes[1, j].set_xlabel("kept out-degree per live source, c")
    axes[0, 0].set_ylabel("real − mean shuffle\n(interior backbone fraction)")
    axes[1, 0].set_ylabel("same, small c (zoom)")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Where the sieve's excess and deficit come from (within-layer shuffles; dashed: c*). Exploratory.",
                 x=0.01, ha="left", fontsize=10, color=plotstyle.INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_nulls(data, path):
    names = [(n, t) for n, t in GRAPHS if n in data]
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.2), squeeze=False)
    for j, (name, title) in enumerate(names):
        ax = axes[0, j]
        for key, label, color in NULLS:
            r = data[name]["nulls"].get(key)
            if r is None:
                continue
            ax.plot(r["cs"], r["z_curve"], color=color, lw=1.2, label=label)
        ax.axhline(0, color=plotstyle.BASELINE, lw=0.8)
        ax.axhline(3, color=plotstyle.MUTED, lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("c")
    axes[0, 0].set_ylabel("z of backbone fraction vs null")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Learned connectivity z under five weight nulls (100 shuffles each). Exploratory.",
                 x=0.01, ha="left", fontsize=10, color=plotstyle.INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def compact(rec):
    """Drop per-shuffle arrays; keep curves and summaries."""
    out = json.loads(json.dumps(rec))
    for v in out.get("joint_degree", {}).values():
        v.pop("null", None)
    return out


def main():
    data = load()
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    fig_decomposition(data, FIG / "decomposition.png")
    fig_nulls(data, FIG / "nulls.png")
    for name, rec in data.items():
        (RES / f"explore_{name}.json").write_text(json.dumps(compact(rec)))
    if (OUT / "explore_init.json").exists():
        (RES / "explore_init.json").write_text((OUT / "explore_init.json").read_text())
    print("figures:", sorted(p.name for p in FIG.glob("*.png")), "results:", sorted(p.name for p in RES.glob("*")))


if __name__ == "__main__":
    main()
