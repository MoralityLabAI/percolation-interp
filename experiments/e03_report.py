"""e03 figures: sieve curves for both graphs, and the cross-method enrichment tests.

  python experiments/e03_report.py
Reads out/e03/analysis.json and out/e03/enrichment_followup.json; writes docs/figures/e03/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from percinterp import plotstyle

plotstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "e03"
FIG = ROOT / "docs" / "figures" / "e03"


def fig_sieve(r, path):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.6), sharex="col")
    for j, (key, title) in enumerate([("vpd_sieve", "VPD component graph"), ("sae_sieve", "SAE feature graph")]):
        s = r[key]
        cs, real, sh, z = map(np.array, (s["cs"], s["real"], s["shuffle_mean"], s["z"]))
        ax = axes[0, j]
        ax.plot(cs, real, color=plotstyle.SERIES[0], label="graph")
        ax.plot(cs, sh, color=plotstyle.SERIES[1], label="within-layer shuffles (mean)")
        ax.axvline(s["c_star"], color=plotstyle.MUTED, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_title(f"{title}: peak z {s['z_peak']:.1f}, p {s['p']:.3f}", fontsize=9)
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
    fig.suptitle("Percolation sieve: connected structure at small c, hub concentration at large c (dashed: c*)",
                 x=0.01, ha="left", fontsize=10, color=plotstyle.INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_enrichment(r, f, path):
    rows = [
        ("P2 co-firing, PMI (pre-registered)", r["enrichment"]["P2"], True),
        ("P2, components rate-matched too", f["P2_rate_matched_components"], False),
        ("P2 secondary: Jaccard", f["P2_jaccard"], False),
        ("Jaccard, both sides rate-matched", f["P2_jaccard_rate_matched_both"], False),
        ("P3 weight |cos| (pre-registered)", r["enrichment"]["P3"], True),
        ("P3, components rate-matched too", f["P3_rate_matched_components"], False),
        ("P3, VPD side fixed", f["P3_components_fixed"], False),
        ("P3, SAE side fixed", f["P3_features_fixed"], False),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = np.arange(len(rows))[::-1]
    for yi, (label, res, prereg) in zip(y, rows):
        color = plotstyle.SERIES[0] if res["p_value"] < 0.01 else plotstyle.MUTED
        ax.plot([1, res["ratio"]], [yi, yi], color=plotstyle.GRID, lw=1.2, zorder=1)
        ax.plot(res["ratio"], yi, "o" if prereg else "D", color=color, markersize=6, zorder=2)
        ax.text(max(res["ratio"], 1) + 0.04, yi, f"p {res['p_value']:.3f}", va="center", fontsize=7.5,
                color=plotstyle.INK_2)
    ax.axvline(1, color=plotstyle.BASELINE, lw=1)
    ax.set_yticks(y, [r_[0] for r_ in rows])
    ax.set_xlabel("sieved sets / median of matched random sets")
    ax.set_xlim(0.6, 2.6)
    ax.grid(axis="y", visible=False)
    ax.set_title("Do the sieved VPD and SAE structures agree? (circles: pre-registered; blue: p < 0.01)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    r = json.loads((OUT / "analysis.json").read_text())
    f = json.loads((OUT / "enrichment_followup.json").read_text())
    fig_sieve(r, FIG / "sieve_curves.png")
    fig_enrichment(r, f, FIG / "enrichment.png")
    print("figures written")


if __name__ == "__main__":
    main()
