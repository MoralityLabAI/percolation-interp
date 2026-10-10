"""Shared matplotlib styling: recessive chrome, thin marks, fixed color roles.

Colors are the pre-validated reference palette steps: categorical slots in
fixed order, and an ordinal blue ramp (no lighter than step 250 on the light
surface) for ordered series such as width.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3: blue, orange, aqua

_ORDINAL = {
    2: ["#5598e7", "#0d366b"],
    3: ["#6da7ec", "#256abf", "#0d366b"],
    4: ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
}


def ordinal(n: int) -> list[str]:
    return _ORDINAL[n]


def apply():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
