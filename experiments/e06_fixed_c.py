"""e06 exploratory: decomposition at the fixed c values of the e06 draft (0.1, 1, 4) on the
discovery graphs, against within-layer shuffles and in/out decoupling (100 draws each).

  python experiments/e06_fixed_c.py      (after e06_explore.py has cached the graphs)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import numpy as np  # noqa: E402

from e06_explore import load  # noqa: E402
from percinterp.diagnostics import decompose  # noqa: E402

CS = np.array([0.1, 1.0, 4.0])

for name in ["e04_s0", "e04_s1", "e03_sae", "e03_vpd"]:
    g, *_ = load(name, None)
    floor = 1 / (~(g.sources | g.sinks)).sum()
    for null in ["shuffle", "decouple"]:
        dn = decompose(g, g.weight, cs=CS, n_shuffles=100, rng=np.random.default_rng(5), null=null)
        sh = np.array([d.total("b") for d in dn.shuffles])
        z = (dn.real.total("b") - sh.mean(0)) / np.maximum(sh.std(0), floor)
        print(name, null, " | ".join(
            f"c={c:g}: db {a:+.4f} align {b:+.4f} cover {cc:+.4f} z {zz:.1f}"
            for c, a, b, cc, zz in zip(CS, dn.excess("b"), dn.excess("cov"), dn.excess("coverage"), z)), flush=True)
