"""Known answers for Bernoulli bond percolation on complete layered graphs.

Setting: layers of widths n_0..n_L, every node in layer l joined to every node
in layer l+1, each edge kept independently with probability p, all of layer 0
as sources and all of layer L as sinks.

If r nodes of layer l are reachable, each node of layer l+1 is reachable
independently with probability 1 - (1-p)^r (its in-edges from those r nodes
are disjoint from everyone else's). So the reachable count is an exact Markov
chain, R_{l+1} | R_l = r ~ Binomial(n_{l+1}, 1 - (1-p)^r), and the spanning
probability is 1 - P(R_L = 0). No approximation; use it to test the pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.special import lambertw
from scipy.stats import binom


def reach_chain(widths: list[int], p: float) -> dict[str, np.ndarray | float]:
    """Exact spanning probability and mean reachable fraction per layer."""
    widths = [int(w) for w in widths]
    dist = np.zeros(widths[0] + 1)
    dist[-1] = 1.0
    mean_frac = [1.0]
    transitions: dict[tuple[int, int], np.ndarray] = {}
    for n, m in zip(widths[:-1], widths[1:]):
        if (n, m) not in transitions:
            r = np.arange(n + 1)
            pi = 1.0 - (1.0 - p) ** r
            transitions[(n, m)] = binom.pmf(np.arange(m + 1)[None, :], m, pi[:, None])
        dist = dist @ transitions[(n, m)]
        mean_frac.append(float(dist @ np.arange(m + 1)) / m)
    return {"span_prob": float(1.0 - dist[0]), "mean_reach_frac": np.array(mean_frac)}


def mean_field_reach(c: float, depth: int) -> np.ndarray:
    """Large-width limit of the reachable fraction, with c = p * width.

    rho_{l+1} = 1 - exp(-c rho_l), rho_0 = 1. Converges to the giant-component
    fraction for c > 1, to 0 for c < 1, and decays like 2/l at c = 1.
    """
    rho = [1.0]
    for _ in range(depth):
        rho.append(1.0 - np.exp(-c * rho[-1]))
    return np.array(rho)


def giant_fraction(c) -> np.ndarray:
    """Fixed point of rho = 1 - exp(-c rho): the Erdos-Renyi giant-component fraction."""
    c = np.atleast_1d(np.asarray(c, dtype=float))
    out = np.zeros_like(c)
    hi = c > 1
    out[hi] = 1.0 + np.real(lambertw(-c[hi] * np.exp(-c[hi]))) / c[hi]
    return out


def path_count_threshold(widths: list[int]) -> float:
    """p at which the expected number of surviving input-output paths is 1.

    Paths = prod(widths), each needs L = len(widths)-1 edges, so p = paths^(-1/L).
    For equal widths N this is N^(-(L+1)/L): the spanning point at finite depth
    is set by path counting and only approaches 1/N as depth grows.
    """
    L = len(widths) - 1
    return float(np.exp(-np.sum(np.log(widths)) / L))
