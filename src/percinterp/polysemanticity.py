"""Bipartite component/coordinate sharing and bond-percolation diagnostics.

These are structural measurements, not semantic labels or causal interventions.
The degree-moment threshold assumes a sparse, locally tree-like configuration
ensemble; it is not a theorem about an arbitrary learned finite support graph.
"""
from __future__ import annotations

import numpy as np


def direction_energy(directions):
    """Unit row energy, invariant to each component's nonzero scalar gauge."""
    a = np.asarray(directions, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError("directions must be a nonempty finite matrix")
    # Scale first to avoid overflow/underflow in squared norms.
    scale = np.max(np.abs(a), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("zero directions have no normalized support")
    a = a / scale
    q = a * a
    return q / q.sum(axis=1, keepdims=True)


def support_edges(energy, k):
    """Largest k strictly positive energy entries per row; index-stable ties."""
    q = np.asarray(energy, dtype=float)
    if q.ndim != 2 or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("energy must be a finite nonnegative matrix")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= q.shape[1]:
        raise ValueError("k must be an integer in [1, number of coordinates]")
    order = np.argsort(-q, axis=1, kind="stable")[:, :k]
    left = np.repeat(np.arange(len(q)), k)
    right = order.ravel()
    positive = q[left, right] > 0
    return left[positive], right[positive]


def _edges(left, right, n_left, n_right):
    left, right = np.asarray(left), np.asarray(right)
    if (left.ndim != 1 or right.shape != left.shape or n_left < 1 or n_right < 1
            or left.dtype.kind not in "iu" or right.dtype.kind not in "iu"):
        raise ValueError("integer edge arrays and positive part sizes required")
    if (np.any(left < 0) or np.any(left >= n_left)
            or np.any(right < 0) or np.any(right >= n_right)):
        raise ValueError("edge index out of range")
    return left.astype(np.int64), right.astype(np.int64)


def degree_summary(left, right, n_left, n_right):
    left, right = _edges(left, right, n_left, n_right)
    dl, dr = np.bincount(left, minlength=n_left), np.bincount(right, minlength=n_right)
    def excess(degree):
        return float(np.sum(degree * (degree - 1)) / degree.sum()) if degree.sum() else 0.0
    kl, kr = excess(dl), excess(dr)
    product = kl * kr
    return {"left": dl, "right": dr, "excess_left": kl, "excess_right": kr,
            "branching_product": product,
            "configuration_bond_pc": 1 / np.sqrt(product) if product > 0 else None}


def bond_curve(left, right, n_left, n_right, probabilities, priorities):
    """Nested Bernoulli bond draws U_e < p; cluster mass counts LEFT vertices.

    Parallel edges, when present in a configuration multigraph, are independently
    occupied. Right-only clusters do not enter the susceptibility denominator.
    """
    left, right = _edges(left, right, n_left, n_right)
    ps = np.asarray(probabilities, dtype=float)
    u = np.asarray(priorities, dtype=float)
    if (ps.ndim != 1 or not np.isfinite(ps).all() or np.any(np.diff(ps) < 0)
            or np.any(ps < 0) or np.any(ps > 1) or u.shape != left.shape
            or not np.isfinite(u).all() or np.any(u < 0) or np.any(u >= 1)):
        raise ValueError("sorted probabilities in [0,1] and edge uniforms in [0,1) required")
    order = np.argsort(u, kind="stable")
    stops = np.searchsorted(u[order], ps, side="left")
    parent = np.arange(n_left + n_right)
    size = np.ones(n_left + n_right, dtype=np.int64)
    mass = np.r_[np.ones(n_left, dtype=np.int64), np.zeros(n_right, dtype=np.int64)]
    second, largest, count, at = n_left, 1, n_left, 0
    giant, susceptibility, counts = [], [], []
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for stop in stops:
        for index in order[at:stop]:
            a, b = find(left[index]), find(n_left + right[index])
            if a == b:
                continue
            if size[a] < size[b]:
                a, b = b, a
            ma, mb = int(mass[a]), int(mass[b])
            second += 2 * ma * mb
            count -= int(ma > 0 and mb > 0)
            parent[b] = a
            size[a] += size[b]
            mass[a] += mass[b]
            largest = max(largest, int(mass[a]))
        at = stop
        giant.append(largest / n_left)
        remaining = n_left - largest
        susceptibility.append((second - largest ** 2) / remaining if remaining else 0.0)
        counts.append(count)
    return {"giant_left_fraction": np.asarray(giant),
            "susceptibility_left": np.asarray(susceptibility),
            "left_cluster_count": np.asarray(counts)}


def configuration_giant(degree_left, degree_right, probabilities, tol=1e-11):
    """Asymptotic two-type PGF prediction for a configuration ensemble.

    x=1-p+p G_R1(y), y=1-p+p G_L1(x), S_L=1-G_L0(x).
    This deliberately returns zero at/below the generic branching threshold;
    critical degenerate ensembles and finite hubs require direct measurement.
    """
    dl, dr = np.asarray(degree_left), np.asarray(degree_right)
    if (dl.ndim != 1 or dr.ndim != 1 or len(dl) == 0 or len(dr) == 0
            or dl.dtype.kind not in "iu" or dr.dtype.kind not in "iu"
            or np.any(dl < 0) or np.any(dr < 0) or dl.sum() != dr.sum()):
        raise ValueError("nonnegative integer degrees with equal stub sums required")
    ps = np.asarray(probabilities, dtype=float)
    if not np.isfinite(ps).all() or np.any(ps < 0) or np.any(ps > 1):
        raise ValueError("probabilities must be in [0,1]")
    if dl.sum() == 0:
        return np.zeros_like(ps)
    def pgfs(degrees):
        p0 = np.bincount(degrees) / len(degrees)
        p1 = np.arange(1, len(p0)) * p0[1:] / degrees.mean()
        return p0, p1
    l0, l1 = pgfs(dl)
    _, r1 = pgfs(dr)
    kl = np.sum(dl * (dl - 1)) / dl.sum()
    kr = np.sum(dr * (dr - 1)) / dr.sum()
    x, y = np.zeros_like(ps), np.zeros_like(ps)
    subcritical = ps * ps * kl * kr <= 1 + 1e-12
    x[subcritical], y[subcritical] = 1, 1
    for _ in range(20000):
        xx = 1 - ps + ps * np.polynomial.polynomial.polyval(y, r1)
        yy = 1 - ps + ps * np.polynomial.polynomial.polyval(x, l1)
        xx[subcritical], yy[subcritical] = 1, 1
        error = max(np.max(np.abs(xx - x)), np.max(np.abs(yy - y)))
        x, y = xx, yy
        if error < tol:
            break
    else:
        raise RuntimeError("configuration fixed point failed to converge")
    s = np.clip(1 - np.polynomial.polynomial.polyval(x, l0), 0, 1)
    s[ps * ps * kl * kr <= 1 + 1e-12] = 0
    return s


def mix_basis(directions, rng, rounds=3):
    """Seeded signed/permuted block-Hadamard orthogonal coordinate changes.

    Preserves all row inner products. This is a geometry control, not a claim
    that rotating a nonlinear MLP preserves the network's computation.
    """
    a = np.array(directions, dtype=np.float64, copy=True)
    if a.ndim != 2 or not np.isfinite(a).all() or a.shape[1] < 2:
        raise ValueError("finite matrix with at least two coordinates required")
    width = a.shape[1] & -a.shape[1]
    if width == 1:
        raise ValueError("Hadamard control requires an even coordinate dimension")
    for _ in range(rounds):
        a = a[:, rng.permutation(a.shape[1])] * rng.choice([-1, 1], size=a.shape[1])
        blocks = a.reshape(len(a), -1, width)
        stride = 1
        while stride < width:
            view = blocks.reshape(len(a), -1, 2, stride)
            lo, hi = view[:, :, 0].copy(), view[:, :, 1].copy()
            view[:, :, 0], view[:, :, 1] = lo + hi, lo - hi
            stride *= 2
        a /= np.sqrt(width)
    return a
