"""Bench a one-pass DAG profile against SciPy traversal at every threshold.

The graph and weights are synthetic; results say nothing about model fidelity.
The reference constructs CSR and runs forward/reverse BFS at each threshold.
No timing claim is made against optimized competing all-threshold solvers.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from statistics import median
import time

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order
from percinterp import LayeredDAG, profile, rank_entry


def fixture(width=192, density=.55, seed=813):
    if width < 2 or not 0 < density <= 1:
        raise ValueError("width or density out of range")
    rng = np.random.default_rng(seed)
    sizes = [1, width, width, width, 1]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    s, d, w = [], [], []
    for layer, (left, right) in enumerate(zip(sizes[:-1], sizes[1:])):
        mask = rng.random((right, left)) < density
        i, j = np.nonzero(mask)
        s.append(j + offsets[layer])
        d.append(i + offsets[layer + 1])
        w.append(rng.lognormal(0, 1, size=len(i)))
    return LayeredDAG(sizes, np.concatenate(s), np.concatenate(d), np.concatenate(w))


def reference_curve(g, entry, ps):
    starts, sinks = np.flatnonzero(g.sources), np.flatnonzero(g.sinks)
    if len(starts) != 1 or len(sinks) != 1:
        raise ValueError("reference requires one source and one sink")
    out = []
    for p in ps:
        keep = entry <= p
        csr = csr_matrix(
            (np.ones(int(keep.sum()), dtype=np.int8), (g.src[keep], g.dst[keep])),
            shape=(g.n_nodes, g.n_nodes),
        )
        forward = breadth_first_order(csr, int(starts[0]), directed=True,
                                      return_predecessors=False)
        backward = breadth_first_order(csr.T.tocsr(), int(sinks[0]), directed=True,
                                       return_predecessors=False)
        a, b = np.zeros(g.n_nodes, dtype=bool), np.zeros(g.n_nodes, dtype=bool)
        a[forward] = True
        b[backward] = True
        out.append(np.mean(a & b))
    return np.asarray(out)


def benchmark(width=192, density=.55, points=41, repeats=3, seed=813):
    if points < 2 or repeats < 1:
        raise ValueError("points must be >=2; repeats must be >=1")
    graph = fixture(width, density, seed)
    entry = rank_entry(graph.weight, graph.edge_layer("dst"))
    grid = np.linspace(0, 1, points)
    fast = lambda: profile(graph, entry).backbone_node_fraction(grid)
    slow = lambda: reference_curve(graph, entry, grid)
    base, output = slow(), fast()
    difference = float(np.max(np.abs(base - output)))
    if difference > 1e-12:
        raise AssertionError(f"profile mismatch: {difference}")

    def timed(fn):
        readings = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            readings.append(time.perf_counter() - t0)
        return float(median(readings))

    profile_s, reference_s = timed(fast), timed(slow)
    return dict(
        schema="percinterp.benchmark.v1",
        fixture="synthetic five-layer DAG with lognormal edge magnitudes",
        method_profile="one forward/backward min-max pass with all-threshold queries",
        method_reference="repeated SciPy CSR+directed BFS at every threshold",
        seed=seed, width=width, density=density,
        nodes=graph.n_nodes, edges=graph.n_edges, points=points, repeats=repeats,
        max_curve_difference=difference,
        profile_seconds_median=profile_s, reference_seconds_median=reference_s,
        speedup_reference_over_profile=reference_s / profile_s,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        platform=platform.platform(),
        interpretation="synthetic graph runtime; not evidence of causal circuit fidelity",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--density", type=float, default=.55)
    ap.add_argument("--points", type=int, default=41)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=813)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = benchmark(args.width, args.density, args.points, args.repeats, args.seed)
    body = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(body, end="")


if __name__ == "__main__":
    main()
