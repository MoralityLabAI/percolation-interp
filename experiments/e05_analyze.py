"""e05: percolation sieve on the layer 0-1 VPD component graph (Q1), and Q4's null sets.

Graph as e04 (e04_analyze.build_graph). Sieved against within-layer and in-strength-preserving
shuffles, 1,000 each; Q1 passes when p <= 0.01 under both (docs/e05_design.md).

Q4's null sets: the backbones of the first 100 within-layer shuffled graphs at the real c*.
Nodes keep their identity under a shuffle (weights are permuted, not nodes), so a shuffled
backbone is a set of real components. Per module it is trimmed or padded to the real backbone's
size by taking that module's nodes in order of their entry c in the shuffled graph.

  python experiments/e05_analyze.py [--n-shuffles 1000]      (this repo's venv)
Writes <E04_OUT>/analysis.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from e04_analyze import OUT, build_graph, record
from percinterp.sieve import sieve

N_Q4 = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shuffles", type=int, default=1000)
    args = ap.parse_args()
    g, w, ids, n_alive = build_graph()
    print(f"graph: {g.n_nodes} nodes, {g.n_edges} edges, {int(g.sources.sum())} sources; alive per module {n_alive}",
          flush=True)
    report = {"n_alive": n_alive, "n_nodes": g.n_nodes, "n_edges": g.n_edges, "n_shuffles": args.n_shuffles}
    results = {}
    for name, kw in [("within_layer", {"keep_null_entries": True}), ("in_strength", {"null_groups": g.dst})]:
        res = sieve(g, w, n_shuffles=args.n_shuffles, rng=np.random.default_rng(len(name)), **kw)
        results[name] = res
        print(f"{name}: z_peak {res.z_peak:.2f} (shuffle max {res.null_peaks.max():.2f}), p {res.p_value:.4f}, "
              f"c* {res.c_star:.3g}, backbone {int(res.backbone.sum())} nodes", flush=True)
        report[name] = record(res, ids)
    report["Q1_pass"] = all(report[k]["p"] <= 0.01 for k in ("within_layer", "in_strength"))

    # Q4: shuffled graphs' backbones, sized per module like the real one.
    real = results["within_layer"]
    module_of = np.array([m for m, _ in ids])
    comp = np.array([c for _, c in ids])
    size = Counter(module_of[real.backbone & (module_of != "output")])
    nulls = []
    for k in range(N_Q4):
        entry = real.null_node_entry[k]
        chosen = []
        for m, n in size.items():
            idx = np.flatnonzero(module_of == m)
            order = idx[np.argsort(entry[idx], kind="stable")]
            chosen += [[m, int(comp[i])] for i in order[:n]]
        nulls.append(chosen)
    report["q4_null_backbones"] = nulls
    report["q4_null_on_backbone_at_cstar"] = [
        float(((real.null_node_entry[k] <= real.c_star) & (module_of != "output")).sum()) for k in range(N_Q4)]
    (OUT / "analysis.json").write_text(json.dumps(report))
    print(f"Q1 {'pass' if report['Q1_pass'] else 'fail'}; Q4 null sets: {N_Q4}, sizes per module {dict(size)}")


if __name__ == "__main__":
    main()
