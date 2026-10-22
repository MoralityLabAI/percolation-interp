"""E10 budget-matched selectors. Selection uses attribution/statistics rows only.

The scorer is deliberately independent of target model execution. In particular,
the local-degree null never traverses source-to-sink paths. Freeze its manifest
before doing any causal inference on held-out rows.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Mapping

import numpy as np


DETERMINISTIC = ("sieve", "local_degree", "local_weighted_degree", "top_ci",
                 "top_attr", "top_rate")
SCHEMA = "percinterp.e10.selector.v1"


def contact_scores(n_nodes: int, src, dst, weight, entry, occupation: float) -> dict:
    """Local graph statistics at the frozen occupation; never path reachability."""
    s, d = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)
    w, e = np.asarray(weight, dtype=np.float64), np.asarray(entry, dtype=np.float64)
    if n_nodes <= 0 or any(len(v) != len(s) for v in (d, w, e)):
        raise ValueError("node/edge sizes are inconsistent")
    if not (np.isfinite(occupation) and occupation >= 0):
        raise ValueError("occupation must be finite and >= 0")
    if len(s) and (min(s.min(), d.min()) < 0 or max(s.max(), d.max()) >= n_nodes):
        raise ValueError("edge endpoint outside graph")
    if np.any(~np.isfinite(w)):
        raise ValueError("all edge weights must be finite")
    kept = np.isfinite(e) & (e <= occupation) & (w != 0)
    incident_in = np.bincount(d[kept], minlength=n_nodes)
    incident_out = np.bincount(s[kept], minlength=n_nodes)
    aw = np.abs(w)
    kept_in = np.bincount(d[kept], weights=aw[kept], minlength=n_nodes)
    kept_out = np.bincount(s[kept], weights=aw[kept], minlength=n_nodes)
    full_in = np.bincount(d, weights=aw, minlength=n_nodes)
    full_out = np.bincount(s, weights=aw, minlength=n_nodes)
    return {
        "in_degree": incident_in.astype(int),
        "out_degree": incident_out.astype(int),
        "two_sided": ((incident_in > 0) & (incident_out > 0)).astype(int),
        "balanced_degree": np.minimum(incident_in, incident_out).astype(int),
        "incident_degree": (incident_in + incident_out).astype(int),
        "kept_two_sided_strength": np.sqrt(kept_in * kept_out),
        "kept_total_strength": kept_in + kept_out,
        "total_absolute_attribution": full_in + full_out,
    }


def graph_fingerprint(n_nodes, src, dst, weight, entry, node_ids) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"{SCHEMA}:{n_nodes}\n".encode("ascii"))
    for values, dtype in ((src, "<i8"), (dst, "<i8"),
                          (weight, "<f8"), (entry, "<f8")):
        a = np.asarray(values, dtype=dtype)
        hasher.update(f"{dtype}:{a.size}\n".encode("ascii"))
        hasher.update(a.tobytes())
    hasher.update(json.dumps([[str(m), int(i)] for m, i in node_ids],
                             separators=(",", ":")).encode())
    return hasher.hexdigest()


def rank_top(pool, quota, key):
    p = [int(i) for i in pool]
    if len(set(p)) != len(p) or not (0 <= quota <= len(p)):
        raise ValueError("invalid pool or module quota")
    return sorted(sorted(p, key=lambda i: tuple(-float(x) for x in key(i)) + (i,))[:quota])


def _rate_matched_draw(pool, chosen, rate, rng):
    """Random without replacement within firing-rate deciles; nearest-bin fallback."""
    pool = list(map(int, pool))
    chosen = list(map(int, chosen))
    if not chosen:
        return [], {"same_decile": 0, "fallback": 0}
    rates = np.asarray(rate, dtype=np.float64)
    if not np.all(np.isfinite(rates[pool])) or np.any(rates[pool] < 0):
        raise ValueError("firing rates must be finite and nonnegative")
    qs = np.quantile(rates[pool], np.linspace(0, 1, 11))
    def decile(i):
        return int(np.clip(np.searchsorted(qs, rates[i], side="right") - 1, 0, 9))
    # Random processing order avoids consistently privileging the first ID.
    target_order = list(rng.permutation(chosen))
    remaining = set(pool)
    selected = []
    matched = 0
    for target in target_order:
        if not remaining:
            raise ValueError("not enough available candidates")
        need = decile(target)
        distances = {i: abs(decile(i) - need) for i in remaining}
        min_distance = min(distances.values())
        candidates = [i for i in remaining if distances[i] == min_distance]
        draw = int(rng.choice(candidates))
        selected.append(draw)
        matched += int(min_distance == 0)
        remaining.remove(draw)
    return sorted(selected), {"same_decile": matched, "fallback": len(chosen)-matched}


def select_sets(*, n_nodes: int, src, dst, weight, entry, occupation: float,
                node_ids, pools: Mapping[str, list[int]], backbone: Mapping[str, list[int]],
                ci_mean: Mapping[str, object], firing_rate: Mapping[str, object],
                seed: int = 20261022, n_random: int = 20) -> dict:
    """Freeze all same-per-module-budget selectors from a fixed graph and metadata.

    node_ids: (module, local component index), in the graph's node order.
    pools: alive component indices per module; no source or sink nodes.
    backbone: frozen E03 sieve component set by module, derived previously.
    ci_mean/rate: 1D arrays indexed by local component index per module.
    """
    if len(node_ids) != n_nodes or n_random < 0 or n_random > 1000:
        raise ValueError("invalid node IDs or random-draw count")
    pairs = [(str(m), int(i)) for m, i in node_ids]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate graph node ID")
    positions = {pair: idx for idx, pair in enumerate(pairs)}
    mods = sorted(pools)
    if set(mods) != set(backbone) or set(mods) != set(ci_mean) or set(mods) != set(firing_rate):
        raise ValueError("module keys must match across pools, backbone and metadata")
    stats = contact_scores(n_nodes, src, dst, weight, entry, occupation)
    plans = {name: {} for name in DETERMINISTIC}
    quotas = {}
    candidate_overlap = {}
    for mod in mods:
        pool = sorted(int(x) for x in pools[mod])
        bb = sorted(int(x) for x in backbone[mod])
        if len(set(bb)) != len(bb) or not set(bb).issubset(pool):
            raise ValueError(f"backbone outside eligible pool for {mod}")
        for i in pool:
            if (mod, i) not in positions:
                raise ValueError(f"candidate {(mod, i)} missing from graph")
        if not pool and bb:
            raise ValueError("empty pool cannot contain backbone")
        ci = np.asarray(ci_mean[mod], dtype=np.float64)
        rate = np.asarray(firing_rate[mod], dtype=np.float64)
        if pool and (len(ci) <= max(pool) or len(rate) <= max(pool)):
            raise ValueError(f"missing metadata entries for {mod}")
        if pool and (not np.all(np.isfinite(ci[pool])) or not np.all(np.isfinite(rate[pool]))):
            raise ValueError("non-finite component metadata")
        quotas[mod] = len(bb)
        plans["sieve"][mod] = bb
        plans["local_degree"][mod] = rank_top(pool, len(bb), lambda i: (
            stats["two_sided"][positions[(mod, i)]],
            stats["balanced_degree"][positions[(mod, i)]],
            stats["incident_degree"][positions[(mod, i)]]))
        plans["local_weighted_degree"][mod] = rank_top(pool, len(bb), lambda i: (
            stats["kept_two_sided_strength"][positions[(mod, i)]],
            stats["kept_total_strength"][positions[(mod, i)]]))
        plans["top_ci"][mod] = rank_top(pool, len(bb), lambda i: (ci[i],))
        plans["top_attr"][mod] = rank_top(pool, len(bb), lambda i: (
            stats["total_absolute_attribution"][positions[(mod, i)]],))
        plans["top_rate"][mod] = rank_top(pool, len(bb), lambda i: (rate[i],))
        candidate_overlap[mod] = {"pool": len(pool), "quota": len(bb),
                                  "two_sided_candidates": int(sum(
                                      stats["two_sided"][positions[(mod, i)]]
                                      for i in pool))}
    rng = np.random.default_rng(seed)
    random_quality = {}
    for draw in range(n_random):
        name = f"random_{draw:03d}"
        plans[name] = {}
        random_quality[name] = {}
        for mod in mods:
            chosen, diag = _rate_matched_draw(pools[mod], backbone[mod],
                                             firing_rate[mod], rng)
            plans[name][mod] = chosen
            random_quality[name][mod] = diag
    overlaps = {}
    sieve_set = {(m, i) for m, ids in plans["sieve"].items() for i in ids}
    for name in DETERMINISTIC[1:]:
        comp = {(m, i) for m, ids in plans[name].items() for i in ids}
        intersection = len(comp & sieve_set)
        union = len(comp | sieve_set)
        overlaps[name] = {
            "same_selected_components": bool(comp == sieve_set),
            "intersection": intersection,
            "union": union,
            "jaccard": intersection / union if union else 1.0,
            "replacements": len(sieve_set - comp),
        }
    return {
        "schema": SCHEMA,
        "graph_sha256": graph_fingerprint(n_nodes, src, dst, weight, entry, pairs),
        "occupation": float(occupation),
        "n_nodes": int(n_nodes),
        "n_edges": len(src),
        "module_quotas": quotas,
        "pool_stats": candidate_overlap,
        "selection": plans,
        "overlap": overlaps,
        "rate_match_quality": random_quality,
        "design": {
            "sieve": "previously frozen source-to-sink backbone at c_star",
            "local_degree": "two-sided retained contact, then min(in,out) and total degree",
            "local_weighted_degree": "sqrt(kept incoming abs-weight * kept outgoing abs-weight)",
            "top_ci": "top mean causal importance among alive components",
            "top_attr": "top total incident absolute attribution on the full graph",
            "top_rate": "top activity rate among alive components",
            "random": "without replacement, approximate firing-rate decile matching",
            "tie_break": "ascending local component ID",
            "budget": "identical per-module count for every selector",
            "source_rows": "previously saved E03 attribution/stats; causal rows not used",
        },
        "seed": int(seed),
    }
