"""Fixed e08 statistics, context clustering, and nonuniform bond percolation.

Context clusters are proposals, not validated semantic labels. This module
does not load a model, modify checkpoints, or allocate CUDA memory.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CANDIDATES = ((0, "gate", 196), (1, "gate", 33), (1, "up", 1037))
DISCOVERY_ROWS = EVALUATION_ROWS = 512
N_RANDOM, N_RESAMPLES, SEED = 100, 1000, 81022
# Frozen before e08 measurements. Dense all-coordinate graphs have a much
# earlier onset than top-eight graphs; a 0.1 first point cannot resolve it.
EXPOSURES = np.r_[0., np.geomspace(1e-5, 4., 40)]


def split_row_ids(n_rows, permutation):
    """Use the e04 causal range, never wrap an undersized cache."""
    order = np.asarray(permutation)
    if (order.shape != (n_rows,) or order.dtype.kind not in 'iu'
            or not np.array_equal(np.sort(order), np.arange(n_rows))):
        raise ValueError("expected a complete row permutation")
    if n_rows < 4000 + DISCOVERY_ROWS + EVALUATION_ROWS:
        raise ValueError("cached causal rows are insufficient; refusing substitution")
    return order[4000:4512].copy(), order[4512:5024].copy()


def selected_position(activation):
    """Noninitial position with a next token inside the same row; stable ties."""
    a = np.asarray(activation)
    if a.ndim != 1 or len(a) < 3 or not np.isfinite(a).all():
        raise ValueError("finite activation row of length >=3 required")
    return int(np.argmax(np.abs(a[1:-1]))) + 1


def normalize_rows(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite matrix required")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("zero contextual representation")
    return x / norm


def context_clusters(discovery, evaluation, k=4, seed=81024):
    """Spherical k-means++ on discovery only, nearest-centroid evaluation.

    Four proposals allow >=32 members in the 128-row discovery budget.
    Label review is a separate mandatory stage.
    """
    x, y = normalize_rows(discovery), normalize_rows(evaluation)
    if not 1 <= k <= len(x):
        raise ValueError("invalid cluster count")
    rng = np.random.default_rng(seed)
    centers = [x[int(rng.integers(len(x)))]]
    for _ in range(k - 1):
        distance = np.maximum(0, 1 - np.max(x @ np.array(centers).T, axis=1))
        if distance.sum() < 1e-12:
            return {"status": "degenerate_representations"}
        centers.append(x[int(rng.choice(len(x), p=distance / distance.sum()))])
    centers = np.array(centers)
    previous = None
    for _ in range(200):
        labels = np.argmax(x @ centers.T, axis=1)
        if previous is not None and np.array_equal(previous, labels):
            break
        for j in range(k):
            if not np.any(labels == j):
                return {"status": "empty_cluster"}
            mean = x[labels == j].mean(0)
            if np.linalg.norm(mean) < 1e-12:
                return {"status": "degenerate_centroid"}
            centers[j] = mean / np.linalg.norm(mean)
        previous = labels.copy()
    else:
        return {"status": "clustering_not_converged"}
    return {"status": "proposed", "centroids": centers,
            "discovery": labels, "evaluation": np.argmax(y @ centers.T, axis=1)}


def decile_bins(x):
    """Quantile bins keep exact ties together; empty bins are permitted."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or not len(x):
        raise ValueError("finite nonempty vector required")
    return np.searchsorted(np.quantile(x, np.arange(1, 10) / 10), x, side="left")


def matched_groups(groups, rate, energy, rng, n=N_RANDOM):
    """Without replacement per draw; exclude both selected groups, no bin relaxation."""
    rate, energy = np.asarray(rate), np.asarray(energy)
    if rate.shape != energy.shape or np.any(rate < 0) or np.any(energy < 0):
        raise ValueError("nonnegative paired component metrics required")
    bins = 10 * decile_bins(rate) + decile_bins(energy)
    excluded = np.unique(np.concatenate(groups))
    if np.any(excluded < 0) or np.any(excluded >= len(rate)):
        raise ValueError("component ID outside matching pool")
    available = np.setdiff1d(np.arange(len(rate)), excluded)
    output = []
    for group_index, group in enumerate(groups):
        wants, counts = np.unique(bins[group], return_counts=True)
        pools = [available[bins[available] == b] for b in wants]
        if any(len(pool) < count for pool, count in zip(pools, counts)):
            return {"status": "insufficient_exact_matches", "bins": bins.tolist(),
                    "group_index": group_index, "requested_ids": np.asarray(group).tolist(),
                    "deficits": [{"bin": int(b), "needed": int(count), "available": len(pool)}
                        for b, pool, count in zip(wants, pools, counts) if len(pool) < count]}
        output.append(np.array([np.concatenate([rng.choice(pool, int(count), replace=False)
            for pool, count in zip(pools, counts)]) for _ in range(n)], dtype=int))
    return {"status": "matched", "groups": output, "bins": bins.tolist()}


def contrast_groups(signed, labels, contacts, max_size=8):
    z, labels, contacts = np.asarray(signed), np.asarray(labels), np.asarray(contacts)
    if z.ndim != 2 or len(labels) != len(z) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("two labeled discovery families required")
    contrast = z[labels == 0].mean(0) - z[labels == 1].mean(0)
    order = contacts[np.lexsort((contacts, -contrast[contacts]))]
    n = min(max_size, len(order) // 2)
    if n == 0:
        raise ValueError("fewer than two contacting components")
    return [order[:n], order[-n:]], contrast


def stratify(position, activation, discovery_activation):
    """Four fixed position bands crossed with discovery activation quintiles."""
    position = np.asarray(position)
    magnitude = np.abs(np.asarray(activation))
    edges = np.quantile(np.abs(discovery_activation), [.2, .4, .6, .8])
    return (5 * np.minimum(position // 32, 3)
            + np.searchsorted(edges, magnitude, side="left")).astype(int)


def dissociation(effects, labels, strata, rng, resamples=N_RESAMPLES):
    """Both directed KL interactions must be positive, common-stratum weighting.

    Permutations and bootstrap preserve row units and within-stratum counts.
    One row has exactly one observation per candidate.
    """
    effects = np.asarray(effects, dtype=float)
    labels, strata = np.asarray(labels), np.asarray(strata)
    if effects.shape != (len(labels), 2) or len(strata) != len(labels) or not np.isfinite(effects).all():
        raise ValueError("finite row-by-two effect matrix required")
    cells = []
    for s in np.unique(strata):
        ids = [np.flatnonzero((strata == s) & (labels == k)) for k in (0, 1)]
        if min(map(len, ids)) >= 2:
            cells.append(ids)
    if (min([np.sum(labels == k) for k in (0, 1)]) < 32 or not cells
            or min(sum(len(c[k]) for c in cells) for k in (0, 1)) < 16):
        return {"status": "insufficient_matched_rows"}
    weights = np.array([min(map(len, c)) for c in cells], dtype=float)
    weights /= weights.sum()
    def matrix(groups):
        return sum(w * np.array([effects[ids].mean(0) for ids in c])
                   for w, c in zip(weights, groups))
    def differences(m):
        return np.array([m[0, 0] - m[1, 0], m[1, 1] - m[0, 1]])
    observed = matrix(cells)
    directed = differences(observed)
    null, boot = [], []
    for _ in range(resamples):
        shuffled, sampled = [], []
        for a, b in cells:
            p = rng.permutation(np.r_[a, b])
            shuffled.append([p[:len(a)], p[len(a):]])
            sampled.append([rng.choice(a, len(a), replace=True), rng.choice(b, len(b), replace=True)])
        null.append(float(np.min(differences(matrix(shuffled)))))
        boot.append(differences(matrix(sampled)))
    return {"status": "scored", "matrix_family_by_group": observed.tolist(),
            "directed_interactions": directed.tolist(), "statistic_min": float(directed.min()),
            "p": float((1 + np.sum(np.array(null) >= directed.min())) / (resamples + 1)),
            "interaction_ci95": np.quantile(boot, [.025, .975], axis=0).T.tolist(),
            "matched_rows_per_family": [sum(len(c[k]) for c in cells) for k in (0, 1)],
            "both_positive": bool(np.all(directed > 0)), "resamples": resamples}


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite and in [0,1]")
    order = np.argsort(p, kind="stable")
    answer = np.empty_like(p)
    answer[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return answer


def weighted_curve(q, exposures, uniforms):
    """Exact nested independent contacts; sparse CC stops after all LEFTs join.

    The early stop changes no measurements: future edge additions cannot
    split the sole left-bearing cluster. No critical-point approximation.
    """
    q = np.asarray(q, dtype=float)
    ts, u = np.asarray(exposures, dtype=float), np.asarray(uniforms, dtype=float)
    if (q.ndim != 2 or min(q.shape) < 1 or not np.isfinite(q).all() or np.any(q < 0)
            or not np.allclose(q.sum(1), 1) or u.shape != q.shape
            or not np.isfinite(u).all() or np.any((u < 0) | (u >= 1))
            or ts.ndim != 1 or not np.isfinite(ts).all() or np.any(ts < 0) or np.any(np.diff(ts) < 0)):
        raise ValueError("unit-energy rows, sorted nonnegative exposures and edge uniforms required")
    nl, nr = q.shape
    arrival = np.full(q.shape, np.inf)
    np.divide(-np.log1p(-u), nr * q, out=arrival, where=q > 0)
    giant, chi, count = [], [], []
    connected = False
    for t in ts:
        if connected:
            giant.append(1.); chi.append(0.); count.append(1)
            continue
        left, right = np.nonzero(arrival < t)
        graph = coo_matrix((np.ones(len(left), dtype=np.int8), (left, nl + right)),
                           shape=(nl + nr, nl + nr)).tocsr()
        _, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels[:nl])
        sizes = sizes[sizes > 0]
        largest = int(sizes.max())
        giant.append(largest / nl)
        rest = nl - largest
        chi.append(float((np.sum(sizes ** 2) - largest ** 2) / rest) if rest else 0.)
        count.append(len(sizes))
        connected = rest == 0
    return {"giant_left_fraction": np.array(giant), "susceptibility_left": np.array(chi),
            "left_cluster_count": np.array(count)}


def weighted_expectations(q, exposures):
    q = np.asarray(q, dtype=float)
    expected_edges, retained = [], []
    for t in exposures:
        probability = -np.expm1(-float(t) * q.shape[1] * q)
        expected_edges.append(float(probability.mean()))
        retained.append(float(np.sum(q * probability) / len(q)))
    order = np.sort(q, axis=1)[:, ::-1]
    cumulative = np.cumsum(order, axis=1)
    return {"expected_edge_fraction": expected_edges, "expected_retained_energy": retained,
            "contacts_for_50pct_energy": (np.argmax(cumulative >= .5 - 1e-12, axis=1) + 1).tolist(),
            "contacts_for_90pct_energy": (np.argmax(cumulative >= .9 - 1e-12, axis=1) + 1).tolist()}
