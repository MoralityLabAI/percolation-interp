"""Vectorized import of signed attribution edges from NumPy or JAX producers.

An attribution producer supplies stable vertex IDs, topological stages and
indexed directed edges. The graph stores absolute weights for filtration;
the original signed values survive in ImportedGraph.signed_weight.
JAX users can pass jax.device_get arrays. No JAX dependency is required.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .graph import LayeredDAG
from .interop import ImportedGraph


def from_attribution_arrays(
    node_ids: Sequence[str],
    layers: Sequence[int],
    src_indices: np.ndarray,
    dst_indices: np.ndarray,
    signed_weight: np.ndarray,
    *,
    sources: Sequence[str] | None = None,
    sinks: Sequence[str] | None = None,
) -> ImportedGraph:
    """Convert indexed attribution arrays to a stage-sorted LayeredDAG.

    Edge endpoints refer to the original node_ids order. All edges must go
    from an earlier stage to a later one. Duplicate endpoints are rejected,
    including duplicates with zero weight. Exact zero weights are removed.
    This consumes an existing graph, without validating its causal origin.
    """
    ids = tuple(node_ids)
    if not ids or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("node IDs must be nonempty strings")
    if len(set(ids)) != len(ids) or len(layers) != len(ids):
        raise ValueError("node IDs must be unique and aligned with layers")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
           or x < 0 for x in layers):
        raise ValueError("layers must be nonnegative integer topological stages")
    stage = np.asarray(layers, dtype=np.int64)
    levels = np.unique(stage)
    if len(levels) < 2 or not np.array_equal(levels, np.arange(len(levels))):
        raise ValueError("stages must be consecutive, nonempty, and begin at zero")

    n = len(ids)
    ordering = np.argsort(stage, kind="stable")
    sorted_ids = tuple(ids[int(i)] for i in ordering)
    inverse = np.empty(n, dtype=np.int64)
    inverse[ordering] = np.arange(n)
    sorted_stage = stage[ordering]

    def mask(requested: Sequence[str] | None, default: int) -> np.ndarray:
        if requested is None:
            names = {sorted_ids[i] for i in np.flatnonzero(sorted_stage == default)}
        else:
            if isinstance(requested, str):
                raise ValueError("source/sink IDs must be sequences, not strings")
            names = set(requested)
            if len(names) != len(requested):
                raise ValueError("duplicate source/sink IDs")
        if not names or not names.issubset(sorted_ids):
            raise ValueError("source/sink IDs must be nonempty and known")
        return np.fromiter((x in names for x in sorted_ids), dtype=bool, count=n)

    src_mask = mask(sources, 0)
    dst_mask = mask(sinks, len(levels) - 1)
    if np.any(src_mask & dst_mask):
        raise ValueError("source and sink sets must be disjoint")
    a = np.asarray(src_indices)
    b = np.asarray(dst_indices)
    w = np.asarray(signed_weight)
    if a.ndim != 1 or b.ndim != 1 or w.ndim != 1 or not (a.shape == b.shape == w.shape):
        raise ValueError("edge arrays must be one-dimensional and equal-length")
    if a.dtype.kind not in "iu" or b.dtype.kind not in "iu":
        raise ValueError("edge endpoints must have integer dtype")
    limit = np.iinfo(np.int64).max
    if (a.dtype.kind == "u" and a.size and a.max() > limit or
            b.dtype.kind == "u" and b.size and b.max() > limit):
        raise ValueError("edge endpoints exceed signed 64-bit range")
    a = a.astype(np.int64, copy=False)
    b = b.astype(np.int64, copy=False)
    try:
        w = w.astype(np.float64, copy=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("edge weights must be numeric") from exc
    if not np.all(np.isfinite(w)):
        raise ValueError("edge weights must be finite")
    if np.any(a < 0) or np.any(b < 0) or np.any(a >= n) or np.any(b >= n):
        raise ValueError("edge endpoints are outside node_ids")
    if np.any(stage[a] >= stage[b]):
        raise ValueError("edges must go from earlier to later stages")
    a, b = inverse[a], inverse[b]
    if len(a) > 1:
        perm = np.lexsort((b, a))
        sa, sb = a[perm], b[perm]
        if np.any((sa[1:] == sa[:-1]) & (sb[1:] == sb[:-1])):
            raise ValueError("duplicate directed edges: aggregate upstream")
    keep = w != 0
    g = LayeredDAG(
        list(map(int, np.bincount(stage))),
        a[keep], b[keep], np.abs(w[keep]),
        sources=src_mask, sinks=dst_mask,
    )
    return ImportedGraph(g, sorted_ids, w[keep], int((~keep).sum()))
