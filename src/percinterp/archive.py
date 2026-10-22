"""Portable, pickle-free NPZ storage for imported influence graphs.

The archive carries signed values, external vertex identities, stage sizes,
boundary masks and user-supplied provenance. It is not a VPD checkpoint reader
and does not certify the originating attribution method.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np

from .graph import LayeredDAG
from .interop import ImportedGraph

SCHEMA = "percinterp.influence_graph.v1"
KEYS = frozenset({
    "schema", "node_ids", "layer_sizes", "src", "dst", "signed_weight",
    "sources", "sinks", "provenance_json",
})


def save_graph_npz(
    path: str | Path,
    data: ImportedGraph,
    *,
    provenance: Mapping[str, str] | None = None,
) -> Path:
    """Write compressed arrays with NumPy object serialization disabled.

    Record producer run/checkpoint, target, sampling, direct-attribution
    method, aggregation and other evidence in provenance when available.
    """
    dest = Path(path)
    if dest.suffix != ".npz":
        raise ValueError("archive path must end in .npz")
    meta = dict(provenance or {})
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in meta.items()):
        raise TypeError("provenance keys and values must be strings")
    dest.parent.mkdir(parents=True, exist_ok=True)
    g = data.graph
    np.savez_compressed(
        dest,
        schema=np.asarray(SCHEMA),
        node_ids=np.asarray(data.node_ids, dtype=str),
        layer_sizes=np.asarray(g.layer_sizes, dtype=np.int64),
        src=g.src.astype(np.int64, copy=False),
        dst=g.dst.astype(np.int64, copy=False),
        signed_weight=data.signed_weight.astype(np.float64, copy=False),
        sources=g.sources.astype(bool, copy=False),
        sinks=g.sinks.astype(bool, copy=False),
        provenance_json=np.asarray(json.dumps(meta, sort_keys=True, allow_nan=False)),
    )
    return dest


def _integers(arr: np.ndarray, field: str) -> np.ndarray:
    if arr.ndim != 1 or arr.dtype.kind not in "iu":
        raise ValueError(f"{field} must be a 1D integer array")
    if arr.dtype.kind == "u" and arr.size and arr.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{field} exceeds signed 64-bit range")
    return arr.astype(np.int64, copy=False)


def load_graph_npz(path: str | Path) -> tuple[ImportedGraph, dict[str, str]]:
    """Strictly validate and decode a v1 PercInterp graph archive.

    NumPy pickle/object arrays are prohibited. The graph remains a diagnostic,
    irrespective of the provenance claims supplied by its producer.
    """
    source = Path(path)
    if source.suffix != ".npz":
        raise ValueError("archive path must end in .npz")
    with np.load(source, allow_pickle=False) as z:
        if set(z.files) != KEYS:
            raise ValueError("invalid graph archive fields")
        schema = z["schema"]
        if schema.ndim != 0 or str(schema.item()) != SCHEMA:
            raise ValueError("unsupported graph archive schema")
        ids = z["node_ids"]
        sizes = _integers(z["layer_sizes"], "layer_sizes")
        src = _integers(z["src"], "src")
        dst = _integers(z["dst"], "dst")
        w = z["signed_weight"]
        sm = z["sources"]
        tm = z["sinks"]
        meta = z["provenance_json"]
    if ids.ndim != 1 or ids.dtype.kind != "U" or not ids.size:
        raise ValueError("node_ids must be a nonempty Unicode array")
    names = tuple(str(v) for v in ids)
    if any(not x for x in names) or len(set(names)) != len(names):
        raise ValueError("node IDs must be unique and nonempty")
    n = len(names)
    if len(sizes) < 2 or np.any(sizes <= 0) or sizes.sum() != n:
        raise ValueError("invalid layer_sizes")
    if src.shape != dst.shape or w.shape != src.shape:
        raise ValueError("edge arrays have inconsistent shapes")
    if w.dtype.kind not in "fiu" or np.any(~np.isfinite(w)) or np.any(w == 0):
        raise ValueError("edge weights must be numeric, finite, and nonzero")
    if np.any(src < 0) or np.any(dst < 0) or np.any(src >= n) or np.any(dst >= n):
        raise ValueError("edge index outside node array")
    if len(src) > 1:
        perm = np.lexsort((dst, src))
        a, b = src[perm], dst[perm]
        if np.any((a[1:] == a[:-1]) & (b[1:] == b[:-1])):
            raise ValueError("duplicate directed edges")
    if (sm.shape != (n,) or tm.shape != (n,) or
            sm.dtype.kind != "b" or tm.dtype.kind != "b"):
        raise ValueError("invalid source/sink masks")
    if not np.any(sm) or not np.any(tm) or np.any(sm & tm):
        raise ValueError("source/sink masks must be nonempty and disjoint")
    if meta.ndim != 0 or meta.dtype.kind != "U":
        raise ValueError("provenance must be a Unicode JSON scalar")
    try:
        provenance = json.loads(str(meta.item()))
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid provenance JSON") from exc
    if (not isinstance(provenance, dict) or
            any(not isinstance(k, str) or not isinstance(v, str)
                for k, v in provenance.items())):
        raise ValueError("provenance must map strings to strings")
    g = LayeredDAG(
        list(map(int, sizes)), src, dst, np.abs(w.astype(np.float64)),
        sources=sm, sinks=tm,
    )
    return ImportedGraph(g, names, w.astype(np.float64), 0), provenance
