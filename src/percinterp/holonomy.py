"""Gauge-covariant discrete frame transport on a percolating contact graph.

Each directed edge transport maps column coordinates in the src frame to the
corresponding coordinates in the dst frame. An arbitrary local basis change
U_i -> U_i R_i conjugates a loop operator at its base point. Consequently its
eigenphases and distance to identity are gauge invariant.
"""
from __future__ import annotations

import numpy as np

from .cycle_filtration import BornCycle, cycle_filtration


def frame_edge_transports(frames: np.ndarray, src: np.ndarray, dst: np.ndarray,
                          *, min_overlap: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polar Procrustes transport Q_{u->v} = polar(U_v.T @ U_u).

    frames: [node, ambient_dim, rank], with orthonormal columns. The smallest
    singular value of cross-frame overlap marks reliably aligned links. The
    caller must also validate that frames describe comparable feature patches.
    """
    frames = np.asarray(frames, dtype=float)
    src, dst = np.asarray(src, dtype=int), np.asarray(dst, dtype=int)
    if frames.ndim != 3 or frames.shape[1] < frames.shape[2] or frames.shape[2] < 1:
        raise ValueError("frames must have shape (nodes, ambient>=rank, rank>=1)")
    if src.shape != dst.shape or src.ndim != 1 or np.any(src < 0) or np.any(dst < 0) or \
            np.any(src >= len(frames)) or np.any(dst >= len(frames)):
        raise ValueError("invalid edge endpoints")
    if not 0 <= min_overlap <= 1:
        raise ValueError("min_overlap must be in [0, 1]")
    if not np.isfinite(frames).all():
        raise ValueError("frames must be finite")
    gram = frames.transpose(0, 2, 1) @ frames
    if not np.allclose(gram, np.eye(frames.shape[2]), atol=1e-5, rtol=0):
        raise ValueError("frame columns must be orthonormal")
    cross = frames[dst].transpose(0, 2, 1) @ frames[src]
    a, sigma, bt = np.linalg.svd(cross)
    q = a @ bt
    weakest = sigma[:, -1] if len(sigma) else np.empty(0)
    reliable = weakest >= min_overlap
    return q, reliable, weakest


def loop_matrix(transports: np.ndarray, cycle: BornCycle) -> tuple[np.ndarray, list[np.ndarray]]:
    """Ordered product of edge maps, plus cumulative maps after each step."""
    q = np.asarray(transports, dtype=float)
    if q.ndim != 3 or q.shape[1] != q.shape[2] or q.shape[1] < 1:
        raise ValueError("transports must have shape (edges, rank, rank)")
    if not np.isfinite(q).all() or not np.allclose(q.transpose(0, 2, 1) @ q,
                                                   np.eye(q.shape[1]), atol=1e-6, rtol=0):
        raise ValueError("each edge transport must be orthogonal")
    product = np.eye(q.shape[1])
    cumulative = []
    for edge, sign in cycle.steps:
        if not 0 <= edge < len(q) or sign not in (-1, +1):
            raise ValueError("invalid signed cycle edge")
        product = (q[edge] if sign == +1 else q[edge].T) @ product
        cumulative.append(product.copy())
    return product, cumulative


def signature(H: np.ndarray) -> dict:
    """Gauge-invariant spectral diagnostics; rank-2 signed angle is gauge dependent."""
    H = np.asarray(H, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] < 1:
        raise ValueError("H must be square")
    r = len(H)
    det = float(np.linalg.det(H))
    eig = np.linalg.eigvals(H)
    phases = np.sort(np.abs(np.rad2deg(np.angle(eig))))
    signed = None
    angle = None
    if r == 2 and det > 0:
        signed = float(np.rad2deg(np.arctan2(H[1, 0] - H[0, 1], H[0, 0] + H[1, 1])))
        angle = abs(signed)
    if r == 3 and det > 0:
        angle = float(np.rad2deg(np.arccos(np.clip((np.trace(H) - 1) / 2, -1, 1))))
    return {"rank": r, "determinant": det,
            "frobenius_distance": float(np.linalg.norm(H - np.eye(r), ord="fro")),
            "angle_degrees": angle, "signed_angle_degrees": signed,
            "eigenphases_degrees": phases.tolist()}


def analyze_connection(n_nodes: int, src: np.ndarray, dst: np.ndarray, entry: np.ndarray,
                       transports: np.ndarray, *, reliable: np.ndarray | None = None,
                       n_component_vertices: int | None = None, ps: np.ndarray | None = None,
                       epsilon: float = 0.05, max_recorded_cycles: int = 256) -> dict:
    """Full graph percolation + holonomy on its certified-edge subgraph.

    Entry values are fixed for a realization. A loop's birth is its final edge's
    entry. First nontrivial birth is an exact onset if reached before recording
    is capped: fundamental loop generators suffice even for non-Abelian groups.
    Uncertified edges remain in structural curves but are excluded from the
    holonomy filtration, avoiding false cycles through unreliable transports.
    """
    src, dst, entry = np.asarray(src), np.asarray(dst), np.asarray(entry, dtype=float)
    q = np.asarray(transports, dtype=float)
    if q.ndim != 3 or q.shape[0] != len(src) or q.shape[1] != q.shape[2] or q.shape[1] < 1:
        raise ValueError("one square rank>=1 transport matrix is required per edge")
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be nonnegative and finite")
    if reliable is None:
        reliable = np.ones(len(src), dtype=bool)
    reliable = np.asarray(reliable, dtype=bool)
    if reliable.shape != entry.shape:
        raise ValueError("reliable must match the edge array")
    full = cycle_filtration(n_nodes, src, dst, entry, ps,
                            n_component_vertices=n_component_vertices, max_recorded_cycles=0)
    accepted = np.flatnonzero(reliable)
    certified = cycle_filtration(n_nodes, src[accepted], dst[accepted], entry[accepted], full.ps,
                                  n_component_vertices=n_component_vertices,
                                  max_recorded_cycles=max_recorded_cycles)
    # Validate once, including unobserved edges, so the user gets an early error.
    if not np.isfinite(q).all() or not np.allclose(q.transpose(0, 2, 1) @ q, np.eye(q.shape[1]),
                                                   atol=1e-6, rtol=0):
        raise ValueError("transports must be finite orthogonal matrices")
    loops = []
    onset = None
    for cycle in certified.cycles:
        original = BornCycle(cycle.birth, cycle.vertices,
                             tuple((int(accepted[e]), sign) for e, sign in cycle.steps))
        H, intermediates = loop_matrix(q, original)
        sig = signature(H)
        record = {"birth": cycle.birth, "vertices": list(original.vertices),
                  "steps": [{"edge": e, "sign": s,
                             "matrix": (q[e] if s == 1 else q[e].T).tolist(),
                             "cumulative": m.tolist(),
                             "cumulative_signed_degrees": signature(m)["signed_angle_degrees"]}
                            for (e, s), m in zip(original.steps, intermediates)],
                  "matrix": H.tolist(), **sig,
                  "nontrivial": sig["frobenius_distance"] > epsilon}
        loops.append(record)
        if onset is None and record["nontrivial"]:
            onset = cycle.birth
    return {"schema": "percinterp.holonomy.v1", "epsilon": epsilon,
            "rank": int(q.shape[1]), "n_nodes": int(n_nodes), "n_edges": int(len(src)),
            "n_certified_edges": int(reliable.sum()),
            "n_component_vertices": int(n_nodes if n_component_vertices is None else n_component_vertices),
            "src": src.astype(int).tolist(), "dst": dst.astype(int).tolist(), "entry": entry.tolist(),
            "ps": full.ps.tolist(), "largest": full.largest.tolist(), "betti1": full.betti1.tolist(),
            "kept_edges": full.kept_edges.tolist(), "certified_betti1": certified.betti1.tolist(),
            "total_certified_cycles": certified.total_cycles, "cycles_recorded": len(loops),
            "truncated": certified.truncated, "p_h": onset,
            "p_h_known": onset is not None or not certified.truncated,
            "loops": loops}
