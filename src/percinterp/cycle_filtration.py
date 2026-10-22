"""Nested undirected bond percolation and fundamental cycle births.

This is deliberately independent of LayeredDAG: VPD component/neuron contact
graphs are bipartite and undirected, whereas influence graphs are DAGs.
Edges with entry <= p are retained, matching percinterp.measure semantics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BornCycle:
    """One fundamental loop: (edge index, traversal sign) in walk order."""

    birth: float
    vertices: tuple[int, ...]  # starts and ends at the same vertex
    steps: tuple[tuple[int, int], ...]  # +1 src->dst, -1 dst->src


@dataclass
class CycleFiltration:
    ps: np.ndarray
    largest: np.ndarray  # fraction of designated component vertices, not all nodes
    betti1: np.ndarray  # cycle rank |E_p| - |V| + connected_components
    kept_edges: np.ndarray
    cycles: list[BornCycle]
    total_cycles: int
    truncated: bool


def cycle_filtration(
    n_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    entry: np.ndarray,
    ps: np.ndarray | None = None,
    *,
    n_component_vertices: int | None = None,
    max_recorded_cycles: int = 256,
) -> CycleFiltration:
    """One stable Kruskal sweep; record the fundamental loops of a spanning forest.

    A non-tree edge closes one cycle, with birth equal to its entry key. Loops
    depend on the stable edge ordering for ties; Betti numbers and first
    nontrivial-holonomy onset (when all certified cycles are assessed) do not.
    Parallel edges are supported as length-two loops. Self edges are rejected.
    Recording can be capped without changing any aggregate percolation curves.
    """
    if not isinstance(n_nodes, (int, np.integer)) or n_nodes < 1:
        raise ValueError("n_nodes must be positive")
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    entry = np.asarray(entry, dtype=float)
    if src.ndim != 1 or dst.shape != src.shape or entry.shape != src.shape:
        raise ValueError("src, dst, entry must be equal-length 1-D arrays")
    if np.any(src < 0) or np.any(src >= n_nodes) or np.any(dst < 0) or np.any(dst >= n_nodes):
        raise ValueError("edge endpoint outside node range")
    if np.any(src == dst):
        raise ValueError("self-edges are not supported")
    if not np.isfinite(entry).all() or np.any((entry < 0) | (entry > 1)):
        raise ValueError("entry keys must be finite and in [0, 1]")
    if n_component_vertices is None:
        n_component_vertices = n_nodes
    if not 1 <= n_component_vertices <= n_nodes:
        raise ValueError("invalid n_component_vertices")
    if max_recorded_cycles < 0:
        raise ValueError("max_recorded_cycles must be nonnegative")
    if ps is None:
        ps = np.linspace(0.0, 1.0, 41)
    ps = np.asarray(ps, dtype=float)
    if ps.ndim != 1 or not np.isfinite(ps).all() or np.any((ps < 0) | (ps > 1)):
        raise ValueError("ps must be a 1-D array within [0, 1]")

    order = np.argsort(entry, kind="stable")
    parent = np.arange(n_nodes)
    size = np.ones(n_nodes, dtype=np.int64)
    comp_size = (np.arange(n_nodes) < n_component_vertices).astype(np.int64)
    forest: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
    max_comp = 1
    b1 = 0
    largest = [max_comp / n_component_vertices]
    betti = [0]
    born: list[BornCycle] = []

    def root(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = int(parent[v])
        return v

    def forest_path(start: int, goal: int) -> tuple[list[int], list[tuple[int, int]]]:
        # The accepted spanning edges form a tree inside each DSU component.
        pending = [start]
        previous: dict[int, tuple[int, int]] = {start: (-1, -1)}
        while pending and goal not in previous:
            u = pending.pop()
            for v, edge in forest[u]:
                if v not in previous:
                    previous[v] = (u, edge)
                    pending.append(v)
        vertices = [goal]
        steps = []
        while vertices[-1] != start:
            v = vertices[-1]
            u, edge = previous[v]
            steps.append((edge, +1 if src[edge] == u else -1))
            vertices.append(u)
        vertices.reverse()
        steps.reverse()
        return vertices, steps

    for edge in order:
        edge = int(edge)
        u, v = int(src[edge]), int(dst[edge])
        ru, rv = root(u), root(v)
        if ru != rv:
            if size[ru] < size[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            size[ru] += size[rv]
            comp_size[ru] += comp_size[rv]
            max_comp = max(max_comp, int(comp_size[ru]))
            forest[u].append((v, edge))
            forest[v].append((u, edge))
        else:
            b1 += 1
            if len(born) < max_recorded_cycles:
                vertices, steps = forest_path(u, v)
                born.append(BornCycle(float(entry[edge]), tuple(vertices + [u]),
                                      tuple(steps + [(edge, -1)])))
        largest.append(max_comp / n_component_vertices)
        betti.append(b1)

    counts = np.searchsorted(entry[order], ps, side="right")
    return CycleFiltration(
        ps=ps, largest=np.asarray(largest)[counts],
        betti1=np.asarray(betti, dtype=int)[counts], kept_edges=counts,
        cycles=born, total_cycles=b1, truncated=len(born) < b1,
    )
