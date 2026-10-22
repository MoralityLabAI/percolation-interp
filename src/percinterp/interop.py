"""Small, dependency-light interchange boundary for neural influence graphs.

Input rows describe *direct* attribution edges. ``LayeredDAG`` uses nonnegative
magnitudes for graph filtration; this adapter preserves the original signed
values alongside them. The graph is a diagnostic and makes no causal or
faithfulness claim without separate validation on the originating model.

Identifiers are plain strings. A producer using Goodfire's ``ComponentKey`` can
pass ``key.encoded()``, avoiding a JAX, torch, or pydantic runtime dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

import numpy as np

from .graph import LayeredDAG


@dataclass(frozen=True)
class ImportedGraph:
    """Internal layered graph with an exact mapping back to external node IDs.

    ``node_ids[i]`` names internal vertex ``i``. ``signed_weight[j]`` records
    the signed weight of internal edge ``j``; ``graph.weight[j]`` is its absolute
    value. Zero-weight edges are discarded at the import boundary.
    """

    graph: LayeredDAG
    node_ids: tuple[str, ...]
    signed_weight: np.ndarray
    dropped_zero_edges: int

    def selected_ids(self, mask: Sequence[bool]) -> tuple[str, ...]:
        """Translate a boolean graph-node mask back to external identifiers."""
        selected = np.asarray(mask)
        if selected.dtype != np.dtype(bool) or selected.shape != (self.graph.n_nodes,):
            raise ValueError("mask must be a boolean vector of length n_nodes")
        return tuple(node for node, keep in zip(self.node_ids, selected) if keep)

    def signed_edges(self) -> tuple[tuple[str, str, float], ...]:
        """Return nonzero directed rows in internal edge order."""
        return tuple(
            (self.node_ids[int(src)], self.node_ids[int(dst)], float(weight))
            for src, dst, weight in zip(
                self.graph.src, self.graph.dst, self.signed_weight
            )
        )


def from_attribution_edges(
    node_ids: Sequence[str],
    layers: Sequence[int],
    edges: Iterable[tuple[str, str, float]],
    *,
    sources: Sequence[str] | None = None,
    sinks: Sequence[str] | None = None,
) -> ImportedGraph:
    """Import a topologically ordered influence graph from stable node IDs.

    Parameters
    ----------
    node_ids:
        Unique, nonempty strings. For a VPD subcomponent, an upstream consumer
        may supply a ``param_decomp_formats.ComponentKey.encoded()`` value.
        Tokens and output metrics can use other explicitly namespaced IDs.
    layers:
        A zero-based *topological stage* per node, not necessarily the raw
        transformer block number. Stages must be nonempty and consecutive.
    edges:
        Directed (source ID, target ID, signed direct-attribution weight) rows.
        Skip-layer edges are supported. Parallel rows are rejected so that the
        caller must choose an explicit aggregation rule before import.
        Exact zero weights are dropped. Negative weights retain their sign in
        ``ImportedGraph.signed_weight``, while ``LayeredDAG.weight`` is |weight|.
    sources, sinks:
        Optional designated input and output node IDs. The defaults are all
        nodes in the first and last stages. Explicit sets must be nonempty.

    Returns
    -------
    ImportedGraph
        A graph suitable for ``percinterp.profile`` / ``percinterp.sieve``
        and a stable mapping from selected node indices to external identifiers.

    Notes
    -----
    This function imports attribution *data*. It does not compute causal
    importances, certify a parameter decomposition, infer an influence graph,
    or establish mechanistic fidelity of a pruned circuit.
    """
    original_ids = tuple(node_ids)
    raw_layers = tuple(layers)
    if not original_ids or any(not isinstance(s, str) or not s for s in original_ids):
        raise ValueError("node_ids must be nonempty, nonblank strings")
    if len(set(original_ids)) != len(original_ids):
        raise ValueError("node_ids must be unique")
    if len(raw_layers) != len(original_ids):
        raise ValueError("layers and node_ids must have the same length")
    if any(isinstance(layer, bool) or not isinstance(layer, (int, np.integer))
           or layer < 0 for layer in raw_layers):
        raise ValueError("layers must be nonnegative integer topological stages")
    stages = tuple(int(layer) for layer in raw_layers)
    unique_stages = sorted(set(stages))
    if len(unique_stages) < 2 or unique_stages != list(range(len(unique_stages))):
        raise ValueError("need at least two nonempty, consecutive stages from zero")
    counts = np.bincount(np.asarray(stages, dtype=np.int64))

    order = sorted(range(len(original_ids)), key=lambda i: (stages[i], i))
    ordered_ids = tuple(original_ids[i] for i in order)
    node_stages = tuple(stages[i] for i in order)
    index = {node: i for i, node in enumerate(ordered_ids)}

    def boundary(ids: Sequence[str] | None, stage: int, label: str) -> np.ndarray:
        if ids is None:
            resolved = set(n for n, s in zip(ordered_ids, node_stages) if s == stage)
        else:
            if isinstance(ids, str):
                raise ValueError(f"{label} must be a sequence of node IDs")
            provided = tuple(ids)
            if len(set(provided)) != len(provided):
                raise ValueError(f"{label} contains duplicate node IDs")
            resolved = set(provided)
            if not resolved.issubset(index):
                raise ValueError(f"{label} contains unknown node IDs")
        if not resolved:
            raise ValueError(f"{label} must contain at least one node")
        return np.array([name in resolved for name in ordered_ids], dtype=bool)

    source_mask = boundary(sources, 0, "sources")
    sink_mask = boundary(sinks, len(counts) - 1, "sinks")
    if np.any(source_mask & sink_mask):
        raise ValueError("sources and sinks must be disjoint")
    src: list[int] = []
    dst: list[int] = []
    magnitude: list[float] = []
    signed: list[float] = []
    seen: set[tuple[int, int]] = set()
    zeros = 0
    for row in edges:
        try:
            from_id, to_id, value = row
        except (TypeError, ValueError) as exc:
            raise ValueError("each edge must contain (source, target, weight)") from exc
        if from_id not in index or to_id not in index:
            raise ValueError("edge references an unknown node ID")
        a, b = index[from_id], index[to_id]
        if node_stages[a] >= node_stages[b]:
            raise ValueError("edges must point from earlier to later stages")
        if (a, b) in seen:
            raise ValueError("duplicate directed edge: aggregate explicitly upstream")
        seen.add((a, b))
        try:
            weight = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("edge weights must be finite numbers") from exc
        if not math.isfinite(weight):
            raise ValueError("edge weights must be finite numbers")
        if weight == 0:
            zeros += 1
            continue
        src.append(a)
        dst.append(b)
        magnitude.append(abs(weight))
        signed.append(weight)

    graph = LayeredDAG(
        list(map(int, counts)),
        np.asarray(src, dtype=np.int64),
        np.asarray(dst, dtype=np.int64),
        np.asarray(magnitude, dtype=np.float64),
        sources=source_mask,
        sinks=sink_mask,
    )
    return ImportedGraph(
        graph=graph,
        node_ids=ordered_ids,
        signed_weight=np.asarray(signed, dtype=np.float64),
        dropped_zero_edges=zeros,
    )
