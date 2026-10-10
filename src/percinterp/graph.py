"""Layered directed acyclic graphs with weighted edges.

Nodes are numbered contiguously layer by layer. Every edge must point from an
earlier layer to a later one; skip edges (more than one layer) are allowed, so
residual-stream graphs fit the same structure.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LayeredDAG:
    layer_sizes: list[int]
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    sources: np.ndarray | None = None  # boolean mask over nodes; default: layer 0
    sinks: np.ndarray | None = None  # boolean mask over nodes; default: last layer
    offsets: np.ndarray = field(init=False)
    node_layer: np.ndarray = field(init=False)

    def __post_init__(self):
        self.layer_sizes = [int(s) for s in self.layer_sizes]
        self.offsets = np.concatenate([[0], np.cumsum(self.layer_sizes)]).astype(np.int64)
        self.node_layer = np.repeat(np.arange(len(self.layer_sizes)), self.layer_sizes)
        self.src = np.asarray(self.src, dtype=np.int64)
        self.dst = np.asarray(self.dst, dtype=np.int64)
        self.weight = np.asarray(self.weight, dtype=np.float64)
        if not (len(self.src) == len(self.dst) == len(self.weight)):
            raise ValueError("src, dst and weight must have the same length")
        if len(self.src) and np.any(self.node_layer[self.src] >= self.node_layer[self.dst]):
            raise ValueError("every edge must go from an earlier layer to a later one")
        if self.sources is None:
            self.sources = self.node_layer == 0
        if self.sinks is None:
            self.sinks = self.node_layer == len(self.layer_sizes) - 1
        self.sources = np.asarray(self.sources, dtype=bool)
        self.sinks = np.asarray(self.sinks, dtype=bool)

    @property
    def n_nodes(self) -> int:
        return int(self.offsets[-1])

    @property
    def n_edges(self) -> int:
        return len(self.src)

    @property
    def n_layers(self) -> int:
        return len(self.layer_sizes)

    def layer_nodes(self, layer: int) -> slice:
        return slice(int(self.offsets[layer]), int(self.offsets[layer + 1]))

    def edge_layer(self, by: str = "dst") -> np.ndarray:
        """Group label per edge, used for per-layer ranking and shuffling.

        by: "dst" (layer of the target), "src" (layer of the source) or
        "pair" (one group per (src layer, dst layer) combination).
        """
        if by == "dst":
            return self.node_layer[self.dst]
        if by == "src":
            return self.node_layer[self.src]
        if by == "pair":
            return self.node_layer[self.src] * self.n_layers + self.node_layer[self.dst]
        raise ValueError(f"unknown grouping {by!r}")

    def with_weight(self, weight: np.ndarray) -> "LayeredDAG":
        """Same topology, new weights. Shares cached edge orderings with self."""
        weight = np.asarray(weight, dtype=np.float64)
        if len(weight) != self.n_edges:
            raise ValueError("weight length does not match the edge count")
        new = copy.copy(self)
        new.weight = weight
        return new

    @classmethod
    def from_dense(cls, mats: list[np.ndarray], drop_zeros: bool = True) -> "LayeredDAG":
        """Build an adjacent-layer graph from weight matrices.

        mats[l] has shape (n_{l+1}, n_l), the usual (out, in) layout, and its
        entry [i, j] is the weight of the edge from node j in layer l to node i
        in layer l+1. Exact zeros are dropped by default: a zero attribution is
        no influence, so it is not an edge.
        """
        sizes = [mats[0].shape[1]] + [m.shape[0] for m in mats]
        for l, m in enumerate(mats):
            if m.shape[1] != sizes[l]:
                raise ValueError(f"matrix {l} has {m.shape[1]} inputs, expected {sizes[l]}")
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        srcs, dsts, ws = [], [], []
        for l, m in enumerate(mats):
            i, j = np.nonzero(m) if drop_zeros else np.indices(m.shape).reshape(2, -1)
            srcs.append(j + offsets[l])
            dsts.append(i + offsets[l + 1])
            ws.append(m[i, j])
        return cls(sizes, np.concatenate(srcs), np.concatenate(dsts), np.concatenate(ws))
