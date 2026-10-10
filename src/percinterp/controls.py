"""Graphs with known or controllable structure: random graphs, MLPs at init, shuffles."""

from __future__ import annotations

import numpy as np

from .graph import LayeredDAG


def complete_layered(widths: list[int], rng: np.random.Generator) -> LayeredDAG:
    """Every node joined to every node in the next layer, i.i.d. uniform weights.

    Using the weights directly as entry keys gives Bernoulli bond percolation;
    ranking them gives the fixed-edge-count version.
    """
    offsets = np.concatenate([[0], np.cumsum(widths)])
    srcs, dsts = [], []
    for l in range(len(widths) - 1):
        i, j = np.indices((widths[l + 1], widths[l])).reshape(2, -1)
        srcs.append(j + offsets[l])
        dsts.append(i + offsets[l + 1])
    src, dst = np.concatenate(srcs), np.concatenate(dsts)
    return LayeredDAG(list(widths), src, dst, rng.random(len(src)))


def permute_within(weight: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle weights among edges sharing a group label (layer, source node, ...).

    Within-layer shuffling on a fixed skeleton turns any ranking into plain
    bond percolation on that skeleton: the null every trained graph is
    compared against.
    """
    out = np.array(weight, dtype=np.float64, copy=True)
    order = np.argsort(groups, kind="stable")
    bounds = np.flatnonzero(np.diff(groups[order])) + 1
    for block in np.split(order, bounds):
        out[block] = out[rng.permutation(block)]
    return out


# --- ReLU MLPs ---------------------------------------------------------------
# Ws[l] maps layer l to layer l+1 with shape (n_{l+1}, n_l). acts[0] is the
# input; hidden layers are ReLU; the last layer is linear and the task metric
# is readout . output.


def init_mlp(widths: list[int], rng: np.random.Generator, gain: float = 2.0) -> list[np.ndarray]:
    """Gaussian init with variance gain / fan_in (gain 2 = Kaiming for ReLU)."""
    return [
        rng.normal(0.0, np.sqrt(gain / widths[l]), size=(widths[l + 1], widths[l]))
        for l in range(len(widths) - 1)
    ]


def mlp_forward(Ws: list[np.ndarray], x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Activations and pre-activations for one input. pre[0] is unused (None)."""
    acts, pre = [np.asarray(x, dtype=np.float64)], [None]
    for l, W in enumerate(Ws):
        z = W @ acts[-1]
        pre.append(z)
        acts.append(z if l == len(Ws) - 1 else np.maximum(z, 0.0))
    return acts, pre


def mlp_grads(Ws: list[np.ndarray], pre: list[np.ndarray], readout: np.ndarray) -> list[np.ndarray]:
    """d metric / d pre-activation for layers 1..L (index 0 unused)."""
    L = len(Ws)
    grads: list[np.ndarray | None] = [None] * (L + 1)
    grads[L] = np.asarray(readout, dtype=np.float64)
    for l in range(L - 1, 0, -1):
        grads[l] = (Ws[l].T @ grads[l + 1]) * (pre[l] > 0)
    return grads


def mlp_edge_matrices(
    Ws: list[np.ndarray],
    x: np.ndarray | None = None,
    kind: str = "weight",
    readout: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Edge-weight magnitudes of an MLP's influence graph, one (out, in) matrix per layer.

    kind="weight":  |W_ij|                 (the wiring; input independent)
    kind="direct":  |W_ij a_j|             (direct effect of j on i's pre-activation)
    kind="eap":     |a_j W_ij g_i|         (edge attribution patching against a zero
                                            baseline; g_i = d metric / d z_i)
    """
    if kind == "weight":
        return [np.abs(W) for W in Ws]
    acts, pre = mlp_forward(Ws, x)
    if kind == "direct":
        return [np.abs(W * acts[l][None, :]) for l, W in enumerate(Ws)]
    if kind == "eap":
        if readout is None:
            readout = np.ones(Ws[-1].shape[0])
        grads = mlp_grads(Ws, pre, readout)
        return [np.abs(grads[l + 1][:, None] * W * acts[l][None, :]) for l, W in enumerate(Ws)]
    raise ValueError(f"unknown kind {kind!r}")


def mlp_graph(Ws, x=None, kind="weight", readout=None) -> LayeredDAG:
    """Influence graph of an MLP (see mlp_edge_matrices). Dead-ReLU zeros are non-edges."""
    return LayeredDAG.from_dense(mlp_edge_matrices(Ws, x, kind, readout))
