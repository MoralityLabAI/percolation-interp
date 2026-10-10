"""Batched percolation measurements for adjacent-layer graphs stored as dense tensors.

The same quantities as measure.Profile, for a batch of graphs that share a
layer structure: E[l] has shape (B, n_{l+1}, n_l) and holds entry keys (edge
present at p when key <= p; +inf = not an edge). The forward and backward
passes become min-max matrix-vector products, so a batch of per-input graphs
is processed in one go, on GPU if available.
"""

from __future__ import annotations

import torch


def _ranks(S: torch.Tensor) -> torch.Tensor:
    """1-based descending rank of each edge within its layer, per graph (zeros rank last)."""
    B = S.shape[0]
    flat = S.reshape(B, -1)
    order = torch.argsort(flat, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(flat.shape[1], device=S.device).expand(B, -1).contiguous())
    return (ranks + 1).to(torch.float32).reshape(S.shape)


def _scale(S: torch.Tensor, unit: str) -> torch.Tensor:
    """(B, 1, 1) divisor turning ranks into occupation units.

    "fraction": nonzero edges in the layer, so keys are the kept fraction p.
    "degree":   live source nodes (any nonzero out-edge), so keys are c, the mean
                kept out-degree per live source. Comparable across layers of
                different widths; equal to pN for complete N -> N layers.
    """
    nz = S > 0
    if unit == "fraction":
        n = nz.flatten(1).sum(1)
    elif unit == "degree":
        n = nz.any(dim=1).sum(1)
    else:
        raise ValueError(f"unknown unit {unit!r}")
    return n.clamp(min=1).to(torch.float32)[:, None, None]


def rank_entry(scores: list[torch.Tensor], unit: str = "fraction") -> list[torch.Tensor]:
    """Keys from per-layer ranks: edge kept at occupation t when key <= t; zeros are not edges.

    With unit="fraction" this matches measure.rank_entry with one group per
    layer, on the skeleton of nonzero edges.
    """
    return [torch.where(S > 0, _ranks(S) / _scale(S, unit), torch.inf) for S in scores]


def shuffled_entry(scores: list[torch.Tensor], generator: torch.Generator | None = None,
                   unit: str = "fraction") -> list[torch.Tensor]:
    """The within-layer shuffle null: a uniformly random ranking of each layer's nonzero edges.

    The generator, if given, must live on the same device as the scores.
    """
    out = []
    for S in scores:
        u = torch.rand(S.shape, device=S.device, generator=generator) * (S > 0)
        out.append(torch.where(S > 0, _ranks(u) / _scale(S, unit), torch.inf))
    return out


def _frac_le(values: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
    """(B, len(ps)) fraction of each row's values that are <= p."""
    srt = torch.sort(values, dim=1)[0].contiguous()
    counts = torch.searchsorted(srt, ps.to(srt.dtype).expand(srt.shape[0], -1).contiguous(), right=True)
    return counts.to(torch.float32) / values.shape[1]


def reference_entry(scores: list[torch.Tensor], ref: list[torch.Tensor],
                    ref_scale: list[float] | None = None) -> list[torch.Tensor]:
    """Absolute-threshold keys: per layer, the fraction of reference scores >= this score.

    ref[l] is a 1-D tensor of reference scores for layer l (e.g. pooled from the
    final checkpoint), so a fixed key is a fixed absolute score threshold per
    layer, comparable across checkpoints. ref_scale[l] multiplies the fraction:
    pass the reference's mean nonzero edges per live source for degree units.
    """
    out = []
    for l, (S, r) in enumerate(zip(scores, ref)):
        r = torch.sort(r.flatten())[0]
        n_ge = len(r) - torch.searchsorted(r, S.contiguous(), right=False)
        q = n_ge.to(torch.float32) / len(r)
        if ref_scale is not None:
            q = q * ref_scale[l]
        out.append(torch.where(S > 0, q, torch.full_like(q, torch.inf)))
    return out


def forward_entry(E: list[torch.Tensor]) -> list[torch.Tensor]:
    """F per node layer: smallest p at which the node is reachable from layer 0 (all sources)."""
    B = E[0].shape[0]
    F = [torch.full((B, E[0].shape[2]), -torch.inf, device=E[0].device)]
    for El in E:
        F.append(torch.maximum(F[-1][:, None, :], El).amin(dim=2))
    return F


def backward_entry(E: list[torch.Tensor], sinks: torch.Tensor | None = None) -> list[torch.Tensor]:
    """B per node layer: smallest p at which the node reaches a sink.

    sinks: (B, n_last) boolean, per graph; default every last-layer node.
    """
    Bsz, n_last = E[-1].shape[0], E[-1].shape[1]
    if sinks is None:
        sinks = torch.ones(Bsz, n_last, dtype=torch.bool, device=E[-1].device)
    Bk = [torch.where(sinks, -torch.inf, torch.inf).to(E[-1].dtype)]
    for El in reversed(E):
        Bk.append(torch.maximum(El, Bk[-1][:, :, None]).amin(dim=1))
    return Bk[::-1]


class DenseProfile:
    def __init__(self, E: list[torch.Tensor], sinks: torch.Tensor | None = None):
        """sinks: (B, n_last) boolean per graph; default every last-layer node."""
        self.E = E
        self.F = forward_entry(E)
        self.Bk = backward_entry(E, sinks)
        self.sinks = sinks

    @property
    def p_span(self) -> torch.Tensor:
        """(B,) smallest p with a source-to-sink path."""
        F = self.F[-1]
        if self.sinks is not None:
            F = torch.where(self.sinks, F, torch.full_like(F, torch.inf))
        return F.amin(dim=1)

    def node_entry(self, layer: int) -> torch.Tensor:
        return torch.maximum(self.F[layer], self.Bk[layer])

    def edge_backbone_entry(self, l: int) -> torch.Tensor:
        return torch.maximum(torch.maximum(self.F[l][:, None, :], self.E[l]), self.Bk[l + 1][:, :, None])

    def backbone_node_fraction(self, ps: torch.Tensor, layers: list[int] | None = None) -> torch.Tensor:
        """(B, len(ps)) fraction of nodes in the given layers (default: hidden) on a surviving path."""
        layers = list(range(1, len(self.F) - 1)) if layers is None else layers
        return _frac_le(torch.cat([self.node_entry(l) for l in layers], dim=1), ps)

    def backbone_edge_fraction(self, ps: torch.Tensor) -> torch.Tensor:
        """(B, len(ps)) fraction of the skeleton's edges that are kept and on a surviving path."""
        be = torch.cat([self.edge_backbone_entry(l).flatten(1) for l in range(len(self.E))], dim=1)
        n_edges = torch.cat([torch.isfinite(El).flatten(1) for El in self.E], dim=1).sum(1)
        return _frac_le(be, ps) * be.shape[1] / n_edges[:, None].clamp(min=1)

    def reach_fraction(self, ps: torch.Tensor, layer: int) -> torch.Tensor:
        return _frac_le(self.F[layer], ps)

    def masks(self, p: float, backbone_only: bool = False) -> list[torch.Tensor]:
        """Edge masks (1 = kept) at occupation p, optionally restricted to the backbone."""
        if backbone_only:
            return [(self.edge_backbone_entry(l) <= p).float() for l in range(len(self.E))]
        return [(El <= p).float() for El in self.E]
