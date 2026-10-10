"""Batched percolation measurements for adjacent-layer graphs stored as dense tensors.

The same quantities as measure.Profile, for a batch of graphs that share a
layer structure: E[l] has shape (B, n_{l+1}, n_l) and holds entry keys (edge
present at p when key <= p; +inf = not an edge). The forward and backward
passes become min-max matrix-vector products, so a batch of per-input graphs
is processed in one go, on GPU if available.
"""

from __future__ import annotations

import torch


def rank_entry(scores: list[torch.Tensor]) -> list[torch.Tensor]:
    """Per graph and per layer: fraction of the layer's nonzero edges scoring >= this one.

    Zero scores are not edges (+inf). Matches measure.rank_entry with one group
    per layer, on the skeleton of nonzero edges.
    """
    out = []
    for S in scores:
        B = S.shape[0]
        flat = S.reshape(B, -1)
        nz = flat > 0
        order = torch.argsort(flat, dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(flat.shape[1], device=S.device).expand(B, -1).contiguous())
        q = (ranks + 1).to(torch.float32) / nz.sum(1, keepdim=True).clamp(min=1)
        out.append(torch.where(nz, q, torch.full_like(q, torch.inf)).reshape(S.shape))
    return out


def shuffled_entry(scores: list[torch.Tensor], generator: torch.Generator | None = None) -> list[torch.Tensor]:
    """The within-layer shuffle null: a uniformly random ranking of each layer's nonzero edges.

    The generator, if given, must live on the same device as the scores.
    """
    return rank_entry(
        [torch.rand(S.shape, device=S.device, generator=generator) * (S > 0) for S in scores]
    )


def _frac_le(values: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
    """(B, len(ps)) fraction of each row's values that are <= p."""
    srt = torch.sort(values, dim=1)[0].contiguous()
    counts = torch.searchsorted(srt, ps.to(srt.dtype).expand(srt.shape[0], -1).contiguous(), right=True)
    return counts.to(torch.float32) / values.shape[1]


def reference_entry(scores: list[torch.Tensor], ref: list[torch.Tensor]) -> list[torch.Tensor]:
    """Absolute-threshold keys: per layer, the fraction of reference scores >= this score.

    ref[l] is a 1-D tensor of reference scores for layer l (e.g. pooled from the
    final checkpoint). A fixed key threshold is then a fixed absolute score
    threshold per layer, comparable across checkpoints.
    """
    out = []
    for S, r in zip(scores, ref):
        r = torch.sort(r.flatten())[0]
        n_ge = len(r) - torch.searchsorted(r, S.contiguous(), right=False)
        q = n_ge.to(torch.float32) / len(r)
        out.append(torch.where(S > 0, q, torch.full_like(q, torch.inf)))
    return out


def forward_entry(E: list[torch.Tensor]) -> list[torch.Tensor]:
    """F per node layer: smallest p at which the node is reachable from layer 0 (all sources)."""
    B = E[0].shape[0]
    F = [torch.full((B, E[0].shape[2]), -torch.inf, device=E[0].device)]
    for El in E:
        F.append(torch.maximum(F[-1][:, None, :], El).amin(dim=2))
    return F


def backward_entry(E: list[torch.Tensor]) -> list[torch.Tensor]:
    """B per node layer: smallest p at which the node reaches the last layer (all sinks)."""
    Bsz = E[-1].shape[0]
    Bk = [torch.full((Bsz, E[-1].shape[1]), -torch.inf, device=E[-1].device)]
    for El in reversed(E):
        Bk.append(torch.maximum(El, Bk[-1][:, :, None]).amin(dim=1))
    return Bk[::-1]


class DenseProfile:
    def __init__(self, E: list[torch.Tensor]):
        self.E = E
        self.F = forward_entry(E)
        self.Bk = backward_entry(E)

    @property
    def p_span(self) -> torch.Tensor:
        """(B,) smallest p with a source-to-sink path."""
        return self.F[-1].amin(dim=1)

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
