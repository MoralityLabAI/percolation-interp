"""TopK sparse autoencoders for a tower over a model's residual stream.

Per-token TopK (Gao et al. 2024): each token keeps its k largest encoder
pre-activations (after ReLU), so "feature i fires on this token" means "i is
in this token's top k with a positive value". Dead latents get an auxiliary
TopK loss on the reconstruction error (AuxK). Inputs are scaled by a fixed
scalar so their mean norm is sqrt(d); decode() returns the original scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d: int, n_latents: int, k: int, k_aux: int = 256, dead_tokens: int = 2_000_000):
        super().__init__()
        self.d, self.n, self.k, self.k_aux, self.dead_tokens = d, n_latents, k, k_aux, dead_tokens
        W = torch.randn(d, n_latents)
        W = W / W.norm(dim=0, keepdim=True)
        self.W_enc = nn.Parameter(W.clone())
        self.W_dec = nn.Parameter(W.T.clone())  # (n, d), unit rows
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.register_buffer("scale", torch.tensor(0.0))
        self.register_buffer("tokens_since_fired", torch.zeros(n_latents, dtype=torch.long))

    @torch.no_grad()
    def set_scale(self, x: torch.Tensor):
        """Fix the input scale so the mean token norm becomes sqrt(d); call once on a sample."""
        self.scale.fill_((self.d**0.5) / x.norm(dim=-1).mean())

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.scale - self.b_dec) @ self.W_enc + self.b_enc

    def topk(self, pre: torch.Tensor) -> torch.Tensor:
        """Keep each token's top k pre-activations (ReLU'd); everything else is 0."""
        vals, idx = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, idx, F.relu(vals))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.topk(self.pre_acts(x))

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return (a @ self.W_dec + self.b_dec) / self.scale

    def forward(self, x: torch.Tensor):
        pre = self.pre_acts(x)
        a = self.topk(pre)
        return self.decode(a), a, pre

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Normalised MSE plus AuxK on dead latents. x: (tokens, d)."""
        xs = x * self.scale
        pre = (xs - self.b_dec) @ self.W_enc + self.b_enc
        a = self.topk(pre)
        recon = a @ self.W_dec + self.b_dec
        err = xs - recon
        var = (xs - xs.mean(0)).pow(2).sum(-1).mean()
        mse = err.pow(2).sum(-1).mean() / var
        with torch.no_grad():
            fired = (a > 0).any(0)
            self.tokens_since_fired += x.shape[0]
            self.tokens_since_fired[fired] = 0
            dead = self.tokens_since_fired > self.dead_tokens
        aux = torch.zeros((), device=x.device)
        n_dead = int(dead.sum())
        if n_dead:
            k_aux = min(self.k_aux, n_dead)
            pre_dead = pre.masked_fill(~dead, float("-inf"))
            vals, idx = pre_dead.topk(k_aux, dim=-1)
            a_aux = torch.zeros_like(pre).scatter(-1, idx, F.relu(vals))
            aux_recon = a_aux @ self.W_dec
            aux = (err.detach() - aux_recon).pow(2).sum(-1).mean() / var
        return mse + aux / 32, {"fvu": mse.item(), "aux": aux.item(), "dead_frac": n_dead / self.n}

    @torch.no_grad()
    def normalize_decoder(self):
        """Keep decoder rows at unit norm (call after each optimiser step)."""
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

    @torch.no_grad()
    def remove_parallel_grad(self):
        """Project out the gradient component that would change decoder row norms."""
        if self.W_dec.grad is not None:
            W = self.W_dec.data
            self.W_dec.grad -= (self.W_dec.grad * W).sum(1, keepdim=True) * W

    def state(self) -> dict:
        return {"config": {"d": self.d, "n_latents": self.n, "k": self.k, "k_aux": self.k_aux,
                           "dead_tokens": self.dead_tokens},
                "state_dict": self.state_dict()}

    @classmethod
    def from_state(cls, blob: dict) -> "TopKSAE":
        sae = cls(**blob["config"])
        sae.load_state_dict(blob["state_dict"])
        return sae
