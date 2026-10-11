"""Modular addition with deep ReLU MLPs: data, training, edge attribution, subgraph fidelity.

Influence graph for one input (a, b) with answer c = a + b mod P:

    layer 0      2 token nodes, a and b
    layers 1..D  hidden ReLU neurons, width N
    layer D+1    one sink: the task metric logit_c - mean(logits)

Edges are mean-ablation edges. Token node a sends u_a = W[:, a-block] (x_a - 1/P),
its embedding minus the mean embedding, into the first layer. Hidden edge
(j -> i) carries W_ij * (h_j - mean h_j), where the mean is the clean model's
mean activation over all P^2 inputs. Neuron j's edge to the sink is its whole
write to the logits, W[:, j] * (h_j - mean h_j). Removing an edge replaces the
source by its mean on that edge only, so masks interpolate between the full
model (all ones) and the fully mean-ablated model (all zeros), and the same
masked forward pass scores subgraph fidelity.

Why token nodes rather than 2P one-hot units, and one sink rather than P
logits: those finer graphs carry trivial structure (the two active units score
~P times the 2P-2 inactive ones; edges into the answer logit ~P times the
others) that a within-layer shuffle destroys, so the null would credit the
input encoding and the metric's definition to the network.

Edge scores follow EAP-IG: |dh_j * W_ij * gbar_i|, where gbar_i is
d metric / d pre_i averaged over inputs interpolated from the mean input (uniform
1/P one-hot) to the clean input. steps=1 evaluates the gradient at the clean
input only, which is plain EAP.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    P: int = 113
    depth: int = 2  # hidden layers
    width: int = 256
    frac_train: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1.0
    beta2: float = 0.98
    steps: int = 20000
    seed: int = 0
    n_log_ckpts: int = 40  # checkpoints log-spaced over training, plus step 0 and the last
    n_lin_ckpts: int = 40  # and evenly spaced ones
    init_scale: float = 1.0  # multiplies every parameter after init (Omnigrok's alpha)
    init: str = "torch"  # "torch": nn.Linear default; "kaiming": He normal for ReLU, zero biases

    @property
    def name(self) -> str:
        """Run name; hyperparameters that differ from the step 2 defaults are appended."""
        extra = ""
        if self.lr != 1e-3:
            extra += f"_lr{self.lr:g}"
        if self.frac_train != 0.3:
            extra += f"_f{self.frac_train:g}"
        if self.weight_decay != 1.0:
            extra += f"_wd{self.weight_decay:g}"
        if self.init_scale != 1.0:
            extra += f"_i{self.init_scale:g}"
        if self.init != "torch":
            extra += f"_{self.init}"
        return f"P{self.P}_D{self.depth}_N{self.width}_s{self.seed}{extra}"


class MLP(nn.Module):
    def __init__(self, P: int, depth: int, width: int):
        super().__init__()
        sizes = [2 * P] + [width] * depth + [P]
        self.layers = nn.ModuleList(nn.Linear(sizes[l], sizes[l + 1]) for l in range(len(sizes) - 1))
        self.P = P

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for l, lin in enumerate(self.layers):
            h = lin(h)
            if l < len(self.layers) - 1:
                h = F.relu(h)
        return h

    def hidden(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Activations of every node layer: [input, hidden_1, ..., hidden_D]."""
        hs, h = [x], x
        for lin in self.layers[:-1]:
            h = F.relu(lin(h))
            hs.append(h)
        return hs


def one_hot_pairs(a: torch.Tensor, b: torch.Tensor, P: int, dtype=torch.float32) -> torch.Tensor:
    x = torch.zeros(len(a), 2 * P, device=a.device, dtype=dtype)
    x[torch.arange(len(a)), a] = 1.0
    x[torch.arange(len(a)), P + b] = 1.0
    return x


def all_pairs(P: int, device="cpu"):
    a = torch.arange(P, device=device).repeat_interleave(P)
    b = torch.arange(P, device=device).repeat(P)
    return a, b, (a + b) % P


def split(P: int, frac_train: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(P * P, generator=g)
    n_train = int(frac_train * P * P)
    return perm[:n_train], perm[n_train:]


def checkpoint_steps(cfg: Config) -> list[int]:
    log = torch.logspace(0, math.log10(cfg.steps), cfg.n_log_ckpts).round().long().tolist()
    lin = torch.linspace(0, cfg.steps, cfg.n_lin_ckpts + 1).round().long().tolist()
    return sorted(set([0] + log + lin))


def train(cfg: Config, out_dir: Path, device: str = "cuda", log_every: int = 100, progress: int = 0) -> dict:
    """Full-batch AdamW training. Saves every checkpoint's state dict into one file.

    progress > 0 prints a status line every that many steps.
    """
    torch.manual_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b, c = all_pairs(cfg.P, device)
    x = one_hot_pairs(a, b, cfg.P)
    tr, te = split(cfg.P, cfg.frac_train, cfg.seed)
    tr, te = tr.to(device), te.to(device)
    if device == "cpu":
        # tiny activations in deep nets otherwise hit denormal floats, which are very slow on CPU
        torch.set_flush_denormal(True)
    model = MLP(cfg.P, cfg.depth, cfg.width).to(device)
    with torch.no_grad():
        if cfg.init == "kaiming":
            # nn.Linear's default shrinks a ReLU net's signal variance about 6x per layer
            for lin in model.layers:
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
                nn.init.zeros_(lin.bias)
        elif cfg.init != "torch":
            raise ValueError(f"unknown init {cfg.init!r}")
        if cfg.init_scale != 1.0:
            for p in model.parameters():
                p.mul_(cfg.init_scale)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, cfg.beta2)
    )
    ckpts = set(checkpoint_steps(cfg))
    states, history = {}, []
    t0 = time.time()
    for step in range(cfg.steps + 1):
        if step in ckpts:
            states[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if step % log_every == 0 or step in ckpts:
            with torch.no_grad():
                logits = model(x)
                rec = {"step": step}
                for name, idx in (("train", tr), ("test", te)):
                    rec[f"{name}_loss"] = F.cross_entropy(logits[idx], c[idx]).item()
                    rec[f"{name}_acc"] = (logits[idx].argmax(-1) == c[idx]).float().mean().item()
                rec["weight_norm"] = math.sqrt(sum((p**2).sum().item() for p in model.parameters()))
                history.append(rec)
                if progress and step % progress == 0:
                    print(f"  {cfg.name} step {step}: train acc {rec['train_acc']:.3f}, test acc "
                          f"{rec['test_acc']:.3f}, {time.time() - t0:.0f}s", flush=True)
        if step == cfg.steps:
            break
        loss = F.cross_entropy(model(x[tr]), c[tr])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    result = {
        "config": asdict(cfg),
        "seconds": time.time() - t0,
        "history": history,
        "final": history[-1],
        "device": device,
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else 0,
    }
    torch.save({"config": asdict(cfg), "states": states}, out_dir / f"{cfg.name}.pt")
    (out_dir / f"{cfg.name}.json").write_text(json.dumps(result))
    return result


def load_checkpoints(path: Path, device="cpu") -> tuple[Config, dict[int, MLP]]:
    blob = torch.load(path, map_location=device)
    cfg = Config(**blob["config"])
    models = {}
    for step, sd in blob["states"].items():
        m = MLP(cfg.P, cfg.depth, cfg.width).to(device)
        m.load_state_dict(sd)
        m.eval()
        models[int(step)] = m
    return cfg, models


# --- Mean-ablation edges --------------------------------------------------------


@torch.no_grad()
def mean_activations(model: MLP, P: int, device="cpu") -> list[torch.Tensor]:
    a, b, _ = all_pairs(P, device)
    dtype = model.layers[0].weight.dtype
    return [h.mean(0) for h in model.hidden(one_hot_pairs(a, b, P, dtype))]


def metric_direction(model: MLP, c: torch.Tensor) -> torch.Tensor:
    """Rows of d metric / d logits: one-hot(c) - 1/P."""
    P = model.P
    d = torch.full((len(c), P), -1.0 / P, device=c.device, dtype=model.layers[0].weight.dtype)
    d[torch.arange(len(c)), c] += 1.0
    return d


def token_contributions(model: MLP, x: torch.Tensor) -> torch.Tensor:
    """(B, N, 2): what tokens a and b add to the first pre-activation, relative to the mean input."""
    P, W0 = model.P, model.layers[0].weight
    dx = x - 1.0 / P
    return torch.stack([dx[:, :P] @ W0[:, :P].T, dx[:, P:] @ W0[:, P:].T], dim=2)


def edge_scores(model: MLP, x: torch.Tensor, c: torch.Tensor, means: list[torch.Tensor], steps: int = 8):
    """EAP-IG edge scores, one (B, n_out, n_in) magnitude tensor per edge layer.

    Shapes: (B, N, 2) from the token nodes, (B, N, N) between hidden layers,
    (B, 1, N) from the last hidden layer to the sink.
    """
    hs = model.hidden(x)
    dh = [h - m for h, m in zip(hs, means)]
    W = [lin.weight for lin in model.layers]
    D = len(W) - 1  # hidden layers
    mdir = metric_direction(model, c)
    gsum = [torch.zeros(x.shape[0], w.shape[0], device=x.device, dtype=x.dtype) for w in W[:-1]]
    x0 = torch.full_like(x, 1.0 / model.P)
    with torch.enable_grad():
        for k in range(steps):
            alpha = (k + 0.5) / steps if steps > 1 else 1.0
            xa = (x0 + alpha * (x - x0)).detach()
            pres, h = [], xa
            for l, lin in enumerate(model.layers):
                z = lin(h)
                if l < D:
                    pres.append(z)
                    h = F.relu(z)
                else:
                    metric = (z * mdir).sum()
            grads = torch.autograd.grad(metric, pres)
            for l in range(D):
                gsum[l] += grads[l]
    scores = [(token_contributions(model, x) * (gsum[0] / steps)[:, :, None]).abs().detach()]
    for l in range(1, D):
        g = gsum[l] / steps
        scores.append((g[:, :, None] * W[l][None, :, :] * dh[l][:, None, :]).abs().detach())
    # sink edges: neuron j writes W[:, j] dh_j into the logits; the metric reads it through mdir
    scores.append(((mdir @ W[D]) * dh[D]).abs()[:, None, :].detach())
    return scores


@torch.no_grad()
def masked_logits(model: MLP, x: torch.Tensor, masks: list[torch.Tensor], means: list[torch.Tensor]):
    """Forward pass keeping edge (j -> i) where masks[l][b, i, j] = 1, mean-ablating it where 0.

    Mask shapes match edge_scores: (B, N, 2), (B, N, N)..., (B, 1, N).
    """
    D = len(model.layers) - 1
    z = model.layers[0](means[0].expand_as(x)) + (masks[0] * token_contributions(model, x)).sum(2)
    h = F.relu(z)
    for l in range(1, D):
        lin = model.layers[l]
        z = lin(means[l].expand_as(h)) + torch.einsum("bij,ij,bj->bi", masks[l], lin.weight, h - means[l])
        h = F.relu(z)
    out = model.layers[D]
    return out(means[D].expand_as(h)) + (masks[D][:, 0, :] * (h - means[D])) @ out.weight.T


def fidelity(logits: torch.Tensor, c: torch.Tensor, full_metric: torch.Tensor, ablated_metric: torch.Tensor):
    """Accuracy, loss, and fraction of the logit metric recovered, per input."""
    metric = logits.gather(1, c[:, None])[:, 0] - logits.mean(1)
    return {
        "acc": (logits.argmax(1) == c).float(),
        "loss": F.cross_entropy(logits, c, reduction="none"),
        "recovered": (metric - ablated_metric) / (full_metric - ablated_metric),
    }
