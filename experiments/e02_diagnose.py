"""Exploratory follow-ups to the e02 report (not pre-registered).

1. H1a mechanism: which kept-but-dangling edges carry the gap? Hypothesis: edges
   out of neurons that are inactive for this input. Their deviation from the mean
   is -mean(h), so they carry the input-dependent "this unit is off" signal, but
   their in-edges score zero (a dead ReLU passes no gradient), so they can never
   be reached from the tokens and never join the backbone.
2. H3 shape: where does learned connectivity (ranked minus shuffled backbone
   fraction at c = 2) peak, relative to t_grok?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from percinterp import dense
from percinterp.modadd import (
    all_pairs, edge_scores, load_checkpoints, masked_logits, mean_activations,
    metric_direction, one_hot_pairs, split,
)

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def dangling_breakdown(run: Path, n_inputs=64, c=8.0):
    cfg, models = load_checkpoints(run)
    model = models[max(models)]
    a, b, ans = all_pairs(cfg.P)
    _, te = split(cfg.P, cfg.frac_train, cfg.seed)
    pick = te[torch.randperm(len(te), generator=torch.Generator().manual_seed(1234))[:n_inputs]]
    x, cc = one_hot_pairs(a[pick], b[pick], cfg.P), ans[pick]
    means = mean_activations(model, cfg.P)
    S = edge_scores(model, x, cc, means, steps=8)
    prof = dense.DenseProfile(dense.rank_entry(S, unit="degree"))
    hs = model.hidden(x)
    kept = prof.masks(c)
    bb = prof.masks(c, backbone_only=True)
    out = {"c": c}
    # dangling = kept and not backbone; classify by whether the source neuron is inactive
    tot = dead = 0
    for l in range(1, len(S)):  # edges out of hidden layers
        dang = (kept[l] - bb[l]).clamp(min=0)
        src_dead = (hs[l] == 0).float()[:, None, :].expand_as(dang)
        tot += dang.sum().item()
        dead += (dang * src_dead).sum().item()
    out["dangling_out_of_hidden"] = tot
    out["share_from_inactive_sources"] = dead / max(tot, 1)
    # recovery with: all kept; backbone only; backbone plus dangling edges out of inactive sources
    mdir = metric_direction(model, cc)
    full = (model(x) * mdir).sum(1).mean()
    abl = (masked_logits(model, x, [torch.zeros_like(k) for k in kept], means) * mdir).sum(1).mean()

    def rec(masks):
        return (((masked_logits(model, x, masks, means) * mdir).sum(1).mean() - abl) / (full - abl)).item()

    plus_dead = [bb[0]] + [
        torch.maximum(bb[l], (kept[l] - bb[l]).clamp(min=0) * (hs[l] == 0).float()[:, None, :])
        for l in range(1, len(S))
    ]
    out["recovered_all_kept"] = rec(kept)
    out["recovered_backbone"] = rec(bb)
    out["recovered_backbone_plus_inactive_dangling"] = rec(plus_dead)
    # split dangling edges: source unreachable from the tokens, or target unable to reach the sink
    unreach = [((prof.F[l] > c).float()[:, None, :]).expand_as(kept[l]) for l in range(len(S))]
    noexit = [((prof.Bk[l + 1] > c).float()[:, :, None]).expand_as(kept[l]) for l in range(len(S))]
    dang = [(kept[l] - bb[l]).clamp(min=0) for l in range(len(S))]
    n_dang = sum(d.sum().item() for d in dang)
    out["share_source_unreachable"] = sum((d * u).sum().item() for d, u in zip(dang, unreach)) / max(n_dang, 1)
    out["share_target_no_exit"] = sum((d * e).sum().item() for d, e in zip(dang, noexit)) / max(n_dang, 1)
    out["recovered_backbone_plus_unreachable_source"] = rec(
        [torch.maximum(b_, d * u) for b_, d, u in zip(bb, dang, unreach)])
    out["recovered_backbone_plus_no_exit_target"] = rec(
        [torch.maximum(b_, d * e) for b_, d, e in zip(bb, dang, noexit)])
    return out


def peak_vs_grok(summary_path: Path):
    s = json.loads(summary_path.read_text())
    rows = []
    for r in s["runs"]:
        if not r["grokked"]:
            continue
        ser = r["series"]
        k = int(np.argmax(ser["learned"]))
        rows.append({"name": r["name"], "t_mem": r["t_mem"], "t_grok": r["t_grok"],
                     "peak_step": ser["step"][k], "peak_over_t_grok": ser["step"][k] / r["t_grok"],
                     "test_acc_at_peak": ser["test_acc"][k],
                     "peak_learned": ser["learned"][k], "final_learned": ser["learned"][-1],
                     "init_learned": ser["learned"][0]})
    return rows


def fig_normalised(summary_path: Path, path: Path):
    from percinterp import plotstyle

    plotstyle.apply()
    import matplotlib.pyplot as plt

    s = json.loads(summary_path.read_text())
    runs = [r for r in s["runs"] if r["grokked"]]
    widths = sorted({r["width"] for r in runs})
    colors = dict(zip(widths, plotstyle.ordinal(len(widths))))
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.0), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    seen = set()
    for r in runs:
        ser = r["series"]
        t = (np.array(ser["step"]) + 1) / r["t_grok"]
        label = f"N = {r['width']}" if r["width"] not in seen else None
        seen.add(r["width"])
        axes[0].plot(t, ser["test_acc"], color=colors[r["width"]], lw=1.0)
        axes[1].plot(t, ser["learned"], color=colors[r["width"]], lw=1.2, label=label)
    for ax in axes:
        ax.axvline(1.0, color=plotstyle.MUTED, lw=0.8, ls="--")
        ax.set_xscale("log")
    axes[0].set_ylabel("test accuracy")
    axes[1].set_ylabel("learned connectivity at c = 2\n(ranked minus shuffled backbone)")
    axes[1].set_xlabel("training step / step of 95% test accuracy")
    axes[1].legend(loc="upper left", title="depth 2, 3 seeds each")
    axes[0].set_title("Learned connectivity peaks as generalisation starts, then falls (exploratory)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    runs_dir = ROOT / "out" / "e02" / "runs"
    names = sys.argv[1:] or ["P113_D2_N128_s0", "P113_D2_N256_s0", "P113_D2_N512_s0"]
    res = {"dangling": {}, "peaks": peak_vs_grok(ROOT / "out" / "e02" / "report" / "runs" / "summary.json")}
    for n in names:
        res["dangling"][n] = dangling_breakdown(runs_dir / f"{n}.pt")
        print(n, json.dumps(res["dangling"][n]), flush=True)
    for row in res["peaks"]:
        print(f"{row['name']}: t_grok {row['t_grok']}, learned connectivity peaks at {row['peak_step']:.0f} "
              f"({row['peak_over_t_grok']:.2f} x t_grok, test acc {row['test_acc_at_peak']:.3f}), "
              f"init {row['init_learned']:+.3f} "
              f"peak {row['peak_learned']:+.3f} final {row['final_learned']:+.3f}")
    out = ROOT / "out" / "e02" / "report" / "runs" / "diagnose.json"
    out.write_text(json.dumps(res, indent=1))
    fig_normalised(ROOT / "out" / "e02" / "report" / "runs" / "summary.json",
                   ROOT / "out" / "e02" / "report" / "runs" / "h3_normalised.png")


if __name__ == "__main__":
    main()
