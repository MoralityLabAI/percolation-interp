"""Shared pieces for e04 (Qwen3-0.6B slice). Runs in param-decomp's venv on the qwen3-topology branch."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("E04_OUT", ROOT / "out" / "e04"))
PD_OUT = Path(os.environ.get("PARAM_DECOMP_OUT_DIR", Path.home() / "param_decomp_out"))
MODEL = "Qwen/Qwen3-0.6B"
STEPS = 20000
SLICE = (2, 3)  # decomposed layers (revised from 13-14; docs/e04_design.md)
MIN_RATE = 1e-5
CI_FIRE = 0.1
STAGE = {"attn.q": 0, "attn.k": 0, "attn.v": 0, "attn.o": 1, "glu.gate": 2, "glu.up": 2, "glu.down": 3}


def find_decomposition() -> Path | None:
    """The e04 decomposition's folder, identified by model, steps and slice in its final_config.yaml."""
    for d in sorted((PD_OUT / "decompositions").glob("*")):
        cfg = d / "final_config.yaml"
        if not cfg.exists():
            continue
        text = cfg.read_text(encoding="utf-8")
        if (f"pretrained_model_name: {MODEL}" in text and re.search(rf"(?m)^steps: {STEPS}$", text)
                and all(f"model.layers.{l}." in text for l in SLICE)):
            return d
    return None


def find_run(checkpoint_step: int | None = None) -> str:
    """Run id of the e04 decomposition, registered in the local runs cache.

    With checkpoint_step, registers a copy holding only that checkpoint (for testing on a
    run that is still training) and returns the path to that checkpoint, which
    ParamDecompRunInfo.from_path accepts (a suffixed id would not parse as a run id).
    """
    d = find_decomposition()
    if d is None:
        raise SystemExit("no e04 decomposition found")
    step = checkpoint_step or STEPS
    ckpt = d / f"model_{step}.pth"
    assert ckpt.exists(), f"{ckpt} not found"
    run_id = d.name if checkpoint_step is None else f"{d.name}-s{step}"
    cache = PD_OUT / "runs" / f"param-decomp-{run_id}"
    if not cache.exists():
        cache.mkdir(parents=True)
        shutil.copyfile(d / "final_config.yaml", cache / "final_config.yaml")
        shutil.copyfile(ckpt, cache / ckpt.name)
    return run_id if checkpoint_step is None else str(cache / ckpt.name)


def load_component_model(run_id: str, device: str = "cuda"):
    from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo

    run_info = ParamDecompRunInfo.from_path(run_id)
    cm = ComponentModel.from_run_info(run_info).to(device).eval()
    return cm, run_info


def heldout_tokens(n_ctx: int = 128) -> torch.Tensor:
    """The held-out 5% of pile-10k as (rows, n_ctx) Qwen token ids, built once and cached.

    Tokenised in a single process: datasets' multiprocess map loses workers on this split
    (some Pile documents run to ~1M tokens). Documents are joined with EOS and cut into rows.
    """
    path = OUT.parent / "e04" / f"heldout_tokens_{n_ctx}.pt"
    if path.exists():
        return torch.load(path)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = load_dataset("NeelNanda/pile-10k", split="train[95%:]")["text"]
    ids: list[int] = []
    for text in docs:
        ids.extend(tok(text)["input_ids"])
        ids.append(tok.eos_token_id)
    rows = torch.tensor(ids[: len(ids) // n_ctx * n_ctx]).view(-1, n_ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, path)
    return rows


def heldout_loader(batch_size: int, seed: int, n_ctx: int = 128):
    """Token batches (B, n_ctx) from the held-out 5% of pile-10k, in a seed-fixed row order."""
    rows = heldout_tokens(n_ctx)
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))
    for i in range(0, len(order) - batch_size + 1, batch_size):
        yield rows[order[i:i + batch_size]]


# Disjoint ranges of one fixed shuffle of the held-out rows (5,910 at n_ctx 128), so the rows
# that select the backbone (attributions) are never the rows that test it (causal).
PARTS = {"stats": (0, 2000), "attributions": (2000, 4000), "causal": (4000, 5910)}


def heldout_part(batch_size: int, part: str, n_ctx: int = 128):
    """Token batches (B, n_ctx) from one part of the held-out rows."""
    rows = heldout_tokens(n_ctx)
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(0))
    lo, hi = PARTS[part]
    order = order[lo:min(hi, len(order))]
    for i in range(0, len(order) - batch_size + 1, batch_size):
        yield rows[order[i:i + batch_size]]


def ci_lower(cm, run_info, idx):
    out = cm(idx, cache_type="input")
    return cm.calc_causal_importances(pre_weight_acts=out.cache, detach_inputs=True,
                                      sampling=run_info.config.sampling).lower_leaky


def stage_of(canon: str) -> tuple[int, int]:
    """'13.glu.gate' -> (13, 2): sort key in residual-stream order."""
    layer, rest = canon.split(".", 1)
    return int(layer), STAGE[rest]


def load_stats():
    return torch.load(OUT / "stats.pt", weights_only=False)
