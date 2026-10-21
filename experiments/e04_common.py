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
# The main run is seed 0 at 20k steps; the follow-ups (docs/e04_design.md, gate addendum) set
# E04_STEPS=40000 (e04b) or E04_SEED=1 (replication), with their own E04_OUT.
STEPS = int(os.environ.get("E04_STEPS", 20000))
SEED = int(os.environ.get("E04_SEED", 0))
BATCH = os.environ.get("E04_BATCH")  # e05b: decompositions differ from e05's only in batch_size
# Decomposed layers: e04 used 2-3 (revised from 13-14; docs/e04_design.md); e05 sets E04_SLICE=0,1.
SLICE = tuple(int(x) for x in os.environ.get("E04_SLICE", "2,3").split(","))


def install_qwen3_schema():
    """Register a Qwen3ForCausalLM path schema with param-decomp's topology at runtime.

    The same schema as the fork's local feature/qwen3-topology branch, so attributions run on
    any param-decomp checkout. Only the topology (attributions) needs it; training does not.
    """
    from param_decomp.topology import path_schemas as ps
    from param_decomp.topology import topology as tp

    if getattr(tp.get_path_schema, "qwen3_registered", False):
        return
    from transformers.models.qwen3 import Qwen3ForCausalLM

    class _HFQwen3PathSchema(ps._PathSchema):
        embedding_path = "model.embed_tokens"
        blocks = "model.layers"
        attn = ps._SeparateAttnPathSchema(base="self_attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
        mlp = ps._GLUPathSchema(base="mlp", gate="gate_proj", up="up_proj", down="down_proj")
        unembed_path = "lm_head"

    original = ps.get_path_schema

    def get_path_schema(model):
        return _HFQwen3PathSchema() if isinstance(model, Qwen3ForCausalLM) else original(model)

    get_path_schema.qwen3_registered = True
    ps.get_path_schema = tp.get_path_schema = get_path_schema
MIN_RATE = 1e-5
CI_FIRE = 0.1
STAGE = {"attn.q": 0, "attn.k": 0, "attn.v": 0, "attn.o": 1, "glu.gate": 2, "glu.up": 2, "glu.down": 3}


def find_decomposition() -> Path | None:
    """The e04 decomposition's folder, identified by model, steps and slice in its final_config.yaml.

    A run that reached its final checkpoint wins; otherwise the newest (a run killed before
    finishing, e.g. by a Windows Update restart, leaves a matching folder behind).
    """
    found = []
    for d in (PD_OUT / "decompositions").glob("*"):
        cfg = d / "final_config.yaml"
        if not cfg.exists():
            continue
        text = cfg.read_text(encoding="utf-8")
        if (f"pretrained_model_name: {MODEL}" in text and re.search(rf"(?m)^steps: {STEPS}$", text)
                and re.search(rf"(?m)^seed: {SEED}$", text) and all(f"model.layers.{l}." in text for l in SLICE)
                and (BATCH is None or re.search(rf"(?m)^batch_size: {BATCH}$", text))):
            found.append(d)
    done = [d for d in found if (d / f"model_{STEPS}.pth").exists()]
    pool = done or found
    return max(pool, key=lambda d: (d / "final_config.yaml").stat().st_mtime) if pool else None


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
