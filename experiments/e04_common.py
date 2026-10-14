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
MIN_RATE = 1e-5
CI_FIRE = 0.1
STAGE = {"attn.q": 0, "attn.k": 0, "attn.v": 0, "attn.o": 1, "glu.gate": 2, "glu.up": 2, "glu.down": 3}


def find_run(checkpoint_step: int | None = None) -> str:
    """Run id of the e04 decomposition, registered in the local runs cache.

    With checkpoint_step, registers a copy holding only that checkpoint (for testing on a
    run that is still training) under the id '<run>-s<step>'.
    """
    for d in sorted((PD_OUT / "decompositions").glob("*")):
        cfg = d / "final_config.yaml"
        if not cfg.exists():
            continue
        text = cfg.read_text(encoding="utf-8")
        if f"pretrained_model_name: {MODEL}" in text and re.search(rf"(?m)^steps: {STEPS}$", text):
            step = checkpoint_step or STEPS
            ckpt = d / f"model_{step}.pth"
            assert ckpt.exists(), f"{ckpt} not found"
            run_id = d.name if checkpoint_step is None else f"{d.name}-s{step}"
            cache = PD_OUT / "runs" / f"param-decomp-{run_id}"
            if not cache.exists():
                cache.mkdir(parents=True)
                shutil.copyfile(cfg, cache / "final_config.yaml")
                shutil.copyfile(ckpt, cache / ckpt.name)
            return run_id
    raise SystemExit("no e04 decomposition found")


def load_component_model(run_id: str, device: str = "cuda"):
    from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo

    run_info = ParamDecompRunInfo.from_path(run_id)
    cm = ComponentModel.from_run_info(run_info).to(device).eval()
    return cm, run_info


def heldout_loader(batch_size: int, seed: int, n_ctx: int = 128):
    """Token batches (B, n_ctx) from the held-out 5% of pile-10k, Qwen tokenizer."""
    from param_decomp.data import DatasetConfig, create_data_loader, input_ids_collate_fn

    cfg = DatasetConfig(name="NeelNanda/pile-10k", is_tokenized=False, hf_tokenizer_path=MODEL, streaming=False,
                        split="train[95%:]", n_ctx=n_ctx, seed=seed, column_name="text")
    loader, _ = create_data_loader(cfg, batch_size=batch_size, buffer_size=1000, global_seed=seed,
                                   collate_fn=input_ids_collate_fn)
    return loader


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
