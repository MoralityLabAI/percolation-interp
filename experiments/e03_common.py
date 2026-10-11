"""Shared pieces for e03. Runs in param-decomp's venv (Python 3.13), where percinterp is installed.

Residual tower levels for the 2-layer target:
    level 0: token embedding (input to block 0)
    level 1: output of block 0
    level 2: output of block 1 (input to the final norm)
"""

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "e03"
TOKENIZER = "SimpleStories/test-SimpleStories-gpt2-1.25M"
DATASET = "SimpleStories/SimpleStories"


def target_path() -> Path:
    """Checkpoint of the e03 target, recorded by the pretraining stage."""
    return Path((OUT / "target.txt").read_text().strip())


def load_target(device: str = "cuda"):
    from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP

    model = LlamaSimpleMLP.from_pretrained(str(target_path())).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def token_loader(split: str, batch_size: int, n_ctx: int = 512, seed: int = 0):
    """Batches of token ids (B, n_ctx) from SimpleStories, tokenised as in the VPD task config."""
    from param_decomp.data import DatasetConfig, create_data_loader, input_ids_collate_fn

    cfg = DatasetConfig(name=DATASET, is_tokenized=False, hf_tokenizer_path=TOKENIZER, streaming=False,
                        split=split, n_ctx=n_ctx, seed=seed, column_name="story")
    loader, _ = create_data_loader(cfg, batch_size=batch_size, buffer_size=1000, global_seed=seed,
                                   collate_fn=input_ids_collate_fn)
    return loader


def residuals(model, idx: torch.Tensor) -> list[torch.Tensor]:
    """The three tower levels, each (B, T, d)."""
    x = model.wte(idx)
    levels = [x]
    for block in model.h:
        x = block(x)
        levels.append(x)
    return levels


def logits_from_level(model, x: torch.Tensor, level: int) -> torch.Tensor:
    """Run the rest of the model from a residual at the given level."""
    for block in list(model.h)[level:]:
        x = block(x)
    return model.lm_head(model.ln_f(x))
