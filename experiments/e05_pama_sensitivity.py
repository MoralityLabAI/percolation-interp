"""Exploratory PAMA sampling sensitivity; never writes or changes a quality gate.

Runs the cached target model only. Means use nested prefixes of the held-out
stats split; every condition is evaluated on the same disjoint causal rows.
See docs/e05_pama_sensitivity.md for the plan fixed before running.
"""

from __future__ import annotations

import json
import os
import time

# This diagnostic must not download weights or data.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from e04_common import MODEL, ROOT, heldout_part
from e05_gate import WRITERS, pama_hooks

MEAN_ROWS = (64, 256, 1024)
EVAL_ROWS = 512
BATCH = 4
SLICES = ((0, 1), (2, 3))
BOOTSTRAPS = 2000
BOOT_SEED = 51021


def summary(values: np.ndarray, indices: np.ndarray) -> dict:
    assert values.shape == (EVAL_ROWS,) and np.isfinite(values).all()
    lo, hi = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return {"mean": float(values.mean()), "row_sd": float(values.std(ddof=1)),
            "bootstrap_95_ci": [float(lo), float(hi)]}


def main():
    started = time.monotonic()
    token_path = ROOT / "out" / "e04" / "heldout_tokens_128.pt"
    if not token_path.is_file():
        raise RuntimeError("Cached held-out tokens missing; refusing to fetch data")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, local_files_only=True
    ).cuda().eval()
    model.config.use_cache = False
    names = [f"model.layers.{layer}.{writer}"
             for pair in SLICES for layer in pair for writer in WRITERS]
    sums0, sums_rest = {}, {}
    means = {}

    def accumulate(name):
        def hook(_, __, output):
            pos0 = output[:, 0].float().sum(0)
            rest = output[:, 1:].float().sum((0, 1))
            sums0[name] = sums0.get(name, 0) + pos0
            sums_rest[name] = sums_rest.get(name, 0) + rest
        return hook

    with torch.inference_mode():
        handles = [model.get_submodule(name).register_forward_hook(accumulate(name))
                   for name in names]
        rows = 0
        try:
            for idx in heldout_part(BATCH, "stats"):
                model(idx.cuda())
                rows += len(idx)
                if rows in MEAN_ROWS:
                    means[rows] = (
                        {name: (sums0[name] / rows).clone() for name in names},
                        {name: (sums_rest[name] / (rows * 127)).clone() for name in names},
                    )
                    print(f"Means captured after {rows} stats rows", flush=True)
                if rows == max(MEAN_ROWS):
                    break
        finally:
            for handle in handles:
                handle.remove()
        assert set(means) == set(MEAN_ROWS)
        values = {f"{pair[0]}-{pair[1]}": {n: [] for n in MEAN_ROWS} for pair in SLICES}
        rows = 0
        for idx in heldout_part(BATCH, "causal"):
            idx = idx.cuda()
            target = F.log_softmax(model(idx).logits.float(), dim=-1)
            probs = target.exp()
            for pair in SLICES:
                pair_names = [f"model.layers.{layer}.{writer}"
                              for layer in pair for writer in WRITERS]
                key = f"{pair[0]}-{pair[1]}"
                for n in MEAN_ROWS:
                    handles = pama_hooks(model, pair_names, *means[n])
                    try:
                        logp = F.log_softmax(model(idx).logits.float(), dim=-1)
                        row_kl = (probs * (target - logp)).sum(-1).mean(-1)
                        values[key][n].extend(row_kl.cpu().double().tolist())
                        del logp, row_kl
                    finally:
                        for handle in handles:
                            handle.remove()
            del target, probs
            rows += len(idx)
            if rows % 32 == 0:
                print(f"Evaluated {rows}/{EVAL_ROWS} causal rows; "
                      f"elapsed {time.monotonic() - started:.0f}s", flush=True)
            if rows == EVAL_ROWS:
                break
    assert rows == EVAL_ROWS
    rng = np.random.default_rng(BOOT_SEED)
    indices = rng.integers(0, EVAL_ROWS, size=(BOOTSTRAPS, EVAL_ROWS))
    result = {"exploratory": True, "model": MODEL, "dtype": "float32",
              "mean_rows": list(MEAN_ROWS), "eval_rows": EVAL_ROWS,
              "batch": BATCH, "context": 128, "bootstrap_draws": BOOTSTRAPS,
              "bootstrap_seed": BOOT_SEED,
              "uncertainty_scope": "row bootstrap conditional on each fitted mean; "
              "documents may span rows; not a document-level confidence guarantee",
              "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "slices": {}}
    for key, conditions in values.items():
        arrays = {n: np.asarray(v) for n, v in conditions.items()}
        base = arrays[256]
        result["slices"][key] = {
            "conditions": {str(n): {
                **summary(v, indices), "original_gate_first_128_rows_mean": float(v[:128].mean()),
                "implied_gate_threshold": 0.25 * float(v.mean()),
            } for n, v in arrays.items()},
            "paired_delta_vs_256_rows": {
                str(n): summary(v - base, indices) for n, v in arrays.items() if n != 256},
            "row_kl": {str(n): v.tolist() for n, v in arrays.items()},
        }
    result["elapsed_seconds"] = time.monotonic() - started
    output = ROOT / "results" / "e05_pama_sensitivity.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace completed exploratory result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}; elapsed {result['elapsed_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
