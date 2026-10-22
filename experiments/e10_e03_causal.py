"""Run frozen E10 E03 selectors against the same target and fresh held-out rows.

Requires pre-existing VPD/target checkpoints on the user's local machine.
Scoring never revises the frozen selection manifest. Only saved E03 results
and calibration fixtures are present in the public GitHub repository.

Use the param-decomp venv with PyTorch and dataset dependencies installed:
    python experiments/e10_e03_causal.py \
        --e03-dir out/e03_seed1 --manifest out/e10_seed1/manifest.json \
        --output out/e10_seed1/causal.json --device cuda

Use an appropriate vramorama lease on a shared 16GB GPU. No cloud rental.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from percinterp.causal_gate import score_sequential_kl
from percinterp.selector_gate import DETERMINISTIC, SCHEMA


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(manifest: dict, path: Path) -> None:
    if manifest.get("schema") != SCHEMA or manifest.get("source_type") != "E03_VPD_SAVED_INPUTS":
        raise ValueError("real E03 selector manifest required (synthetic fixtures prohibited)")
    provenance = manifest["provenance"]
    if path.resolve() != Path(provenance["source_dir"]).resolve():
        raise ValueError("source directory differs from frozen selection")
    for file, field in (("analysis.json", "analysis_file_sha256"),
                        ("crossref.pt", "crossref_file_sha256")):
        if sha256_file(path / file) != provenance[field]:
            raise ValueError(f"frozen {file} changed since selector creation")
    if (path / "vpd_run.txt").read_text(encoding="utf-8").strip() != provenance["vpd_run"]:
        raise ValueError("VPD checkpoint ID differs from frozen manifest")
    sets = manifest["selection"]
    expected = set(manifest["module_quotas"])
    if not set(DETERMINISTIC).issubset(sets):
        raise ValueError("frozen manifest is missing a deterministic comparator")
    for selector, groups in sets.items():
        if set(groups) != expected:
            raise ValueError(f"selector {selector} has different module keys")
        for module, k in manifest["module_quotas"].items():
            items = list(groups[module])
            if len(items) != k or len(items) != len(set(items)):
                raise ValueError(f"selector {selector} violates module {module} budget")


def row_hash(tokens: torch.Tensor) -> str:
    # This records token IDs only as hashes; no input text is written to output.
    arr = tokens.detach().cpu().contiguous().numpy().astype("<i8", copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def exclusion_hashes(token_loader, *, batch_size: int, batches: int) -> set[str]:
    """Exclude the exact sequences used in e03_causal.py's original seed-4 test."""
    result = set()
    for j, idx in enumerate(token_loader("test", batch_size, seed=4)):
        if j >= batches:
            break
        for row in idx[:, :512]:
            result.add(row_hash(row))
    return result


def evaluation_batches(token_loader, *, batch_size: int, n_sequences: int,
                       seed: int, excluded: set[str]):
    """Yield distinct 512-token rows; fail if insufficient eligible rows exist."""
    seen = set(excluded)
    seen_new = []
    count = 0
    for idx in token_loader("test", batch_size, seed=seed):
        chosen = []
        for row in idx[:, :512]:
            key = row_hash(row)
            if key in seen:
                continue
            seen.add(key)
            seen_new.append(key)
            chosen.append(row.detach().cpu())
            count += 1
            if count == n_sequences:
                break
        if chosen:
            yield torch.stack(chosen), len(seen_new)
        if count >= n_sequences:
            return
    raise RuntimeError(f"not enough unique evaluation sequences; got {count} / {n_sequences}")


def _signature(groups: dict, modules: list[str]) -> tuple:
    return tuple((m, tuple(sorted(int(i) for i in groups[m]))) for m in modules)


def evaluate(*, manifest: dict, e03_dir: Path, device: str,
             batch: int, n_batches: int, eval_seed: int,
             bootstrap: int = 2000) -> dict:
    """Run the frozen target/CI masks; score a paired sequence-level dataset."""
    if batch <= 0 or n_batches <= 0:
        raise ValueError("batch and n_batches must be positive")
    if eval_seed == 4:
        raise ValueError("evaluation seed 4 is reserved for previously scored E03 sequences")
    validate_manifest(manifest, e03_dir)
    os.environ["E03_OUT"] = str(e03_dir.resolve())

    from e03_common import load_target, token_loader
    from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
    from param_decomp.models.components import make_mask_infos

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(20261022)
    np.random.seed(20261022)
    info = ParamDecompRunInfo.from_path(manifest["provenance"]["vpd_run"])
    cm = ComponentModel.from_run_info(info).to(dev).eval()
    target = load_target(device).eval()
    modules = list(cm.target_module_paths)
    if set(modules) != set(manifest["module_quotas"]):
        raise ValueError("model target modules disagree with frozen selector manifest")
    selections = dict(manifest["selection"])
    selections["all"] = None
    selections["none"] = {m: [] for m in modules}
    unique = {}
    alias = {}
    for name, groups in selections.items():
        sig = "ALL" if groups is None else _signature(groups, modules)
        if sig in unique:
            alias[name] = unique[sig]
        else:
            unique[sig] = name
            alias[name] = name
    sizes = {m: int(cm.components[m].V.shape[1]) for m in modules}
    masks = {}
    for name in set(alias.values()):
        S = selections[name]
        if S is None:
            masks[name] = None
            continue
        mm = {}
        for m in modules:
            indices = list(map(int, S[m]))
            if any(i < 0 or i >= sizes[m] for i in indices):
                raise ValueError(f"out of range component ID in {name}: {m}")
            mask = torch.zeros(sizes[m], device=dev)
            if indices:
                mask[torch.tensor(indices, device=dev, dtype=torch.long)] = 1.0
            mm[m] = mask
        masks[name] = mm
    # Prior E03 causal study used seed=4, 8 batches of eight rows.
    excluded = exclusion_hashes(token_loader, batch_size=8, batches=8)
    n_sequences = batch * n_batches
    values = {name: [] for name in selections}
    evaluated_hashes = []
    start = time.time()
    with torch.inference_mode():
        for token_batch, seen in evaluation_batches(
            token_loader, batch_size=batch, n_sequences=n_sequences,
            seed=eval_seed, excluded=excluded,
        ):
            idx = token_batch.to(dev)[:, :512]
            evaluated_hashes.extend(row_hash(row) for row in token_batch)
            target_logits = target(idx)[0]
            target_logp = F.log_softmax(target_logits.float(), -1)
            ref = cm(idx, cache_type="input")
            ci = cm.calc_causal_importances(
                pre_weight_acts=ref.cache, detach_inputs=True,
                sampling=info.config.sampling).lower_leaky
            scored = {}
            for name in unique.values():
                mm = masks[name]
                restricted_ci = ci if mm is None else {
                    m: ci[m] * mm[m] for m in modules}
                logits = cm(idx, mask_infos=make_mask_infos(restricted_ci))
                logp = F.log_softmax(logits.float(), -1)
                divergence = (target_logp.exp() * (target_logp - logp)).sum(-1)
                scored[name] = divergence.mean(-1).cpu().tolist()
            for name in values:
                values[name].extend(scored[alias[name]])
            print(f"E10 scored {seen}/{n_sequences} sequence rows in {time.time()-start:.0f}s", flush=True)
    if len(evaluated_hashes) != n_sequences or set(evaluated_hashes) & excluded:
        raise AssertionError("held-out sequence exclusion or row count failed")
    score = score_sequential_kl(values, n_bootstrap=bootstrap)
    return {
        "schema": "percinterp.e10.causal.v1",
        "source_type": manifest["source_type"],
        "source_vpd_run": manifest["provenance"]["vpd_run"],
        "graph_sha256": manifest["graph_sha256"],
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True,
                                                     separators=(",", ":")).encode()).hexdigest(),
        "n_excluded_prior_exact_sequences": len(excluded),
        "n_evaluated_unique_exact_sequences": len(set(evaluated_hashes)),
        "exact_prior_overlap": len(set(evaluated_hashes) & excluded),
        "eval_seed": eval_seed,
        "eval_batch_size": batch,
        "eval_n_batches_requested": n_batches,
        "eval_n_sequences": n_sequences,
        "unique_model_evaluations_per_batch": len(unique),
        "deduplicated_selection_aliases": alias,
        "seconds": time.time()-start,
        "score": score,
        "per_sequence_kl": values,
        "sampling_note": "Different seed and exact-512-token block exclusion from old E03 causal "
                         "study. Unique sequences; source-document disjointness unverified.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e03-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=20261023)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to replace an already scored causal JSON")
    frozen = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = evaluate(manifest=frozen, e03_dir=args.e03_dir.resolve(),
                      device=args.device, batch=args.batch, n_batches=args.n_batches,
                      eval_seed=args.eval_seed, bootstrap=args.bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"E10 causal score {record['score']['status']} -> {args.output}")
    for name, value in record["score"].get("comparators", {}).items():
        print(f"  {name}: incremental recovered gap = {value['extra_gap_closed']:.2%}; "
              f"95% CI {value['extra_gap_ci_95']}")


if __name__ == "__main__":
    main()
