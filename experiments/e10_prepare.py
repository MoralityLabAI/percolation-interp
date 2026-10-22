"""Freeze E10 budget-matched selectors without consulting new causal outcomes.

CPU fixture: python experiments/e10_prepare.py --fixture --output out/e10_fixture.json
Real E03:   python experiments/e10_prepare.py --e03-dir out/e03_seed1 \
                --output out/e10_seed1/manifest.json

Real mode needs the existing saved E03 crossref.pt, VPD attribution tensors and
analysis.json in their *original* directories. It never reads prior causal.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
from percinterp.selector_gate import select_sets


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def fixture(n_random: int, seed: int) -> dict:
    node_ids = [("source", 0), ("source", 1), ("block0", 0),
                ("block0", 1), ("block1", 0), ("block1", 1), ("sink", 0)]
    src = np.array([0, 1, 2, 3, 2, 3, 4, 5])
    dst = np.array([2, 3, 4, 5, 5, 4, 6, 6])
    weight = np.array([10., 9., 8., 7., 1., 1., 6., 2.])
    entry = np.array([.10, .20, .12, .60, .90, .80, .20, .70])
    plan = select_sets(
        n_nodes=7, src=src, dst=dst, weight=weight, entry=entry,
        occupation=.25, node_ids=node_ids,
        pools={"block0": [0, 1], "block1": [0, 1]},
        backbone={"block0": [0], "block1": [0]},
        ci_mean={"block0": [1, 5], "block1": [1, 8]},
        firing_rate={"block0": [.11, .90], "block1": [.2, .90]},
        seed=seed, n_random=n_random)
    plan["source_type"] = "SYNTHETIC_SELECTION_FIXTURE"
    plan["provenance"] = {"explanation": "Seven-node toy graph; no target inference or VPD"}
    return plan


def prepare_e03(path: Path, n_random: int, seed: int) -> dict:
    path = path.resolve()
    if not (path / "analysis.json").exists() or not (path / "crossref.pt").exists():
        raise FileNotFoundError(f"missing E03 analysis.json or crossref.pt under {path}")
    os.environ["E03_OUT"] = str(path)
    # This is a CPU-only loader; e03_analyze reads the original VPD
    # attribution tensor from ~/param_decomp_out/dataset_attributions/.
    import e03_analyze as ea
    from percinterp.sieve import degree_entry
    import torch

    ea.OUT = path
    analysis_path = path / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if "vpd_sieve" not in analysis:
        raise ValueError("expected pre-existing E03 VPD sieve results")
    X, comp_rate = ea.load_crossref()
    g, weight, node_ids = ea.vpd_graph(comp_rate)
    modules = list(X["modules"])
    offsets = X["comp_offsets"]
    ci = X["ci_sum"].numpy() / X["tokens"]
    rate = X["n_comp"].numpy() / X["tokens"]
    pools = {}
    ci_by_module = {}
    rate_by_module = {}
    for j, mod in enumerate(modules):
        a, b = int(offsets[j]), int(offsets[j + 1])
        pools[mod] = np.flatnonzero(comp_rate[mod] >= ea.MIN_RATE).astype(int).tolist()
        ci_by_module[mod] = ci[a:b]
        rate_by_module[mod] = rate[a:b]
    bb = {mod: [] for mod in modules}
    for mod, comp in analysis["vpd_sieve"]["backbone"]:
        if mod in bb:
            bb[mod].append(int(comp))
    entry = degree_entry(g, weight, g.edge_layer("dst"))
    plan = select_sets(
        n_nodes=g.n_nodes, src=g.src, dst=g.dst, weight=weight, entry=entry,
        occupation=float(analysis["vpd_sieve"]["c_star"]), node_ids=node_ids,
        pools=pools, backbone=bb, ci_mean=ci_by_module, firing_rate=rate_by_module,
        seed=seed, n_random=n_random)
    run_id = (path / "vpd_run.txt").read_text(encoding="utf-8").strip()
    plan["source_type"] = "E03_VPD_SAVED_INPUTS"
    plan["provenance"] = {
        "analysis_file_sha256": _digest(analysis_path),
        "crossref_file_sha256": _digest(path / "crossref.pt"),
        "vpd_run": run_id,
        "source_dir": str(path),
        "analysis_c_star": float(analysis["vpd_sieve"]["c_star"]),
        "selection_inputs": "E03 attribution graph and crossref statistics only",
        "future_causal_rows": "E03 test split, new seed with exact-sequence exclusion of old seed4 blocks",
    }
    return plan


def write_manifest(output: Path, manifest: dict, *, overwrite: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"selection manifest already exists: {output}")
    body = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(body, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", action="store_true")
    mode.add_argument("--e03-dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--overwrite", action="store_true",
                    help="explicitly replace an exploratory manifest before causal scoring")
    args = ap.parse_args()
    plan = fixture(args.n_random, args.seed) if args.fixture else prepare_e03(
        args.e03_dir, args.n_random, args.seed)
    write_manifest(args.output, plan, overwrite=args.overwrite)
    print(f"E10 selector plan written to {args.output}")
    print(f"Source={plan['source_type']}, modules={len(plan['module_quotas'])}, "
          f"budget={sum(plan['module_quotas'].values())}")
    for name, stats in plan["overlap"].items():
        print(f"  {name}: Jaccard={stats['jaccard']:.4f}; "
              f"replacements={stats['replacements']}")
    if plan["overlap"]["local_degree"]["same_selected_components"]:
        print("LOCAL-DEGREE GATE: identical selected set; causal scores must match exactly.")


if __name__ == "__main__":
    main()
