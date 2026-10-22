"""Resume E10 from locally saved VPD inputs, without re-training or renting GPUs.

Dry preflight (no mutation):
    python experiments/e10_run_local.py

Freeze selector manifests with CPU-only attribution/statistics reads:
    python experiments/e10_run_local.py --prepare

Run only if source-to-sink selection differs from every cheap comparator:
    python experiments/e10_run_local.py --execute --device cuda

Uses the existing vramorama lease for CUDA when found. Outputs are isolated
under out/e10_decisive; source directories and pre-registered studies stay intact.
The original E03 data remain local and are never uploaded by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from percinterp.selector_gate import DETERMINISTIC, SCHEMA  # noqa: E402
from percinterp.causal_gate import combine_seed_results  # noqa: E402


def probe_seed(directory: Path, param_root: Path) -> dict:
    """Cheap readiness check, including the original attribution file path."""
    directory, param_root = directory.resolve(), param_root.resolve()
    missing_prepare = []
    for name in ("analysis.json", "crossref.pt", "vpd_run.txt"):
        if not (directory / name).is_file():
            missing_prepare.append(str(directory / name))
    run_id = None
    runfile = directory / "vpd_run.txt"
    if runfile.is_file():
        run_id = runfile.read_text(encoding="utf-8").strip()
        if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
            raise ValueError(f"invalid VPD run ID from {runfile}")
        attr = list((param_root / "dataset_attributions" / run_id).glob(
            "da-*/dataset_attributions.pt"))
        if not attr:
            missing_prepare.append(str(param_root / "dataset_attributions" /
                                       run_id / "da-*/dataset_attributions.pt"))
    missing_causal = list(missing_prepare)
    if not (directory / "target.txt").is_file():
        missing_causal.append(str(directory / "target.txt"))
    else:
        raw = (directory / "target.txt").read_text(encoding="utf-8").strip()
        target = Path(raw).expanduser() if raw else None
        if target is None or not target.exists():
            missing_causal.append(f"target model checkpoint: {raw or '<empty>'}")
    if run_id:
        candidates = (param_root / "runs" / f"param-decomp-{run_id}",
                      param_root / "decompositions" / run_id)
        if not any((p / "model_40000.pth").is_file() for p in candidates):
            missing_causal.append("VPD model_40000.pth in " +
                                  " or ".join(map(str, candidates)))
    return {"source_dir": str(directory), "run_id": run_id,
            "missing_prepare": missing_prepare,
            "missing_causal": missing_causal,
            "can_prepare": not missing_prepare,
            "can_evaluate": not missing_causal}


def frozen_manifest_id(m: dict) -> str:
    return hashlib.sha256(json.dumps(m, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_frozen_metadata(manifest: dict, path: Path) -> None:
    """Refuse stale prepared manifests before even applying a cheap stop rule."""
    provenance = manifest.get("provenance", {})
    for filename, label in (("analysis.json", "analysis_file_sha256"),
                            ("crossref.pt", "crossref_file_sha256")):
        actual = path / filename
        if not actual.is_file() or label not in provenance:
            raise ValueError(f"frozen manifest lacks {label} / {actual}")
        h = hashlib.sha256()
        with actual.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != provenance[label]:
            raise ValueError(f"frozen input changed: {actual}")
    vpd_run = (path / "vpd_run.txt").read_text(encoding="utf-8").strip()
    if vpd_run != provenance.get("vpd_run"):
        raise ValueError("frozen VPD run ID changed")


def triage_manifest(m: dict) -> dict:
    if m.get("schema") != SCHEMA or m.get("source_type") != "E03_VPD_SAVED_INPUTS":
        raise ValueError("expected real frozen E03 selector manifest")
    selections = m["selection"]
    quotas = m["module_quotas"]
    if not set(DETERMINISTIC).issubset(selections):
        raise ValueError("missing E10 deterministic comparison")
    for name, groups in selections.items():
        if set(groups) != set(quotas):
            raise ValueError(f"selector {name} has mismatched module keys")
        for module, quota in quotas.items():
            chosen = groups[module]
            if len(chosen) != quota or len(set(chosen)) != quota:
                raise ValueError(f"selector {name} violates {module} quota")
    baseline = selections["sieve"]
    equivalent = sorted(
        name for name in DETERMINISTIC if name != "sieve" and
        all(set(selections[name][module]) == set(baseline[module])
            for module in quotas))
    return {
        "manifest_sha256": frozen_manifest_id(m),
        "n_selected": sum(quotas.values()),
        "exact_equivalent_selectors": equivalent,
        "path_specific_gain_possible": not equivalent,
        "note": ("One deterministic comparator has exactly the sieve membership; "
                 "E10's per-seed incremental gate is mathematically impossible."
                 if equivalent else
                 "Different memberships leave causal performance unresolved."),
    }


def vramorama_binary() -> str | None:
    exe = shutil.which("vramorama")
    if exe:
        return exe
    appdata = os.getenv("LOCALAPPDATA")
    if appdata:
        path = Path(appdata) / "Programs" / "vramorama" / "vramorama.exe"
        if path.is_file():
            return str(path)
    return None


def inference_command(python: str, directory: Path, manifest_path: Path,
                      causal_path: Path, *, device: str, batch: int,
                      n_batches: int, bootstrap: int, vram_mib: int,
                      vrama: str | None, direct_gpu: bool) -> list[str]:
    cmd = [python, str(ROOT / "experiments" / "e10_e03_causal.py"),
           "--e03-dir", str(directory), "--manifest", str(manifest_path),
           "--output", str(causal_path), "--device", device,
           "--batch", str(batch), "--n-batches", str(n_batches),
           "--bootstrap", str(bootstrap)]
    if device.startswith("cuda"):
        if vrama:
            return [vrama, "run", "--vram", f"{vram_mib}M",
                    "--owner", "e10-selector-gate", "--", *cmd]
        if not direct_gpu:
            raise ValueError("CUDA without vramorama requires --allow-direct-cuda")
    return cmd


def run(args: argparse.Namespace) -> dict:
    dirs = [p.resolve() for p in args.e03_dirs]
    if len({p.name for p in dirs}) != len(dirs) or len(set(dirs)) != len(dirs):
        raise ValueError("E03 input directories must have distinct names")
    if args.execute and args.prepare:
        raise ValueError("--execute already performs --prepare; provide one mode")
    mode = "execute" if args.execute else "prepare" if args.prepare else "check"
    param_root = Path.home() / "param_decomp_out"
    output = {"schema": "percinterp.e10.local_run.v1", "mode": mode,
              "n_inputs": len(dirs), "seeds": [], "decision": "CHECK"}
    probes = [probe_seed(p, param_root) for p in dirs]
    for p in probes:
        output["seeds"].append({"readiness": p})
    if mode == "check":
        output["decision"] = ("READY_FOR_SELECTOR_PREPARATION" if
                              all(p["can_prepare"] for p in probes)
                              else "MISSING_LOCAL_VPD_INPUTS")
        return output
    if not all(p["can_prepare"] for p in probes):
        output["decision"] = "MISSING_LOCAL_VPD_INPUTS"
        return output
    vrama = (args.vramorama or vramorama_binary()) if args.device.startswith("cuda") else None
    if args.execute:
        if not all(p["can_evaluate"] for p in probes):
            output["decision"] = "MISSING_LOCAL_CAUSAL_INPUTS"
            return output
        # Check lease availability before creating any manifest or spending compute.
        inference_command(args.causal_python, dirs[0], Path("fixture"),
                          Path("fixture"), device=args.device, batch=args.batch,
                          n_batches=args.n_batches, bootstrap=args.bootstrap,
                          vram_mib=args.vram_mib, vrama=vrama,
                          direct_gpu=args.allow_direct_cuda)
    # CPU manifest preparation. Never overwrite a frozen manifest.
    manifest_records = []
    for path, entry in zip(dirs, output["seeds"]):
        stem = args.output_dir.resolve() / path.name
        manifest_path = stem / "manifest.json"
        if not manifest_path.exists():
            stem.mkdir(parents=True, exist_ok=True)
            subprocess.run([args.analysis_python, str(ROOT / "experiments" /
                            "e10_prepare.py"), "--e03-dir", str(path),
                            "--output", str(manifest_path), "--n-random",
                            str(args.n_random)], cwd=ROOT, check=True)
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_frozen_metadata(m, path)
        gate = triage_manifest(m)
        if Path(m["provenance"]["source_dir"]).resolve() != path:
            raise ValueError(f"frozen source differs from argument: {manifest_path}")
        entry.update({"manifest": str(manifest_path), "triage": gate})
        manifest_records.append((path, manifest_path, gate))
    if any(not g[2]["path_specific_gain_possible"] for g in manifest_records):
        output["decision"] = "EARLY_STOP_EQUIVALENT_DETERMINISTIC_SELECTOR"
        output["description"] = ("At least one decomposition sieve set is identical "
             "to a simpler deterministic selector; two-seed practical advantage "
             "cannot pass. No causal GPU work was launched.")
        return output
    if mode == "prepare":
        output["decision"] = "READY_FOR_FROZEN_CAUSAL_EVALUATION"
        return output
    records = []
    for path, manifest_path, gate in manifest_records:
        causal_path = manifest_path.parent / "causal.json"
        if not causal_path.exists():
            cmd = inference_command(args.causal_python, path, manifest_path,
                                    causal_path, device=args.device, batch=args.batch,
                                    n_batches=args.n_batches, bootstrap=args.bootstrap,
                                    vram_mib=args.vram_mib, vrama=vrama,
                                    direct_gpu=args.allow_direct_cuda)
            subprocess.run(cmd, cwd=ROOT, check=True)
        record = json.loads(causal_path.read_text(encoding="utf-8"))
        if record.get("schema") != "percinterp.e10.causal.v1" or \
           record.get("manifest_sha256") != gate["manifest_sha256"]:
            raise ValueError(f"existing causal result belongs to a different manifest: {causal_path}")
        records.append(record)
    summary = combine_seed_results(records)
    output["decision"] = summary["status"]
    output["causal_summary"] = summary
    summary_path = args.output_dir.resolve() / "combined_score.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite scored summary {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8")
    output["combined_score_path"] = str(summary_path)
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e03-dirs", nargs="+", type=Path,
                    default=[ROOT / "out" / "e03_seed1", ROOT / "out" / "e03_seed2"])
    ap.add_argument("--output-dir", type=Path, default=ROOT / "out" / "e10_decisive")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    ap.add_argument("--analysis-python", default=sys.executable)
    ap.add_argument("--causal-python", default=sys.executable)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vramorama", default=None)
    ap.add_argument("--vram-mib", type=int, default=4096)
    ap.add_argument("--allow-direct-cuda", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--n-random", type=int, default=20)
    args = ap.parse_args()
    if min(args.vram_mib, args.batch, args.n_batches, args.bootstrap) <= 0:
        ap.error("compute budgets must be positive")
    try:
        result = run(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        ap.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["decision"].startswith("MISSING_"):
        raise SystemExit(2)
    if result["decision"].startswith("EARLY_STOP_"):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
