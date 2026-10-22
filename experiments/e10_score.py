"""Combine completed E10 causal JSONs from distinct frozen decomposition manifests.

    python experiments/e10_score.py --results out/e10_seed1/causal.json \
        out/e10_seed2/causal.json --output out/e10_score.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from percinterp.causal_gate import combine_seed_results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("refusing to overwrite an existing combined score")
    records = []
    for path in args.results:
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("schema") != "percinterp.e10.causal.v1":
            ap.error(f"not an E10 causal record: {path}")
        if rec["source_type"] != "E03_VPD_SAVED_INPUTS":
            ap.error(f"synthetic/calibration data prohibited in primary score: {path}")
        records.append(rec)
    report = combine_seed_results(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"E10 decision: {report['status']} ({report['n_seed_results']} decompositions)")
    print(f"Score written to {args.output}")


if __name__ == "__main__":
    main()
