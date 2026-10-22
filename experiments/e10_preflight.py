"""Cheap E10 preflight from saved E06 graphs (no model tensor access needed).

Quantifies exactly how much E06's registered local in/out-degree contact rule
overselects the true path backbone. This is descriptive and applies at E06's
c=0.1 and c=1; E03's frozen causal sieve may use a different c*.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ("heldout_e03_seed1.json", "heldout_e03_seed2.json")


def read_saved(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: None)


def audit_one(document: dict, name: str) -> dict:
    n = int(document["n_interior"])
    if n <= 0:
        raise ValueError("expected positive interior node count")
    cells = document["joint_degree"]["cells"]
    out = {"decomposition": name, "n_interior": n}
    for label in ("c_s", "c_1"):
        rec = cells[label]
        diagnostic = rec["degree_determined"]
        c = float(rec["c"])
        b = int(round(float(diagnostic["backbone"]) * n))
        d = int(round(float(diagnostic["local"]) * n))
        if b > d:
            raise ValueError("a true interior backbone member must have positive local in/out contacts")
        excess = d - b
        out[label] = {
            "occupation": c,
            "path_backbone_nodes": b,
            "degree_positive_nodes": d,
            "local_false_positives": excess,
            "local_false_negatives": 0,
            "local_precision": b / d if d else 1.0,
            "local_recall": 1.0,
            "path_information_nodes": excess,
            "reference_p_wiring_excess": (float(rec["p_high"]) if rec.get("testable") else None),
            "degree_testable": bool(rec.get("testable")),
        }
    return out


def build(root: Path = ROOT) -> dict:
    cases = []
    for name in CASES:
        path = root / "results" / "e06" / name
        cases.append(audit_one(read_saved(path), name))
    return {
        "schema": "percinterp.e10.preflight.v1",
        "source": "saved registered E06 joint-degree null on E03 decomposition seeds 1 and 2",
        "analysis_type": "historical structural comparison; no new causal intervention",
        "cases": cases,
        "at_c01": {"backbone_nodes": sum(x["c_s"]["path_backbone_nodes"] for x in cases),
                   "additional_local_candidates": sum(x["c_s"]["local_false_positives"] for x in cases)},
        "limitation": "These sparse c=0.1 counts are not the E03 causal c_star. "
                      "The E10 frozen manifest must measure exact selector overlap at c_star.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    report = build(a.root)
    for x in report["cases"]:
        print(f"{x['decomposition']} c=0.1: backbone {x['c_s']['path_backbone_nodes']} "
              f"vs local degree-positive {x['c_s']['degree_positive_nodes']}; "
              f"local extras {x['c_s']['local_false_positives']}")
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(a.output)


if __name__ == "__main__":
    main()
