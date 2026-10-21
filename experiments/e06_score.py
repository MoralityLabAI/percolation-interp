"""e06 scoring: P1-P5 of docs/e06_design.md (pre-registered) from e06_heldout.py records.

score(rec) applies the design's pass rules to one graph's record. As a script it scores
result files and says, per prediction, whether it holds on every graph it applies to:

  python experiments/e06_score.py results/e06/heldout_*.json

Which graphs count (design, "Data" and "Predictions"):
  - kind "e05b": P1-P5 all count;
  - kind "e03" (toy-model seeds 1 and 2): P1, P2, P4, P5 count; P3 is reported only, since
    their large-c curves were seen before the design;
  - kind "check" (any other graph, e.g. the e04 code check): nothing counts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ALPHA_P2 = 0.01
ALPHA_P4_CELL = 0.005  # 0.01, Bonferroni over the two cells
ALPHA_P5 = 0.01
TOL = 1e-9  # float slack for the identity-based comparisons (far below one node, 1/n_interior)


def _counts(kind: str, pred: str) -> bool:
    if kind == "e05b":
        return True
    if kind == "e03":
        return pred != "P3"
    return False


def score(rec: dict) -> dict:
    kind = rec["kind"]
    wl = rec["nulls"]["within_layer"]["fixed"]
    dc = rec["nulls"]["in_out_decoupled"]
    out = {}

    # P1: the small-c excess is alignment (N-WL at c_s)
    t = wl["c_s"]["excess"]
    db, da = t["b"], t["cov"]
    ok = db > 0 and da > 0 and da >= 0.5 * db - TOL
    out["P1"] = {"status": "pass" if ok else "fail", "delta_b": db, "delta_alignment": da,
                 "delta_coverage": t["coverage"], "alignment_share": da / db if db != 0 else None,
                 "z_wl": wl["c_s"]["z"], "rule": "db > 0, da > 0, da >= 0.5 db"}

    # P2: alignment beyond chance pairing (N-DC at c_s, one-sided p with +1)
    f = dc["fixed"]["c_s"]
    out["P2"] = {"status": "pass" if f["p_high"] <= ALPHA_P2 else "fail", "real": f["real"],
                 "null_mean": f["null_mean"], "null_sd": f["null_sd"], "z": f["z"], "p": f["p_high"],
                 "secondary_grid_max_stat": {"z_peak": dc["grid"]["z_peak"], "p": dc["grid"]["p"],
                                             "null_peaks_max": dc["grid"]["null_peaks_max"],
                                             "c_peak": dc["grid"]["c_peak"]},
                 "rule": f"p <= {ALPHA_P2}"}

    # P3: the large-c deficit is coverage (N-WL at c_L)
    t = wl["c_L"]["excess"]
    db, dcov, da = t["b"], t["coverage"], t["cov"]
    z = wl["c_L"]["z"]
    ok = z <= -3 and dcov <= db + TOL and da >= -0.1 * abs(db) - TOL
    out["P3"] = {"status": "pass" if ok else "fail", "z": z, "delta_b": db, "delta_coverage": dcov,
                 "delta_alignment": da, "delta_rF": t["rF"], "delta_rB": t["rB"],
                 "z_out_strength": rec["nulls"]["out_strength"]["fixed"]["c_L"]["z"],
                 "z_in_strength": rec["nulls"]["in_strength"]["fixed"]["c_L"]["z"],
                 "rule": "z <= -3, dcov <= db, da >= -0.1 |db|"}

    # P4: no wiring beyond kept degrees (N-JD at c_s and c = 1)
    cells = rec["joint_degree"]["cells"]
    testable = {k: v["testable"] for k, v in cells.items()}
    excess = {k: bool(v["testable"] and v["p_high"] <= ALPHA_P4_CELL) for k, v in cells.items()}
    status = "not testable" if not any(testable.values()) else ("fail" if any(excess.values()) else "pass")
    out["P4"] = {"status": status,
                 "cells": {k: {"testable": v["testable"], "wiring_excess": excess[k], "real": v["real"],
                               "null_mean": v["null_mean"], "null_sd": v["null_sd"], "z": v["z"],
                               "p_high": v["p_high"], "S": v["degree_determined"]["share_equal"],
                               "n_edges": v["n_edges"]} for k, v in cells.items()},
                 "sampler": rec["joint_degree"]["sampler"],
                 "rule": f"no testable cell with p_high <= {ALPHA_P4_CELL}; not testable if no cell has spread"}

    # P5: in/out strength alignment is mostly firing frequency
    s = rec["strength"]
    ok = s["rho"] > 0 and s["p_perm"] <= ALPHA_P5 and s["rho_given_rate"] < 0.5 * s["rho"]
    out["P5"] = {"status": "pass" if ok else "fail", "rho": s["rho"], "p": s["p_perm"],
                 "rho_given_rate": s["rho_given_rate"],
                 "ratio": s["rho_given_rate"] / s["rho"] if s["rho"] else None,
                 "rule": f"rho > 0 with p <= {ALPHA_P5}, rho_given_rate < 0.5 rho"}

    for k, v in out.items():
        v["counts"] = _counts(kind, k)
    return out


def holds(scored: dict[str, dict]) -> dict:
    """Per prediction: holds if it passes on every graph it counts for (P4: pass or not testable)."""
    out = {}
    for p in ("P1", "P2", "P3", "P4", "P5"):
        rows = {name: s[p]["status"] for name, s in scored.items() if s[p]["counts"]}
        if not rows:
            out[p] = {"holds": None, "graphs": {}}
            continue
        ok = all(st in ("pass", "not testable") if p == "P4" else st == "pass" for st in rows.values())
        out[p] = {"holds": ok, "graphs": rows}
    return out


def summary_line(name: str, kind: str, sc: dict) -> str:
    def tag(p):
        return sc[p]["status"] + ("" if sc[p]["counts"] else " (reported only)")
    p1, p2, p3, p4, p5 = (sc[k] for k in ("P1", "P2", "P3", "P4", "P5"))
    return "\n".join([
        f"{name} [{kind}]",
        f"  P1 {tag('P1')}: db {p1['delta_b']:+.4f}, alignment {p1['delta_alignment']:+.4f}, "
        f"coverage {p1['delta_coverage']:+.4f} (share {p1['alignment_share'] if p1['alignment_share'] is None else round(p1['alignment_share'], 3)})",
        f"  P2 {tag('P2')}: real {p2['real']:.4f} vs N-DC {p2['null_mean']:.4f} +- {p2['null_sd']:.4f}, "
        f"z {p2['z']:.1f}, p {p2['p']:.4f}; grid max-stat z {p2['secondary_grid_max_stat']['z_peak']:.1f}, "
        f"p {p2['secondary_grid_max_stat']['p']:.4f}",
        f"  P3 {tag('P3')}: z {p3['z']:.1f}, db {p3['delta_b']:+.4f} = coverage {p3['delta_coverage']:+.4f} "
        f"+ alignment {p3['delta_alignment']:+.4f}; drF {p3['delta_rF']:+.4f}, drB {p3['delta_rB']:+.4f}",
        f"  P4 {tag('P4')}: " + "; ".join(
            f"{k}: {'testable' if v['testable'] else 'not testable'}, real {v['real']:.4f} vs "
            f"{v['null_mean']:.4f} +- {v['null_sd']:.4f}, p {v['p_high']:.4f}, S {v['S']:.4f}"
            for k, v in p4["cells"].items()),
        f"  P5 {tag('P5')}: rho {p5['rho']:+.3f} (p {p5['p']:.4f}), given rate {p5['rho_given_rate']:+.3f} "
        f"(ratio {p5['ratio'] if p5['ratio'] is None else round(p5['ratio'], 3)})",
    ])


def main(paths: list[str]):
    scored = {}
    for p in paths:
        rec = json.loads(Path(p).read_text())
        sc = score(rec)
        scored[rec["name"]] = sc
        print(summary_line(rec["name"], rec["kind"], sc))
        if rec.get("gate") is not None:
            print(f"  gate: {'pass' if rec['gate'].get('pass') else 'fail'}")
    print("\nholds over the graphs given:")
    for p, v in holds(scored).items():
        print(f"  {p}: {v['holds']} {v['graphs']}")


if __name__ == "__main__":
    main(sys.argv[1:])
