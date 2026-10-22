"""E09 retrospective utility audit using only committed PercInterp results.

This is exploratory reanalysis, not a pre-registered or independent validation.
No VPD checkpoints, GPUs, web calls, or training are required.

Usage:
    python experiments/e09_utility_audit.py
    python experiments/e09_utility_audit.py --output out/e09_utility.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAUSAL_FILES = (
    ("toy_seed0", "results/e03/causal.json", "P4"),
    ("toy_seed1", "results/e03_seed1/causal.json", "P4"),
    ("toy_seed2", "results/e03_seed2/causal.json", "P4"),
    ("qwen_seed0", "results/e04/causal.json", None),
    ("qwen_seed1", "results/e04_seed1/causal.json", None),
)
E06_FILES = (
    "results/e06/heldout_e03_seed1.json",
    "results/e06/heldout_e03_seed2.json",
)


def read_json(path: Path) -> dict:
    # Historical JSON contains nonstandard NaN in some unused diagnostic cells.
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: None)


def average(values) -> float:
    seq = list(values)
    if not seq:
        raise ValueError("average needs at least one observation")
    return float(sum(seq) / len(seq))


def linear_interpolate(xs, ys, x: float) -> float:
    """Piecewise-linear template. Endpoints are clamped; repeated x takes first."""
    if len(xs) != len(ys) or not xs:
        raise ValueError("xs/ys length mismatch or empty")
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(1, len(xs)):
        if xs[i] >= x:
            a, b = xs[i - 1], xs[i]
            if a == b:
                return float((ys[i - 1] + ys[i]) / 2)
            return float(ys[i - 1] + (ys[i] - ys[i - 1]) * (x - a) / (b - a))
    raise AssertionError("ordered interpolation exhausted without bracket")


def axis_coordinates(run: dict, axis: str) -> list[float]:
    f = run["fidelity_final"]
    if axis == "retained_fraction":
        return [math.log(c / run["width"]) for c in f["c"]]
    if axis == "kept_outdegree":
        return [math.log(c) for c in f["c"]]
    if axis == "backbone_fraction":
        return list(map(float, f["bb_edge"]))
    raise ValueError(f"unknown axis: {axis}")


def template(runs: list[dict], axis: str, x: float, target: str) -> float:
    return average(linear_interpolate(axis_coordinates(run, axis),
                                      run["fidelity_final"][target], x)
                   for run in runs)


def _measure_group(holdout: str, runs: list[dict]) -> dict:
    held = [r for r in runs if f"{r['depth']}/{r['width']}" == holdout]
    train = [r for r in runs if f"{r['depth']}/{r['width']}" != holdout]
    if not held or not train:
        raise ValueError("need held-out and training run groups")

    # Single coefficient: does backbone fraction predict recovery *conditional
    # on retained fraction*? Fit ONLY to training groups.
    numerator = denominator = 0.0
    for r in train:
        for i, x in enumerate(axis_coordinates(r, "retained_fraction")):
            dy = r["fidelity_final"]["rec_all"][i] - template(
                train, "retained_fraction", x, "rec_all")
            db = r["fidelity_final"]["bb_edge"][i] - template(
                train, "retained_fraction", x, "bb_edge")
            numerator += dy * db
            denominator += db * db
    beta = numerator / denominator if denominator else 0.0

    axes = ("retained_fraction", "kept_outdegree", "backbone_fraction",
            "retained_plus_backbone")
    deviations = {axis: [] for axis in axes}
    transition = {axis: [] for axis in axes}
    for r in held:
        f = r["fidelity_final"]
        for i, y in enumerate(f["rec_all"]):
            preds = {}
            for axis in axes[:-1]:
                x = axis_coordinates(r, axis)[i]
                preds[axis] = template(train, axis, x, "rec_all")
            xp = axis_coordinates(r, "retained_fraction")[i]
            xb = f["bb_edge"][i]
            preds["retained_plus_backbone"] = preds["retained_fraction"] + beta * (
                xb - template(train, "retained_fraction", xp, "bb_edge"))
            for axis, pred in preds.items():
                error = abs(y - pred)
                deviations[axis].append(error)
                if 0.05 <= y <= 0.95:
                    transition[axis].append(error)
    return {
        "group": holdout,
        "held_out_runs": len(held),
        "train_runs": len(train),
        "beta_backbone_conditional_on_retained": beta,
        "mean_absolute_error": {axis: average(deviations[axis]) for axis in axes},
        "transition_mae": {
            axis: average(transition[axis]) if transition[axis] else None
            for axis in axes
        },
        "n_transition_points": len(transition["retained_fraction"]),
    }


def evaluate_fidelity_transfer(summary: dict) -> dict:
    runs = [r for r in summary["runs"] if r["grokked"]
            and r.get("fidelity_final") is not None]
    groups = list(dict.fromkeys(f"{r['depth']}/{r['width']}" for r in runs))
    if len(groups) < 2:
        raise ValueError("need at least two independent configuration groups")
    for r in runs:
        for axis in ("retained_fraction", "kept_outdegree", "backbone_fraction"):
            xs = axis_coordinates(r, axis)
            if any(b < a for a, b in zip(xs, xs[1:])):
                raise ValueError(f"non-monotone {axis} in {r['name']}")
    scored = [_measure_group(group, runs) for group in groups]
    axes = tuple(scored[0]["mean_absolute_error"])
    macro = {axis: average(row["mean_absolute_error"][axis] for row in scored)
             for axis in axes}
    mid = {axis: average(row["transition_mae"][axis] for row in scored
                         if row["transition_mae"][axis] is not None) for axis in axes}
    return {
        "protocol": "leave-one-(depth,width)-configuration-out; mean of interpolated "
                    "training-run recovery curves; absolute error on held-out points",
        "target": "mean logit-metric recovery rec_all after edge sieve",
        "population": "13 grokked runs only, as selected by E02 design",
        "n_runs": len(runs), "n_groups": len(groups), "groups": scored,
        "macro_mae": macro, "macro_transition_mae": mid,
        "backbone_addon_worse_in_groups": sum(
            row["mean_absolute_error"]["retained_plus_backbone"] >
            row["mean_absolute_error"]["retained_fraction"] for row in scored),
        "caveats": [
            "Retrospective exploratory comparison on one modular-addition task",
            "Decomposition seeds share a training setup; five depth-width configurations "
            "are the evaluation units",
            "Templates clamp outside training-axis support and duplicate x uses first value",
            "The retained-plus-backbone slope is fitted on training groups only",
            "Transition points use held-out target 0.05<=recovery<=0.95 for reporting "
            "rather than for model selection",
        ],
    }


def evaluate_causal_cases(root: Path) -> list[dict]:
    rows = []
    for label, path, key in CAUSAL_FILES:
        doc = read_json(root / path)
        case = doc[key] if key else doc
        backbone, top, none, all_ = (
            float(case[k]) for k in ("backbone", "top_ci", "none", "all"))
        if none <= all_:
            raise ValueError(f"unexpected KL baseline order: {label}")
        rows.append({
            "case": label, "source": path,
            "kl_backbone": backbone, "kl_top_importance": top,
            "kl_none": none, "kl_all": all_,
            "benefit_vs_importance_nats": top - backbone,
            "additional_gap_closed_fraction": (top - backbone) / (none - all_),
        })
    return rows


def evaluate_degree_null(root: Path) -> list[dict]:
    rows = []
    for path in E06_FILES:
        d = read_json(root / path)
        cell = d["scores"]["P4"]["cells"]["c_s"]
        rows.append({
            "source": path, "cell": "c_s = 0.1",
            "degree_determined_share": float(cell["S"]),
            "wiring_excess_p": float(cell["p_high"]),
            "wiring_excess_by_registered_threshold": bool(cell["wiring_excess"]),
        })
    return rows


def build_report(root: Path = ROOT) -> dict:
    transfer = evaluate_fidelity_transfer(read_json(
        root / "results/e02_depth/report_summary.json"))
    causal = evaluate_causal_cases(root)
    degree = evaluate_degree_null(root)
    toy = [c for c in causal if c["case"].startswith("toy_")]
    qwen = [c for c in causal if c["case"].startswith("qwen_")]
    return {
        "version": "e09-retrospective-v1",
        "design": "Post-hoc scientific utility audit of committed E02/E03/E04/E06 results",
        "fidelity_transfer": transfer, "causal_cases": causal,
        "degree_null_cases": degree,
        "decision": {
            "label": "NARROW_RETAIN",
            "reason": "Toy-model component subset selection beats top-CI in three "
                      "decomposition seeds. Universal fidelity transition, transfer "
                      "to Qwen and wiring beyond degree-only nulls are unsupported.",
            "toy_selection_additional_gap_fraction_range": [
                min(x["additional_gap_closed_fraction"] for x in toy),
                max(x["additional_gap_closed_fraction"] for x in toy),
            ],
            "qwen_selection_additional_gap_fraction_range": [
                min(x["additional_gap_closed_fraction"] for x in qwen),
                max(x["additional_gap_closed_fraction"] for x in qwen),
            ],
            "condition_for_expansion": "On held-out target prompts, a fixed-budget "
                "sieve selector must beat a degree/local-coverage-matched selector, "
                "top mean-CI, and strongest-attribution selector, with equal "
                "module budgets and matched activation frequency.",
            "retirement_trigger": "No reproducible held-out KL or behavior recovery "
                "advantage over those simpler selectors on an adequately "
                "faithful VPD decomposition.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = build_report(args.root)
    body = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    m = result["fidelity_transfer"]["macro_mae"]
    print(f"E09 held-out configurations={result['fidelity_transfer']['n_groups']}, "
          f"grokked runs={result['fidelity_transfer']['n_runs']}")
    print("MAE retained fraction / outdegree / backbone / retained+backbone: "
          + " / ".join(f"{m[k]:.5f}" for k in (
              "retained_fraction", "kept_outdegree", "backbone_fraction",
              "retained_plus_backbone")))
    print(f"Decision: {result['decision']['label']}")
    if args.output:
        print(f"Audit output: {args.output}")


if __name__ == "__main__":
    main()
