"""E10 paired causal-effect scoring; independent of torch and checkpoints."""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .selector_gate import DETERMINISTIC


MIN_EXTRA_GAP = 0.02
QUALITY_CI_TO_ZERO_RATIO = 0.25  # Explicit E10 toy-model engineering gate.


def score_sequential_kl(values: Mapping[str, object], *,
                        n_bootstrap: int = 2000, seed: int = 20261023,
                        same_selection: set[str] | None = None) -> dict:
    """Paired sequence-level bootstrap of comparator KL minus sieve KL.

    A positive delta is an improvement from using the sieve. All ablation
    methods must be evaluated on the *same* held-out sequence in each row.
    The bootstrap unit is a 512-token input sequence, not an individual token.
    This is an engineering uncertainty estimate; documents may overlap.
    """
    required = set(DETERMINISTIC) | {"all", "none"}
    if not required.issubset(values):
        raise ValueError(f"missing selectors: {sorted(required - set(values))}")
    if n_bootstrap < 100 or n_bootstrap > 100000:
        raise ValueError("n_bootstrap must be between 100 and 100000")
    data = {name: np.asarray(v, dtype=np.float64) for name, v in values.items()}
    sizes = {len(v) for v in data.values()}
    if len(sizes) != 1 or next(iter(sizes)) < 2:
        raise ValueError("need matching paired arrays of at least two sequences")
    n = next(iter(sizes))
    if any(v.ndim != 1 or not np.isfinite(v).all() for v in data.values()):
        raise ValueError("each KL vector must be finite and one-dimensional")
    same_selection = set() if same_selection is None else set(same_selection)
    if not same_selection.issubset(set(DETERMINISTIC) - {"sieve"}):
        raise ValueError("same_selection must contain only nonsieve deterministic names")
    for name in same_selection:
        if not np.array_equal(data[name], data["sieve"]):
            raise ValueError(f"same selected components produced different KL values: {name}")
    mean = {k: float(v.mean()) for k, v in data.items()}
    gap = mean["none"] - mean["all"]
    if gap <= 0:
        return {"status": "INVALID_ABLATION_GAP", "n_sequences": n,
                "kl_means": mean, "ablation_gap": gap, "comparators": {}}
    ratio = mean["all"] / mean["none"] if mean["none"] > 0 else math.inf
    quality_pass = bool(ratio <= QUALITY_CI_TO_ZERO_RATIO)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n), dtype=np.int64)
    sieve = data["sieve"]
    base_gap_boot = (data["none"][indices] - data["all"][indices]).mean(1)
    safe_gap = np.where(base_gap_boot > 0, base_gap_boot, np.nan)
    comparisons = {}
    for name in DETERMINISTIC:
        if name == "sieve":
            continue
        deltas = data[name] - sieve
        delta_samples = deltas[indices].mean(1)
        extra_samples = delta_samples / safe_gap
        valid = extra_samples[np.isfinite(extra_samples)]
        if not len(valid):
            raise ValueError("bootstrap produced no valid positive ablation gaps")
        extra = float(mean[name] - mean["sieve"]) / gap
        low, high = np.quantile(valid, [0.025, 0.975])
        comparisons[name] = {
            "kl_delta_baseline_minus_sieve": float(deltas.mean()),
            "extra_gap_closed": extra,
            "extra_gap_ci_95": [float(low), float(high)],
            "bootstrap_probability_sieve_better": float((valid > 0).mean()),
            "passes_practical_gate": bool(extra >= MIN_EXTRA_GAP and low > 0),
            "per_sequence_win_rate": float((deltas > 0).mean()),
            "exact_same_kl_vectors": bool(np.array_equal(data[name], sieve)),
            "certified_identical_selector": name in same_selection,
            "ci_excludes_two_percent_advantage": bool(high < MIN_EXTRA_GAP),
        }
    all_pass = quality_pass and all(
        m["passes_practical_gate"] for m in comparisons.values())
    if not quality_pass:
        decision_class = "DECOMPOSITION_QUALITY_FAIL"
    elif all_pass:
        decision_class = "INCREMENTAL_ADVANTAGE_DETECTED"
    elif same_selection:
        decision_class = "EXACT_DEGREE_OR_OTHER_SELECTOR_EQUIVALENCE"
    elif any(c["ci_excludes_two_percent_advantage"] for c in comparisons.values()):
        decision_class = "PRACTICAL_ADVANTAGE_EXCLUDED_BY_BOOTSTRAP"
    else:
        decision_class = "INCONCLUSIVE_GATE_NOT_MET"
    random_keys = sorted(k for k in data if k.startswith("random_"))
    random_kl = np.array([mean[k] for k in random_keys], dtype=float)
    return {
        "status": "ALL_BASELINES_PASS" if all_pass else "NO_INCREMENTAL_PERCO_ADVANTAGE",
        "n_sequences": n,
        "kl_means": mean,
        "ablation_gap": float(gap),
        "ci_over_zero_ratio": float(ratio),
        "quality_gate_pass": quality_pass,
        "decision_class": decision_class,
        "certified_equivalent_selectors": sorted(same_selection),
        "comparators": comparisons,
        "random_kl_median": float(np.median(random_kl)) if len(random_kl) else None,
        "random_kl_best": float(np.min(random_kl)) if len(random_kl) else None,
        "bootstrap_unit": "sequence, grouped over 512 tokens",
        "bootstrap_seed": int(seed),
        "bootstrap_draws": int(n_bootstrap),
        "gate_extra_gap_fraction": MIN_EXTRA_GAP,
        "caveat": "Retrospective target/decomposition selection and document-level dependence "
                  "mean this is a within-target diagnostic, not population-level replication.",
    }


def combine_seed_results(records: list[dict]) -> dict:
    """Score already-frozen seed results without tuning selectors."""
    if not records:
        raise ValueError("at least one E10 score required")
    manifest_ids = [r.get("manifest_sha256") for r in records]
    if any(not x for x in manifest_ids) or len(set(manifest_ids)) != len(records):
        raise ValueError("missing or duplicated frozen manifest IDs")
    per_seed = [{"manifest_sha256": r["manifest_sha256"],
                 "status": r["score"]["status"],
                 "quality_gate_pass": r["score"].get("quality_gate_pass", False),
                 "decision_class": r["score"].get("decision_class", "UNKNOWN"),
                 "comparison": r["score"].get("comparators", {})}
                for r in records]
    all_pass = len(per_seed) >= 2 and all(
        r["status"] == "ALL_BASELINES_PASS" for r in per_seed)
    return {
        "status": "WITHIN_TARGET_SEED_GATE_PASS" if all_pass
                  else "WITHIN_TARGET_SEED_GATE_NOT_MET",
        "n_seed_results": len(per_seed),
        "contains_exact_equivalence": any(
            r["decision_class"] == "EXACT_DEGREE_OR_OTHER_SELECTOR_EQUIVALENCE"
            for r in per_seed),
        "interpretation": ("Within-target seed gate passed" if all_pass else
            "At least one decomposition failed the pre-specified gate; consult "
            "per-seed decision_class to distinguish falsification from inconclusive power"),
        "per_seed": per_seed,
        "limitation": "Multiple decomposition seeds of the same target model are not "
                      "independent model replications. Broader applicability still requires "
                      "another faithful target and held-out task.",
    }
