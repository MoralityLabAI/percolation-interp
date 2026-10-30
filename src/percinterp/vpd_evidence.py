"""Standalone VPD evidence receipt checker. Never issues scientific certification."""
import argparse
import hashlib
import json
import math
from pathlib import Path

DIMENSIONS = ("unmasked_parity", "reconstruction", "convergence", "minimality", "adversarial_mask")
METRICS = {"kl_per_position", "mse", "accuracy", "max_abs_parameter_error",
           "active_subcomponents_per_input", "settled_window_drift"}


def validate(certificate, artifact_root):
    """Compare bound observations against explicit study criteria; fail closed on bad artifacts."""
    errors = []
    dimensions = {}
    protocol = certificate.get("protocol")
    if not isinstance(protocol, dict) or not all(
        isinstance(protocol.get(k), str) and protocol[k].strip()
        for k in ("id", "source", "version")
    ):
        errors.append("identified study protocol requires id, source and version")
        protocol = {}
    rules = protocol.get("criteria", {})
    if not isinstance(rules, dict):
        errors.append("protocol criteria must be an object")
        rules = {}
    artifacts = certificate.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        errors.append("at least one bound artifact is required")
        artifacts = {}
    verified = set()
    root = Path(artifact_root).resolve()
    for name, binding in artifacts.items():
        if not isinstance(binding, dict) or not isinstance(binding.get("path"), str) or not isinstance(binding.get("sha256"), str):
            errors.append(f"artifact {name}: invalid binding")
            continue
        try:
            path = (root / binding["path"]).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("outside artifact root or not a file")
            if hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
                raise ValueError("sha256 mismatch")
            verified.add(name)
        except (OSError, ValueError) as exc:
            errors.append(f"artifact {name}: {exc}")
    observations = certificate.get("observations", {})
    if not isinstance(observations, dict):
        observations = {}
        errors.append("observations must be an object")
    for name in DIMENSIONS:
        obs = observations.get(name)
        rule = rules.get(name)
        state, reason = "unknown", "missing measurement or identified study criterion"
        if obs is not None:
            if not isinstance(obs, dict):
                state, reason = "invalid", "observation must be an object"
            elif name == "adversarial_mask" and obs.get("mask_selection") != "fresh_optimized_adversarial":
                state, reason = "unknown", "random, training, or unspecified masks are not fresh adversarial evaluation"
            elif obs.get("artifact") not in verified:
                state, reason = "invalid", "measurement lacks a verified artifact binding"
            elif not isinstance(rule, dict) or not isinstance(rule.get("source"), str) or not rule["source"].strip():
                state, reason = "unknown", "no sourced study-specific criterion"
            elif obs.get("metric") != rule.get("metric") or obs.get("metric") not in METRICS:
                state, reason = "invalid", "metric and criterion mismatch"
            elif any(type(v) not in (float, int) or not math.isfinite(v)
                     for v in (obs.get("value"), rule.get("limit"))) or rule.get("operator") not in ("<=", ">="):
                state, reason = "invalid", "invalid numerical comparison"
            elif name == "convergence" and (
                not isinstance(obs.get("settled_window"), list)
                or len(obs["settled_window"]) < 2
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in obs["settled_window"])
            ):
                state, reason = "unknown", "requires a measured settled training window"
            elif name == "minimality" and not (
                isinstance(obs.get("anchor"), str) and obs["anchor"].strip()
            ):
                state, reason = "unknown", "requires a domain-specific minimality anchor"
            elif name == "adversarial_mask" and (
                not isinstance(obs.get("evaluation_run"), str)
                or not isinstance(obs.get("training_run"), str)
                or obs["evaluation_run"] == obs["training_run"]
                or not isinstance(obs.get("budget"), dict)
                or not obs["budget"]
                or obs.get("shared_across_batch") is not True
            ):
                state, reason = "unknown", "requires distinct evaluation run, attack budget and shared-batch procedure"
            elif name == "unmasked_parity" and obs.get("mask_selection") != "all_ones_delta_excluded":
                state, reason = "invalid", "unmasked parity requires all-ones masks and excluded deltas"
            elif name == "reconstruction" and obs.get("mask_selection") not in (
                "importance_values", "rounded_importance", "random"
            ):
                state, reason = "invalid", "unknown reconstruction mask policy"
            else:
                ok = obs["value"] <= rule["limit"] if rule["operator"] == "<=" else obs["value"] >= rule["limit"]
                state = "meets_stated_criterion" if ok else "fails_stated_criterion"
                reason = "recorded comparison only; underlying measurement remains unverified"
        dimensions[name] = {"status": state, "reason": reason}
    return {"schema": "vpd-evidence-review/v1",
            "artifact_integrity": "verified" if not errors else "failed",
            "errors": errors, "dimensions": dimensions, "scientifically_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    result = validate(json.loads(args.certificate.read_text(encoding="utf8")), args.artifact_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["artifact_integrity"] == "verified"
                     and all(v["status"] != "invalid" for v in result["dimensions"].values()) else 2)


if __name__ == "__main__":
    main()
