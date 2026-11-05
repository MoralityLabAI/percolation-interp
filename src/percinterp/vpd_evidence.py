"""Offline VPD receipt validator. Recorded criteria never constitute certification.

This module uses only the Python standard library and never executes model code.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "vpd-evidence-certificate/v1"
DIMENSIONS = ("unmasked_parity", "reconstruction", "convergence",
              "minimality", "adversarial_mask")
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _pointer_value(document, pointer):
    """Resolve an RFC 6901 JSON pointer without code execution."""
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("record_pointer must be a JSON pointer starting with /")
    value = document
    for escaped in pointer[1:].split("/"):
        if any(escaped[i] == "~" and (i + 1 >= len(escaped) or
                                     escaped[i + 1] not in "01")
               for i in range(len(escaped))):
            raise ValueError("invalid JSON pointer escape")
        segment = escaped.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            if segment not in value:
                raise ValueError("JSON pointer key is absent")
            value = value[segment]
        elif isinstance(value, list) and segment.isdecimal() and (
            segment == "0" or not segment.startswith("0")
        ):
            index = int(segment)
            if index >= len(value):
                raise ValueError("JSON pointer list index is absent")
            value = value[index]
        else:
            raise ValueError("JSON pointer does not select a record")
    return value


def _read_bound_artifact(root, binding):
    if not isinstance(binding, dict) or not (
        isinstance(binding.get("path"), str) and
        isinstance(binding.get("sha256"), str)
    ):
        raise ValueError("binding needs a path and sha256")
    path = (root / binding["path"]).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("artifact escapes root or is not a file")
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("artifact is too large for offline JSON receipt review")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("sha256 mismatch")
    try:
        return json.loads(raw)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("verified artifact is not valid JSON") from exc


def validate(certificate, artifact_root):
    """Review sourced, artifact-bound measurements without endorsing any producer."""
    errors = []
    if not isinstance(certificate, dict):
        errors.append("certificate must be an object")
        certificate = {}
    if certificate.get("schema") != SCHEMA:
        errors.append(f"certificate schema must be {SCHEMA}")

    protocol = certificate.get("protocol")
    if not isinstance(protocol, dict) or not all(
        isinstance(protocol.get(key), str) and protocol[key].strip()
        for key in ("id", "source", "version")
    ):
        errors.append("study protocol requires id, source and version")
        protocol = {}
    rules = protocol.get("criteria")
    if not isinstance(rules, dict):
        errors.append("protocol criteria must be an object")
        rules = {}

    bindings = certificate.get("artifacts")
    if not isinstance(bindings, dict) or not bindings:
        errors.append("at least one artifact binding is required")
        bindings = {}
    verified = {}
    root = Path(artifact_root).resolve()
    for name, binding in bindings.items():
        if not isinstance(name, str) or not name:
            errors.append("artifact names must be nonempty strings")
            continue
        try:
            verified[name] = _read_bound_artifact(root, binding)
        except (OSError, ValueError) as exc:
            errors.append(f"artifact {name}: {exc}")

    observations = certificate.get("observations", {})
    if not isinstance(observations, dict):
        errors.append("observations must be an object")
        observations = {}

    dimensions = {}
    for name in DIMENSIONS:
        state = "unknown"
        reason = "measurement is absent"
        obs = observations.get(name)
        rule = rules.get(name)
        if obs is None:
            pass
        elif not isinstance(obs, dict):
            state, reason = "invalid", "observation must be an object"
        elif name == "adversarial_mask" and obs.get("mask_selection") != "fresh_optimized_adversarial":
            reason = "random, training or unspecified masks are not fresh adversarial evaluation"
        elif not isinstance(obs.get("artifact"), str):
            reason = "observation lacks an artifact name"
        elif obs["artifact"] not in verified:
            state, reason = "invalid", "observation references an unverified artifact"
        elif not isinstance(obs.get("record_pointer"), str):
            reason = "observation lacks a bound JSON record pointer"
        else:
            try:
                record = _pointer_value(verified[obs["artifact"]], obs["record_pointer"])
                declared = {key: value for key, value in obs.items()
                            if key not in ("artifact", "record_pointer")}
                if not isinstance(record, dict) or record != declared:
                    raise ValueError("artifact record and declared observation disagree")
            except ValueError as exc:
                state, reason = "invalid", str(exc)
            else:
                if not isinstance(rule, dict) or not (
                    isinstance(rule.get("source"), str) and rule["source"].strip()
                ):
                    reason = "identified study criterion is missing"
                elif not isinstance(obs.get("metric"), str) or not obs["metric"].strip() or (
                    obs["metric"] != rule.get("metric")
                ):
                    state, reason = "invalid", "metric and protocol criterion mismatch"
                elif not all(_finite_number(v) for v in (obs.get("value"), rule.get("limit"))) or (
                    rule.get("operator") not in ("<=", ">=")
                ):
                    state, reason = "invalid", "comparison must use finite numbers and a supported operator"
                elif name == "unmasked_parity" and (
                    obs["metric"] != "max_abs_parameter_error" or
                    obs.get("mask_selection") != "all_ones_delta_excluded"
                ):
                    state, reason = "invalid", "parity requires parameter error with all-ones masks and no deltas"
                elif name == "reconstruction" and obs.get("mask_selection") not in (
                    "importance_values", "rounded_importance", "random"
                ):
                    state, reason = "invalid", "reconstruction mask policy is unrecognized"
                elif name == "convergence" and (
                    obs["metric"] != "settled_window_drift" or
                    obs.get("annealing_complete") is not True or
                    not isinstance(obs.get("settled_window"), list) or
                    len(obs["settled_window"]) < 3 or
                    not all(_finite_number(v) for v in obs["settled_window"]) or
                    not isinstance(obs.get("step_indices"), list) or
                    len(obs["step_indices"]) != len(obs["settled_window"]) or
                    not all(type(v) is int for v in obs["step_indices"]) or
                    any(a >= b for a, b in zip(obs["step_indices"], obs["step_indices"][1:]))
                ):
                    reason = "convergence needs completed annealing and a valid multi-step settled window"
                elif name == "convergence" and not math.isclose(
                    float(obs["value"]),
                    float(max(obs["settled_window"]) - min(obs["settled_window"])),
                    rel_tol=1e-9, abs_tol=1e-12
                ):
                    state, reason = "invalid", "declared window drift disagrees with artifact's window"
                elif name == "minimality" and (
                    obs["metric"] != "active_subcomponents_per_input" or
                    not isinstance(obs.get("anchor"), str) or not obs["anchor"].strip()
                ):
                    reason = "minimality needs an active-count metric and a named domain anchor"
                elif name == "adversarial_mask" and (
                    not isinstance(obs.get("evaluation_run"), str) or not obs["evaluation_run"].strip() or
                    not isinstance(obs.get("training_run"), str) or not obs["training_run"].strip() or
                    obs["evaluation_run"] == obs["training_run"] or
                    not isinstance(obs.get("budget"), dict) or not obs["budget"] or
                    obs.get("shared_across_batch") is not True
                ):
                    reason = "fresh shared-batch evaluation needs distinct runs and a recorded budget"
                else:
                    outcome = (obs["value"] <= rule["limit"]) if rule["operator"] == "<=" else (
                        obs["value"] >= rule["limit"]
                    )
                    state = "meets_stated_criterion" if outcome else "fails_stated_criterion"
                    reason = "artifact-bound recorded comparison; scientific provenance unverified"
        dimensions[name] = {"status": state, "reason": reason}

    return {
        "schema": "vpd-evidence-review/v1",
        "artifact_integrity": "verified" if not errors else "failed",
        "errors": errors,
        "dimensions": dimensions,
        "scientifically_certified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        parser.error(f"cannot parse certificate: {exc}")
    result = validate(certificate, args.artifact_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["artifact_integrity"] == "verified" and all(
        v["status"] != "invalid" for v in result["dimensions"].values()
    ) else 2)


if __name__ == "__main__":
    main()
