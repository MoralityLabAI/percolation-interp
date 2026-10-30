"""Synthetic VPD receipt fixtures; no real model data or scientific claims."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from percinterp.vpd_evidence import DIMENSIONS, validate


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        metrics = {
            "unmasked_parity": "max_abs_parameter_error",
            "reconstruction": "kl_per_position",
            "convergence": "settled_window_drift",
            "minimality": "active_subcomponents_per_input",
            "adversarial_mask": "kl_per_position",
        }
        limits = {
            "unmasked_parity": 0.01, "reconstruction": 0.1,
            "convergence": 0.01, "minimality": 2.0,
            "adversarial_mask": 0.2,
        }
        self.cert = {
            "schema": "vpd-evidence-certificate/v1",
            "protocol": {
                "id": "SYNTH-01", "source": "tests: synthetic fixture protocol",
                "version": "1", "criteria": {
                    key: {"source": "SYNTH-01", "metric": metrics[key],
                          "operator": "<=", "limit": limits[key]} for key in DIMENSIONS
                }
            },
            "artifacts": {"measured": {"path": "measured.json", "sha256": ""}},
            "observations": {
                key: {
                    "artifact": "measured", "record_pointer": "/observations/" + key,
                    "metric": metrics[key],
                    "value": (0.0 if key == "convergence" else
                              1.0 if key == "minimality" else 0.001),
                    "mask_selection": (
                        "all_ones_delta_excluded" if key == "unmasked_parity"
                        else "fresh_optimized_adversarial" if key == "adversarial_mask"
                        else "importance_values"
                    ),
                    "settled_window": [1.0, 1.0, 1.0],
                    "step_indices": [100, 200, 300],
                    "annealing_complete": True,
                    "anchor": "synthetic ground-truth count",
                    "training_run": "training", "evaluation_run": "fresh-eval",
                    "budget": {"steps": 8}, "shared_across_batch": True,
                } for key in DIMENSIONS
            },
        }
        self.sync()

    def sync(self):
        """Simulate a producer writing a separate bound JSON evidence artifact."""
        record = {"synthetic": True, "observations": {
            key: {field: value for field, value in obs.items()
                  if field not in ("artifact", "record_pointer")}
            for key, obs in self.cert["observations"].items()
        }}
        raw = json.dumps(record, sort_keys=True, allow_nan=False).encode("utf-8")
        (self.root / "measured.json").write_bytes(raw)
        self.cert["artifacts"]["measured"]["sha256"] = hashlib.sha256(raw).hexdigest()

    def status(self, dimension):
        return validate(self.cert, self.root)["dimensions"][dimension]["status"]

    def test_synthetic_baseline_never_certifies(self):
        result = validate(self.cert, self.root)
        self.assertEqual(result["artifact_integrity"], "verified")
        self.assertFalse(result["scientifically_certified"])
        self.assertTrue(all(value["status"] == "meets_stated_criterion"
                            for value in result["dimensions"].values()))

    def test_sparse_but_unfaithful(self):
        self.cert["observations"]["minimality"]["value"] = 1
        self.cert["observations"]["adversarial_mask"]["value"] = 3.0
        self.sync()
        self.assertEqual(self.status("minimality"), "meets_stated_criterion")
        self.assertEqual(self.status("adversarial_mask"), "fails_stated_criterion")

    def test_dense_but_faithful(self):
        self.cert["observations"]["minimality"]["value"] = 100
        self.sync()
        self.assertEqual(self.status("minimality"), "fails_stated_criterion")
        self.assertEqual(self.status("adversarial_mask"), "meets_stated_criterion")

    def test_unconverged_short_window_unknown(self):
        self.cert["observations"]["convergence"]["settled_window"] = [0.1]
        self.cert["observations"]["convergence"]["step_indices"] = [300]
        self.sync()
        self.assertEqual(self.status("convergence"), "unknown")

    def test_unconverged_large_drift_fails(self):
        obs = self.cert["observations"]["convergence"]
        obs["settled_window"] = [1.0, 1.05, 1.10]
        obs["value"] = 0.10
        self.sync()
        self.assertEqual(self.status("convergence"), "fails_stated_criterion")

    def test_convergence_faked_drift_invalid(self):
        self.cert["observations"]["convergence"]["settled_window"] = [1.0, 1.1, 1.2]
        self.sync()
        self.assertEqual(self.status("convergence"), "invalid")

    def test_convergence_annealing_incomplete_unknown(self):
        self.cert["observations"]["convergence"]["annealing_complete"] = False
        self.sync()
        self.assertEqual(self.status("convergence"), "unknown")

    def test_convergence_repeated_steps_unknown(self):
        self.cert["observations"]["convergence"]["step_indices"] = [100, 100, 300]
        self.sync()
        self.assertEqual(self.status("convergence"), "unknown")

    def test_metric_mismatch(self):
        self.cert["observations"]["reconstruction"]["metric"] = "mse"
        self.sync()
        self.assertEqual(self.status("reconstruction"), "invalid")

    def test_non_parameter_parity_invalid(self):
        self.cert["protocol"]["criteria"]["unmasked_parity"]["metric"] = "mse"
        self.cert["observations"]["unmasked_parity"]["metric"] = "mse"
        self.sync()
        self.assertEqual(self.status("unmasked_parity"), "invalid")

    def test_random_mask_cannot_count_as_adversarial(self):
        self.cert["observations"]["adversarial_mask"]["mask_selection"] = "random"
        self.sync()
        self.assertEqual(self.status("adversarial_mask"), "unknown")

    def test_training_adversary_cannot_count_as_fresh(self):
        self.cert["observations"]["adversarial_mask"]["evaluation_run"] = "training"
        self.sync()
        self.assertEqual(self.status("adversarial_mask"), "unknown")

    def test_missing_adversarial_budget_unknown(self):
        self.cert["observations"]["adversarial_mask"]["budget"] = {}
        self.sync()
        self.assertEqual(self.status("adversarial_mask"), "unknown")

    def test_missing_minimality_anchor_unknown(self):
        self.cert["observations"]["minimality"]["anchor"] = ""
        self.sync()
        self.assertEqual(self.status("minimality"), "unknown")

    def test_artifact_mutation_rejects_hash(self):
        (self.root / "measured.json").write_text('{"changed":true}')
        review = validate(self.cert, self.root)
        self.assertEqual(review["artifact_integrity"], "failed")
        self.assertEqual(review["dimensions"]["reconstruction"]["status"], "invalid")

    def test_artifact_record_binding_rejects_fabricated_observation(self):
        self.cert["observations"]["reconstruction"]["value"] = 0.0000001
        self.assertEqual(self.status("reconstruction"), "invalid")

    def test_missing_record_pointer_unknown(self):
        del self.cert["observations"]["minimality"]["record_pointer"]
        self.assertEqual(self.status("minimality"), "unknown")

    def test_hash_matched_unrelated_artifact_invalid(self):
        data = b'{"elsewhere":42}'
        (self.root / "measured.json").write_bytes(data)
        self.cert["artifacts"]["measured"]["sha256"] = hashlib.sha256(data).hexdigest()
        self.assertEqual(self.status("reconstruction"), "invalid")

    def test_path_traversal_rejected(self):
        self.cert["artifacts"]["measured"]["path"] = "../outside"
        self.assertEqual(validate(self.cert, self.root)["artifact_integrity"], "failed")

    def test_missing_measurement_unknown(self):
        del self.cert["observations"]["minimality"]
        self.assertEqual(self.status("minimality"), "unknown")

    def test_missing_protocol_criterion_unknown(self):
        del self.cert["protocol"]["criteria"]["reconstruction"]
        self.assertEqual(self.status("reconstruction"), "unknown")

    def test_bad_certificate_schema_fails_integrity(self):
        self.cert["schema"] = "unrecognized"
        self.assertEqual(validate(self.cert, self.root)["artifact_integrity"], "failed")

    def test_boolean_value_is_invalid(self):
        self.cert["observations"]["reconstruction"]["value"] = True
        self.sync()
        self.assertEqual(self.status("reconstruction"), "invalid")

    def test_cli_can_run_standalone(self):
        receipt = self.root / "certificate.json"
        receipt.write_text(json.dumps(self.cert), encoding="utf-8")
        source = Path(__file__).resolve().parents[1] / "src/percinterp/vpd_evidence.py"
        result = subprocess.run(
            [sys.executable, str(source), str(receipt),
             "--artifact-root", str(self.root)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["scientifically_certified"])


if __name__ == "__main__":
    unittest.main()
