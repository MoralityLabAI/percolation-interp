"""Deterministic synthetic VPD evidence review checks; never real model data."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from percinterp.vpd_evidence import DIMENSIONS, validate


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        artifact = self.root / "synthetic.json"
        artifact.write_text('{"synthetic":true}')
        metrics = {"unmasked_parity": "max_abs_parameter_error",
                   "reconstruction": "kl_per_position",
                   "convergence": "settled_window_drift",
                   "minimality": "active_subcomponents_per_input",
                   "adversarial_mask": "kl_per_position"}
        limits = {"unmasked_parity": 0.01, "reconstruction": 0.1,
                  "convergence": 0.01, "minimality": 2.0, "adversarial_mask": 0.2}
        self.cert = {
            "protocol": {"id": "SYNTH-01", "source": "synthetic fixture protocol",
                         "version": "1", "criteria": {
                             k: {"source": "SYNTH-01", "metric": metrics[k],
                                 "operator": "<=", "limit": limits[k]} for k in DIMENSIONS}},
            "artifacts": {"test": {"path": "synthetic.json",
                                   "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}},
            "observations": {k: {
                "artifact": "test", "metric": metrics[k], "value": 0.001,
                "mask_selection": ("all_ones_delta_excluded" if k == "unmasked_parity"
                                   else "fresh_optimized_adversarial" if k == "adversarial_mask"
                                   else "importance_values"),
                "settled_window": [0.001, 0.001], "anchor": "synthetic ground truth",
                "training_run": "train", "evaluation_run": "fresh-eval",
                "budget": {"steps": 8}, "shared_across_batch": True
            } for k in DIMENSIONS}
        }

    def review(self):
        return validate(self.cert, self.root)

    def status(self, key):
        return self.review()["dimensions"][key]["status"]

    def test_synthetic_receipt_never_certifies(self):
        result = self.review()
        self.assertFalse(result["scientifically_certified"])
        self.assertTrue(all(v["status"] == "meets_stated_criterion"
                            for v in result["dimensions"].values()))

    def test_sparse_but_unfaithful(self):
        self.cert["observations"]["minimality"]["value"] = 1
        self.cert["observations"]["adversarial_mask"]["value"] = 3
        self.assertEqual(self.status("minimality"), "meets_stated_criterion")
        self.assertEqual(self.status("adversarial_mask"), "fails_stated_criterion")

    def test_dense_but_faithful(self):
        self.cert["observations"]["minimality"]["value"] = 100
        self.assertEqual(self.status("minimality"), "fails_stated_criterion")
        self.assertEqual(self.status("adversarial_mask"), "meets_stated_criterion")

    def test_unconverged(self):
        self.cert["observations"]["convergence"]["settled_window"] = [0.1]
        self.assertEqual(self.status("convergence"), "unknown")

    def test_metric_mismatch(self):
        self.cert["observations"]["reconstruction"]["metric"] = "mse"
        self.assertEqual(self.status("reconstruction"), "invalid")

    def test_random_mask_is_not_adversarial(self):
        self.cert["observations"]["adversarial_mask"]["mask_selection"] = "random"
        self.assertEqual(self.status("adversarial_mask"), "unknown")

    def test_training_adversary_not_fresh(self):
        self.cert["observations"]["adversarial_mask"]["evaluation_run"] = "train"
        self.assertEqual(self.status("adversarial_mask"), "unknown")

    def test_artifact_binding_tampered(self):
        (self.root / "synthetic.json").write_text("changed")
        self.assertEqual(self.review()["artifact_integrity"], "failed")
        self.assertEqual(self.status("minimality"), "invalid")

    def test_path_traversal_rejected(self):
        self.cert["artifacts"]["test"]["path"] = "../outside"
        self.assertEqual(self.review()["artifact_integrity"], "failed")

    def test_missing_measurement_unknown(self):
        del self.cert["observations"]["minimality"]
        self.assertEqual(self.status("minimality"), "unknown")

    def test_missing_protocol_cutoff_unknown(self):
        del self.cert["protocol"]["criteria"]["reconstruction"]
        self.assertEqual(self.status("reconstruction"), "unknown")


if __name__ == "__main__":
    unittest.main()
