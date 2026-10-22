"""Optional PyTorch CPU integration test using a minimal fake VPD model.

This checks selection-mask plumbing, original-row exclusion, duplicate-set
inference optimization, paired sequence scoring and provenance enforcement.
It deliberately makes no claim about actual param-decomp checkpoint semantics.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e10_causal", ROOT / "experiments" / "e10_e03_causal.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

P_SPEC = importlib.util.spec_from_file_location("e10_prepare", ROOT / "experiments" / "e10_prepare.py")
assert P_SPEC and P_SPEC.loader
prep = importlib.util.module_from_spec(P_SPEC)
P_SPEC.loader.exec_module(prep)


def _install_fake_modules(monkeypatch):
    e03 = ModuleType("e03_common")
    def loader(split, batch_size, seed):
        assert split == "test"
        # First eight seed-4 batches are the previously seen rows.
        for batch in range(16):
            base = seed * 1000 + batch * batch_size
            yield torch.stack([torch.tensor([base + i, base + i + 1, base + i + 2,
                                             base + i + 3, base + i + 4, base + i + 5,
                                             base + i + 6, base + i + 7])
                               for i in range(batch_size)])
    class Target:
        def to(self, d): return self
        def eval(self): return self
        def __call__(self, idx):
            # Target uses both features in each module.
            base = torch.ones((*idx.shape, 3), dtype=torch.float32)
            base[..., 0] = 5
            base[..., 1] = 1
            base[..., 2] = -3
            return (base,)
    e03.token_loader = loader
    e03.load_target = lambda device: Target()

    root = ModuleType("param_decomp")
    models = ModuleType("param_decomp.models")
    cm_mod = ModuleType("param_decomp.models.component_model")
    comp_mod = ModuleType("param_decomp.models.components")
    class Info:
        config = SimpleNamespace(sampling="fake_sampling")
        @classmethod
        def from_path(cls, run):
            assert run == "fake_vpd"
            return cls()
    class Model:
        def __init__(self):
            self.target_module_paths = ["block0", "block1"]
            self.components = {m: SimpleNamespace(V=torch.ones((1, 2))) for m in self.target_module_paths}
            self.calls = 0
        @classmethod
        def from_run_info(cls, info): return cls()
        def to(self, d): return self
        def eval(self): return self
        def __call__(self, idx, mask_infos=None, cache_type=None):
            if cache_type == "input":
                return SimpleNamespace(cache={"input": "toy"})
            self.calls += 1
            strength = torch.zeros(idx.shape, dtype=torch.float32)
            for m in self.target_module_paths:
                mask = torch.ones((*idx.shape, 2)) if mask_infos is None else mask_infos[m]
                strength += 0.6 * mask[..., 0] + 0.25 * mask[..., 1]
            output = torch.stack([strength * 3 + 1., strength * 0 + 1.,
                                  strength * -2 - 1.], dim=-1)
            return output
        def calc_causal_importances(self, *, pre_weight_acts, detach_inputs, sampling):
            # real code multiplies [B,T,C] by static [C] member masks.
            assert sampling == "fake_sampling"
            b, t = 2, 8  # for the integration test only
            lower = {m: torch.ones((b, t, 2), dtype=torch.float32)
                     for m in self.target_module_paths}
            return SimpleNamespace(lower_leaky=lower)
    cm_mod.ParamDecompRunInfo = Info
    cm_mod.ComponentModel = Model
    comp_mod.make_mask_infos = lambda ci: ci
    for key, mod in (("e03_common", e03), ("param_decomp", root),
                     ("param_decomp.models", models),
                     ("param_decomp.models.component_model", cm_mod),
                     ("param_decomp.models.components", comp_mod)):
        monkeypatch.setitem(sys.modules, key, mod)


def _manifest(tmp_path):
    for name in ("analysis.json", "crossref.pt"):
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "vpd_run.txt").write_text("fake_vpd")
    m = prep.fixture(n_random=3, seed=8)
    m["source_type"] = "E03_VPD_SAVED_INPUTS"
    m["provenance"] = {
        "source_dir": str(tmp_path), "vpd_run": "fake_vpd",
        "analysis_file_sha256": runner.sha256_file(tmp_path / "analysis.json"),
        "crossref_file_sha256": runner.sha256_file(tmp_path / "crossref.pt"),
    }
    return m


def test_e10_runner_end_to_end_with_mocked_ci(tmp_path, monkeypatch):
    _install_fake_modules(monkeypatch)
    manifest = _manifest(tmp_path)
    score = runner.evaluate(manifest=manifest, e03_dir=tmp_path, device="cpu",
                            batch=2, n_batches=4, eval_seed=102, bootstrap=200)
    assert score["schema"] == "percinterp.e10.causal.v1"
    assert score["eval_n_sequences"] == 8
    assert score["exact_prior_overlap"] == 0
    assert score["score"]["n_sequences"] == 8
    assert score["deduplicated_selection_aliases"]["local_degree"] == "sieve"
    assert score["score"]["comparators"]["local_degree"]["exact_same_kl_vectors"]
    assert len(score["per_sequence_kl"]["sieve"]) == 8
    assert score["unique_model_evaluations_per_batch"] < len(manifest["selection"]) + 2


def test_synthetic_manifest_is_prohibited_from_real_runner(tmp_path):
    m = prep.fixture(n_random=1, seed=1)
    with pytest.raises(ValueError, match="real E03 selector manifest"):
        runner.validate_manifest(m, tmp_path)
