"""No-checkpoint E10 local-runner readiness, exact stop and lease tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e10_local", ROOT / "experiments" / "e10_run_local.py")
assert SPEC and SPEC.loader
local = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local)

P_SPEC = importlib.util.spec_from_file_location("e10_prepare", ROOT / "experiments" / "e10_prepare.py")
assert P_SPEC and P_SPEC.loader
prepare = importlib.util.module_from_spec(P_SPEC)
P_SPEC.loader.exec_module(prepare)


def test_probe_rejects_missing_remote_checkpoints(tmp_path):
    result = local.probe_seed(tmp_path / "e03_seed1", tmp_path / "param_decomp_out")
    assert not result["can_prepare"]
    assert not result["can_evaluate"]
    assert any("crossref.pt" in p for p in result["missing_prepare"])


def test_probe_reports_prepare_separately_from_causal(tmp_path):
    e03, param = tmp_path / "out" / "e03_seed1", tmp_path / "param_decomp_out"
    e03.mkdir(parents=True)
    for p in ("analysis.json", "crossref.pt"):
        (e03 / p).write_text("x")
    (e03 / "vpd_run.txt").write_text("p-test-run")
    attr = param / "dataset_attributions" / "p-test-run" / "da-one"
    attr.mkdir(parents=True)
    (attr / "dataset_attributions.pt").write_bytes(b"fake")
    got = local.probe_seed(e03, param)
    assert got["can_prepare"]
    assert not got["can_evaluate"]
    (e03 / "target.txt").write_text(str(tmp_path / "target"))
    (tmp_path / "target").mkdir()
    run = param / "decompositions" / "p-test-run"
    run.mkdir(parents=True)
    (run / "model_40000.pth").write_bytes(b"fake")
    got = local.probe_seed(e03, param)
    assert got["can_evaluate"]
    assert not got["missing_causal"]


def test_triage_halts_for_identical_degree_selector():
    m = prepare.fixture(2, 12)
    m["source_type"] = "E03_VPD_SAVED_INPUTS"
    q = local.triage_manifest(m)
    assert q["n_selected"] == 2
    assert "local_degree" in q["exact_equivalent_selectors"]
    assert not q["path_specific_gain_possible"]
    m["selection"]["local_degree"]["block0"] = [1]
    m["selection"]["local_degree"]["block1"] = [1]
    # Other deterministic selectors could still coincide; test the algorithm,
    # not a hardcoded comparator name.
    q = local.triage_manifest(m)
    assert "local_degree" not in q["exact_equivalent_selectors"]
    with pytest.raises(ValueError, match="violates"):
        m["selection"]["sieve"]["block0"] = [0, 0]
        local.triage_manifest(m)


def test_inference_command_requires_safe_gpu_lease(tmp_path):
    with pytest.raises(ValueError, match="vramorama"):
        local.inference_command("python", tmp_path, tmp_path/"m", tmp_path/"c",
                                device="cuda", batch=8, n_batches=8, bootstrap=2000,
                                vram_mib=4096, vrama=None, direct_gpu=False)
    cmd = local.inference_command("python", tmp_path, tmp_path/"m", tmp_path/"c",
                                  device="cuda", batch=8, n_batches=8, bootstrap=2000,
                                  vram_mib=4096, vrama="vramorama", direct_gpu=False)
    assert cmd[:7] == ["vramorama", "run", "--vram", "4096M", "--owner",
                       "e10-selector-gate", "--"]
    direct = local.inference_command("python", tmp_path, tmp_path/"m", tmp_path/"c",
                                    device="cpu", batch=1, n_batches=2, bootstrap=200,
                                    vram_mib=4096, vrama=None, direct_gpu=False)
    assert direct[0] == "python" and "--device" in direct


def test_check_mode_leaves_saved_directories_unmodified(tmp_path):
    e1 = tmp_path / "e03_seed1"
    args = SimpleNamespace(e03_dirs=[e1], execute=False, prepare=False)
    result = local.run(args)
    assert result["decision"] == "MISSING_LOCAL_VPD_INPUTS"
    assert not e1.exists()
    assert not (tmp_path / "out").exists()


def test_stale_manifest_requires_repreparation(tmp_path):
    d = tmp_path / "saved"
    d.mkdir()
    (d / "analysis.json").write_bytes(b"original")
    (d / "crossref.pt").write_bytes(b"data")
    (d / "vpd_run.txt").write_text("p-abc")
    def sha(name):
        return local.hashlib.sha256((d / name).read_bytes()).hexdigest()
    m = {"provenance": {"vpd_run": "p-abc",
                        "analysis_file_sha256": sha("analysis.json"),
                        "crossref_file_sha256": sha("crossref.pt")}}
    local.validate_frozen_metadata(m, d)
    (d / "analysis.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen input changed"):
        local.validate_frozen_metadata(m, d)
