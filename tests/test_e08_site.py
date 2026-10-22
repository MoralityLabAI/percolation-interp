"""CPU-only assertions that the public website is only an honest calibration."""
from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e08_site_builder", ROOT / "experiments" / "e08_build_site.py")
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_bundles_three_synthetic_connections(tmp_path):
    output = tmp_path / "site"
    result = module.build_site(output)
    text = (output / "calibration_data.js").read_text(encoding="utf-8")
    assert set(result["fixtures"]) == {"planted", "flat", "tangent"}
    assert '"SYNTHETIC_CALIBRATION"' in text
    assert '"planted"' in text and '"flat"' in text and '"tangent"' in text
    assert result["fixtures"]["flat"]["p_h"] is None
    assert result["fixtures"]["planted"]["p_h"] == 0.37
    assert (output / "index.html").is_file()
    assert (output / "library" / "index.html").is_file()
    assert "Goodfire" in (output / "library" / "index.html").read_text(encoding="utf-8")
    assert (output / ".nojekyll").is_file()
    assert "calibration_data.js" in (output / "index.html").read_text(encoding="utf-8")


def test_site_does_not_overwrite_tracked_frontend():
    import pytest
    with pytest.raises(ValueError):
        module.build_site(ROOT / "ui" / "godel_globe")
