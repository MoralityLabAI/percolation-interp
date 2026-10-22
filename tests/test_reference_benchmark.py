"""Exact agreement with SciPy's independently constructed graph traversal."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "percinterp_reference_bench", ROOT / "benchmarks" / "profile_vs_resweep.py",
)
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_profile_matches_threshold_by_threshold_bfs():
    for width in (6, 16):
        data = bench.benchmark(width=width, points=13, repeats=1)
        assert data["max_curve_difference"] == 0
        assert data["nodes"] > 0 and data["edges"] > 0
