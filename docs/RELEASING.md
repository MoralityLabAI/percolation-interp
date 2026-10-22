# PercInterp PyPI release readiness

The source distribution is called `percinterp`. Its current project metadata
uses version 0.1.0. A PyPI release has neither been attempted nor verified.

## Reproduce packaging locally

```sh
python -m pip install -U build twine pytest
python -m pytest -q tests/test_interop.py tests/test_indexed_archive.py tests/test_reference_benchmark.py tests/test_measure.py tests/test_sieve.py tests/test_theory.py
python -m build
python -m twine check dist/*
python -m pip install dist/*.whl
python -c "from percinterp import from_attribution_arrays, save_graph_npz, load_graph_npz; print('imports OK')"
```

The workflow `.github/workflows/distribution.yml` runs packaging checks
and uploads wheel/sdist as private workflow artifacts. It intentionally never
publishes to PyPI and has no PyPI password, API token or OIDC publisher action.

Before a first PyPI release, verify name availability on PyPI, the selected
version, package contents (no checkpoints/secrets), dependency compatibility,
and the example with Goodfire native ComponentKey format. Register a PyPI
Trusted Publisher for the exact GitHub repository/workflow only after review.
Cut a versioned release and validate pip installation from the published wheel.
The model-level E10 causal experiment is independent of utility-package readiness.
