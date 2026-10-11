#!/usr/bin/env bash
# Environment for a Claude Code cloud session (Ubuntu, CPU only). Run from the repo root.
set -euo pipefail
python3 -m venv .venv
# CPU wheels avoid ~2.5 GB of unused CUDA libraries; fall back to PyPI if
# download.pytorch.org is not on the environment's network allowlist.
.venv/bin/pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch \
  || .venv/bin/pip install --quiet torch
.venv/bin/pip install --quiet -e ".[dev]"
.venv/bin/python -c "import torch, percinterp; print('torch', torch.__version__, 'threads', torch.get_num_threads())"
mkdir -p logs results
