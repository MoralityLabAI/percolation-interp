"""Deferred offline e08 driver. Never trains or contacts a cloud provider.

Native waiting consumes no model calls. The four-hour heartbeat reviews
prior results and, later, blinded context proposals; it does not poll often.
"""
from __future__ import annotations
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  PYTHONIOENCODING='utf-8')
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from percinterp.e08_io import (atomic_json, driver_lock, prerequisite_files,
                              read_json, digest, DriverAlreadyRunning, accept_reviewed_repair)

ROOT = Path(__file__).resolve().parents[1]
VRAM = Path.home() / 'AppData/Local/Programs/vramorama/vramorama.exe'
PD_PY = ROOT.parent / 'param-decomp/.venv/Scripts/python.exe'
LOCAL_PY = ROOT / '.venv/Scripts/python.exe'


def active_predecessors():
    """Inspect live handles, not stale logs, and fail closed if inspection fails."""
    if os.name != 'nt':
        raise RuntimeError('Windows process inspection is required for this local driver')
    command = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
               "Where-Object { $_.CommandLine -match 'e05_pipeline.py|e04_attributions.py|e05_analyze.py|e05_causal.py|e06_heldout.py|e06_score.py|e05b_followthrough.py' } | "
               "Select-Object -ExpandProperty ProcessId")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Cannot verify existing workers: ' + result.stderr)
    return bool(result.stdout.strip())


def stage(name, vram, python, script, *args, seed=None):
    from e05_pipeline import seed_env
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    if seed is not None:
        env.update(seed_env(seed, 20000, 32))
    # Preserve offline settings even if seed_env is extended later.
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    command = [str(VRAM), 'run', '--vram', vram, '--owner', f'e08-{name}', '--',
               str(python), '-u', str(ROOT / 'experiments' / script), *map(str, args)]
    log_path = ROOT / f'out/e08/{name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'{name}: submitting {vram} VRAMorama lease', flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def run(wait, reviewed_repair=None):
    with driver_lock(ROOT / 'out/e08/driver.lock'):
        last_log = 0.
        while True:
            ready = prerequisite_files(ROOT)
            # Scientific review/report/commit of e05b/e06 is a prerequisite too.
            reviewed = ROOT / 'results/e08/prerequisites_reviewed.json'
            if ready and reviewed.exists():
                evidence = read_json(reviewed)
                if not evidence.get('e06_commit') or not (ROOT / 'docs/e06_results.md').exists():
                    raise RuntimeError('incomplete e05b/e06 final review marker')
                if not active_predecessors():
                    break
            if not wait:
                print('e05b/e06 is not yet finished/reviewed; no e08 compute started.', flush=True)
                return
            if time.monotonic() - last_log >= 3600:
                atomic_json(ROOT / 'out/e08/status.json', {'state': 'waiting_for_e05b_e06',
                    'prerequisite_files_complete': ready, 'no_e08_model_run_started': True})
                print('Waiting for complete, reviewed e05b/e06 and live worker exit; no e08 compute started.', flush=True)
                last_log = time.monotonic()
            time.sleep(300)
        if shutil.disk_usage(ROOT).free < 1536 * 1024 ** 2:
            raise RuntimeError('e08 requires >=1.5 GiB free disk; preserve all saved artifacts')
        frozen_path = ROOT / 'out/e08/protocol.json'
        frozen = {'design_sha256': digest(ROOT / 'docs/e08_design.md'),
                  'code_sha256': {str(p.relative_to(ROOT)): digest(p)
                    for directory in ('src/percinterp', 'experiments')
                    for p in sorted((ROOT / directory).glob('e08_*.py'))}}
        if frozen_path.exists() and read_json(frozen_path) != frozen:
            if reviewed_repair is None:
                raise RuntimeError('e08 protocol/code changed after execution began; review before resuming')
            accept_reviewed_repair(frozen_path, frozen, reviewed_repair)
        atomic_json(frozen_path, frozen)
        atomic_json(ROOT / 'out/e08/status.json', {'state': 'running',
            'design_sha256': digest(ROOT / 'docs/e08_design.md')})
        if not (ROOT / 'results/e08/weighted.json').exists():
            stage('weighted', '64M', LOCAL_PY, 'e08_weighted.py')
        for seed in (0, 1):
            if not (ROOT / f'out/e08/seed{seed}/collect_done.json').exists():
                stage(f'collect-seed{seed}', '6G', PD_PY, 'e08_collect.py', '--seed', seed, seed=seed)
        stage('prepare', '64M', LOCAL_PY, 'e08_prepare.py')
        labels_path = ROOT / 'out/e08/labels.json'
        if not labels_path.exists():
            atomic_json(ROOT / 'out/e08/status.json', {'state': 'awaiting_blinded_review',
                'review_file': 'out/e08/blinded_contexts.json',
                'label_file_required': 'out/e08/labels.json',
                'message': 'Sparse agent follow-up must review discovery prefixes without component/effect data.'})
            print('Context proposals ready for blinded review; no intervention effects scored yet.', flush=True)
            return
        for seed in (0, 1):
            if not (ROOT / f'out/e08/seed{seed}/effects_done.json').exists():
                stage(f'effects-seed{seed}', '8G', PD_PY, 'e08_effects.py', '--seed', seed, seed=seed)
        stage('score', '64M', LOCAL_PY, 'e08_score.py')
        if not (ROOT / 'docs/e08_results.md').exists():
            stage('report', '64M', LOCAL_PY, 'e08_report.py')
        atomic_json(ROOT / 'out/e08/status.json', {'state': 'computed_needs_final_review',
            'results': ['results/e08/contextual.json', 'results/e08/weighted.json']})
        print('e08 computations complete; review results, write report, run checks and commit locally.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for-e05b-e06', action='store_true')
    parser.add_argument('--reviewed-repair', type=Path)
    args = parser.parse_args()
    try:
        run(args.wait_for_e05b_e06, args.reviewed_repair)
    except DriverAlreadyRunning:
        raise
    except Exception as error:
        atomic_json(ROOT / 'out/e08/status.json', {'state': 'failed',
                    'error': f'{type(error).__name__}: {error}'})
        raise
