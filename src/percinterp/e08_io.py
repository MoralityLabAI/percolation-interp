"""Atomic e08 artifacts, bounded completion checks, and a single driver lock."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path


class DriverAlreadyRunning(OSError):
    """Another e08 driver owns the OS lock; do not overwrite its status."""


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=1, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prerequisite_files(root):
    """Require all stages appropriate to each gate plus successful e06 scoring."""
    root = Path(root)
    try:
        for seed in (0, 1):
            out = root / f'out/e05b_seed{seed}'
            gate = read_json(out / 'gate.json')
            if not isinstance(gate.get('pass'), bool):
                return False
            files = ['stats.pt', 'dataset_attributions.pt']
            if gate['pass']:
                files.extend(['analysis.json', 'causal.json'])
            if any(not (out / name).is_file() or (out / name).stat().st_size == 0 for name in files):
                return False
            for name in ('analysis.json', 'causal.json') if gate['pass'] else ():
                read_json(out / name)
        for name in ('e03_seed1', 'e03_seed2', 'e05b_seed0', 'e05b_seed1'):
            read_json(root / f'results/e06/heldout_{name}.json')
        score = root / 'out/e05b/e06_score.log'
        if not score.is_file() or score.stat().st_size == 0:
            return False
        return 'All analysis stages finished; reports and local commit need final review' in (
            root / 'out/e05b/followthrough.log').read_text(encoding='utf-8')
    except (OSError, ValueError, KeyError):
        return False


@contextmanager
def driver_lock(path):
    """OS-held lock automatically releases on exit; stale files are harmless."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise DriverAlreadyRunning('an e08 driver already owns this lock') from error
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def accept_reviewed_repair(path, current, receipt_path):
    """Explicit, recorded implementation-only migration; reject all others.

    The receipt binds exact before/after manifests. The original protocol
    snapshot remains saved; the scientific design hash cannot change.
    """
    path = Path(path)
    previous = read_json(path)
    receipt = read_json(receipt_path)
    if (receipt.get('previous_manifest') != previous or receipt.get('new_manifest') != current
            or previous['design_sha256'] != current['design_sha256']
            or not receipt.get('reason') or not receipt.get('verification')
            or receipt.get('reviewer') != 'Codex implementation review'
            or receipt.get('scientific_settings_unchanged') is not True):
        raise RuntimeError('implementation repair receipt does not match exact unchanged-design transition')
    changed = sorted(k for k in set(previous['code_sha256']) | set(current['code_sha256'])
                     if previous['code_sha256'].get(k) != current['code_sha256'].get(k))
    if changed != sorted(receipt.get('changed_files', [])) or not changed:
        raise RuntimeError('implementation repair changed-file list mismatch')
    history = path.parent / 'protocol_history' / (digest(path) + '.json')
    if not history.exists():
        atomic_json(history, previous)
    atomic_json(path.parent / 'last_reviewed_repair.json',
                {'receipt_path': str(Path(receipt_path).resolve()), 'receipt_sha256': digest(receipt_path),
                 'previous_snapshot': str(history), 'changed_files': changed})
    atomic_json(path, current)
