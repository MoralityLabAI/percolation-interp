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
