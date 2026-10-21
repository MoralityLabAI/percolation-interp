"""All-energy e08 percolation; run after e05b/e06 via the e08 driver."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path
import time
import numpy as np

from percinterp.checkpoint_arrays import CheckpointArrays
from percinterp.e08_contextual import weighted_curve, weighted_expectations, EXPOSURES
from percinterp.e08_io import atomic_json, read_json
from percinterp.polysemanticity import direction_energy
from e07_geometry import limit_host_memory, half_point

ROOT = Path(__file__).resolve().parents[1]
BASE = Path.home() / 'param_decomp_out/decompositions'


def analyze(q, seed):
    rng = np.random.default_rng(seed)
    uniform = np.full_like(q, 1 / q.shape[1])
    values = {k: [] for k in ('learned', 'row_permuted', 'uniform')}
    for draw in range(32):
        # Independent null permutation each draw, exact row spectra retained.
        permuted = np.array([row[rng.permutation(len(row))] for row in q])
        u = rng.random(q.shape)
        for name, energy in (('learned', q), ('row_permuted', permuted), ('uniform', uniform)):
            values[name].append(weighted_curve(energy, EXPOSURES, u))
        if (draw + 1) % 8 == 0:
            print(f'weighted draw {draw + 1}/32', flush=True)
    result = {'n_components': len(q), 'n_coordinates': q.shape[1], 'draws': 32,
              'rng_seed': seed, 'exposures': EXPOSURES.tolist(), 'curves': {}}
    for name, draws in values.items():
        giant = np.array([d['giant_left_fraction'] for d in draws])
        mean = giant.mean(0)
        result['curves'][name] = {'mean_giant': mean.tolist(),
            'q05_giant': np.quantile(giant, .05, axis=0).tolist(),
            'q95_giant': np.quantile(giant, .95, axis=0).tolist(),
            'mean_susceptibility': np.mean([d['susceptibility_left'] for d in draws], axis=0).tolist(),
            't50_mean_curve': half_point(EXPOSURES, mean),
            'auc_mean_curve': float(np.trapezoid(mean, EXPOSURES)),
            **weighted_expectations(uniform if name == 'uniform' else q, EXPOSURES)}
    result['grid_caveat'] = 'Interpolated finite 50%-cluster crossovers on a fixed log-exposure grid are not critical points.'
    return result


def main():
    peak = limit_host_memory(768)
    started = time.monotonic()
    retrieval = read_json(ROOT / 'results/e05b_retrieval.json')
    maps = {}
    for seed in (0, 1):
        run = retrieval['runs'][str(seed)]['run_id']
        path = BASE / run / 'model_20000.pth'
        state = CheckpointArrays(path)
        for layer in (0, 1):
            for role, suffix in (('gate_write', 'gate_proj.U'), ('up_write', 'up_proj.U'), ('down_read', 'down_proj.V')):
                name = f'seed{seed}_layer{layer}_{role}'
                destination = ROOT / f'results/e08/weighted/{name}.json'
                if destination.exists():
                    maps[name] = read_json(destination)
                    continue
                a = state.array(f'_components.model-layers-{layer}-mlp-{suffix}')
                if role == 'down_read':
                    a = a.T
                # All 768 nonzero rows are expected; never silently discard.
                q = direction_energy(a)
                record = analyze(q, 81023 + 100 * seed + 10 * layer + len(maps))
                record['checkpoint_sha256'] = retrieval['runs'][str(seed)]['sha256']
                atomic_json(destination, record)
                maps[name] = record
                print(f'{name} complete, elapsed {time.monotonic() - started:.0f}s', flush=True)
    atomic_json(ROOT / 'results/e08/weighted.json', {'maps': maps,
        'semantic_interpretation': 'none; static unit-row-energy geometry',
        'elapsed_this_invocation_seconds': time.monotonic() - started,
        'peak_commit_mib': peak(), 'host_commit_limit_mib': 768})


if __name__ == '__main__':
    main()
