"""Held-out paired signed interventions with fixed CI and retained delta."""
from __future__ import annotations
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F

from percinterp.e08_contextual import CANDIDATES
from percinterp.e08_hooks import contribution_hook
from percinterp.e08_io import atomic_json, read_json, digest
from e08_collect import load_inputs, delta_weights, masked_info

ROOT = Path(__file__).resolve().parents[1]


def variants(plan):
    result = [{'name': 'identity', 'group': plan['groups'][0], 'alpha': 1, 'coordinate': False}]
    for group in (0, 1):
        for alpha in (0, 2):
            result.append({'name': f'group{group}_alpha{alpha}', 'group': plan['groups'][group],
                           'alpha': alpha, 'coordinate': False})
        result.append({'name': f'group{group}_coordinate', 'group': plan['groups'][group],
                       'alpha': 0, 'coordinate': True})
    for group in (0, 1):
        for draw, ids in enumerate(plan['random_groups'][group]):
            result.append({'name': f'random{group}_{draw:03d}', 'group': ids,
                           'alpha': 0, 'coordinate': False})
    return result


def run(seed):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    plans = [read_json(ROOT / f'out/e08/seed{seed}_candidate{k}_groups.json') for k in range(3)]
    ready = [k for k, p in enumerate(plans) if p['status'] == 'ready']
    if not ready:
        atomic_json(ROOT / f'out/e08/seed{seed}/effects_done.json', {'status': 'no_eligible_cases'}); return
    rows, _, evaluation, cm, info, _, evidence = load_inputs(seed)
    deltas = delta_weights(cm)
    from e04_common import ci_lower
    from param_decomp.models.components import make_mask_infos
    started, completed = time.monotonic(), 0
    with torch.no_grad():
        for k in ready:
            plan = plans[k]
            layer, role, neuron = CANDIDATES[k]
            module = f'model.layers.{layer}.mlp.{role}_proj'
            comp = cm.components[module]
            mean_coefficient = torch.tensor(plan['means_position_class'][1], device='cuda', dtype=torch.float32)
            settings = variants(plan)
            destination = ROOT / f'out/e08/seed{seed}/effects_candidate{k}'
            destination.mkdir(parents=True, exist_ok=True)
            settings_path = destination / 'variants.json'
            if settings_path.exists() and read_json(settings_path) != settings:
                raise RuntimeError('intervention settings changed on resume')
            atomic_json(settings_path, settings)
            plan_hash = digest(ROOT / f'out/e08/seed{seed}_candidate{k}_groups.json')
            labels = np.array(plan['evaluation_labels'])
            for row in np.flatnonzero(labels >= 0):
                path = destination / f'row_{row:04d}.npz'
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        if str(saved['plan_sha256']) != plan_hash or saved['metrics'].shape != (207, 2):
                            raise RuntimeError('saved paired effects are inconsistent')
                    continue
                idx = rows[int(evaluation[row]):int(evaluation[row])+1].to('cuda')
                with np.load(ROOT / f'out/e08/seed{seed}/row_{row+512:04d}.npz', allow_pickle=False) as saved:
                    position = int(saved['positions'][k])
                    saved_coefficient = saved['ci'][k, 0] * saved['raw_act'][k, 0]
                # Baseline CI is measured on the original target and then held fixed.
                ci = ci_lower(cm, info, idx)
                baseline = cm(idx, mask_infos=masked_info(ci, deltas), cache_type='input')
                coefficient = (ci[module] * comp.get_component_acts(baseline.cache[module]))[0, position].float()
                if not np.allclose(coefficient.cpu().numpy(), saved_coefficient, atol=1e-4, rtol=1e-3):
                    raise RuntimeError('saved and recomputed contextual contributions disagree')
                baseline_logits = baseline.output[0, position].float().clone()
                reference = F.log_softmax(baseline_logits, -1)
                target_token = int(idx[0, position+1])
                metrics, identity_error = [], None
                # Four variants of one context per batch. Masks are expanded,
                # with no CI recomputation or change to scientific sample counts.
                for at in range(0, len(settings), 4):
                    chunk = settings[at:at+4]
                    shifts = []
                    for variant in chunk:
                        ids = torch.tensor(variant['group'], device='cuda')
                        current = coefficient[ids] @ comp.U[ids].float()
                        mean = mean_coefficient[ids] @ comp.U[ids].float()
                        shift = (variant['alpha'] - 1) * (current - mean)
                        if variant['coordinate']:
                            only = torch.zeros_like(shift); only[neuron] = shift[neuron]; shift = only
                        shifts.append(shift)
                    shifts = torch.stack(shifts)
                    repeated_ci = {m: v.expand(len(chunk), -1, -1) for m, v in ci.items()}
                    # alpha=0 with vector=-shift implements the prepared signed
                    # additive difference, including mixed variant types.
                    with contribution_hook(comp, position, -shifts, torch.zeros_like(shifts), alpha=0):
                        output = cm(idx.expand(len(chunk), -1), mask_infos=masked_info(repeated_ci, deltas))
                    logits = output[:, position].float()
                    logp = F.log_softmax(logits, -1)
                    kl = (reference.exp() * (reference - logp)).sum(-1)
                    changes = logp[:, target_token] - reference[target_token]
                    metrics.extend(torch.stack([changes, kl], -1).cpu().numpy().tolist())
                    if at == 0:
                        identity_error = float((logits[0] - baseline_logits).abs().max())
                        if identity_error > .003:
                            raise RuntimeError(f'identity hook or batched invariance failed: {identity_error}')
                    del output
                temporary = path.with_suffix('.npz.tmp')
                with temporary.open('wb') as f:
                    np.savez_compressed(f, metrics=np.array(metrics), plan_sha256=np.array(plan_hash),
                        identity_max_logit_error=np.array(identity_error), row_id=np.array(evaluation[row]),
                        position=np.array(position), family=np.array(labels[row]))
                temporary.replace(path)
                completed += 1
                print(f'seed{seed} candidate{k} evaluation row {row+1}/512; '
                      f'{completed} rows computed in {time.monotonic()-started:.0f}s', flush=True)
                del baseline, ci
    atomic_json(ROOT / f'out/e08/seed{seed}/effects_done.json', {'status': 'complete',
        'eligible_candidates': ready, 'checkpoint_sha256': evidence['sha256']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
    run(parser.parse_args().seed)
