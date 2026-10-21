"""Token-context and signed-coefficient collection from saved decompositions.

Called in a fresh per-seed process by e08_pipeline under a 6 GiB lease.
Only selected tokens and position zero retain per-component records; full
token summaries are accumulated online, avoiding multi-GB redundant tensors.
"""
from __future__ import annotations
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  PYTHONIOENCODING='utf-8')
import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F

from percinterp.e08_contextual import CANDIDATES, selected_position, split_row_ids
from percinterp.e08_io import atomic_json, digest, read_json

ROOT = Path(__file__).resolve().parents[1]


def load_inputs(seed):
    from e04_common import heldout_tokens, find_run, load_component_model
    from transformers import AutoTokenizer
    rows = heldout_tokens()
    discovery, evaluation = split_row_ids(len(rows), torch.randperm(len(rows),
        generator=torch.Generator().manual_seed(0)).numpy())
    evidence = read_json(ROOT / 'results/e05b_retrieval.json')['runs'][str(seed)]
    checkpoint = Path.home() / 'param_decomp_out/decompositions' / evidence['run_id'] / 'model_20000.pth'
    if checkpoint.stat().st_size != evidence['final_checkpoint_bytes'] or digest(checkpoint) != evidence['sha256']:
        raise RuntimeError('saved checkpoint integrity failed; never retrain')
    run = find_run()
    if run != evidence['run_id']:
        raise RuntimeError('e08 configuration discovered the wrong decomposition')
    cm, info = load_component_model(run, 'cuda')
    if info.config.sampling != 'continuous':
        raise RuntimeError('e08 requires deterministic continuous CI, not a stochastic mask realization')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', local_files_only=True)
    return rows, discovery, evaluation, cm, info, tokenizer, evidence


def delta_weights(cm):
    # Keep full-rank residuals in e08; e05's separate gate remains unchanged.
    return {m: (cm.target_model.get_submodule(m).weight.detach().float()
                - comp.weight.detach().float()).to(comp.U.dtype)
            for m, comp in cm.components.items()}


def masked_info(ci, deltas, ones=False):
    from param_decomp.models.components import make_mask_infos
    masks = {m: torch.ones_like(v) if ones else v for m, v in ci.items()}
    return make_mask_infos(masks, weight_deltas_and_masks={
        m: (deltas[m], torch.ones(v.shape[:-1], device=v.device, dtype=v.dtype))
        for m, v in ci.items()})


def kl_rows(target, altered):
    total = 0.
    # Avoid retaining two full-vocabulary log-softmax tensors for a whole row.
    for start in range(0, target.shape[1], 16):
        logp = F.log_softmax(target[:, start:start+16].float(), -1)
        other = F.log_softmax(altered[:, start:start+16].float(), -1)
        total += float((logp.exp() * (logp - other)).sum(-1).sum())
    return total / target.shape[1]


def collect(seed):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    rows, discovery, evaluation, cm, info, tok, evidence = load_inputs(seed)
    directory = ROOT / f'out/e08/seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'seed': seed, 'run_id': evidence['run_id'], 'checkpoint_sha256': evidence['sha256'],
        'discovery_row_ids': discovery.tolist(), 'evaluation_row_ids': evaluation.tolist(),
        'encoder': 'saved Qwen/Qwen3-0.6B original-target lm_head input at selected token, L2 normalized later',
        'record_policy': 'full component records at primary token and position 0; all-token discovery summaries',
        'delta_policy': 'full-rank delta retained unchanged; e05 gate is reported separately',
        'design_sha256': digest(ROOT / 'docs/e08_design.md')}
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise RuntimeError('resume manifest changed')
    atomic_json(manifest_path, manifest)
    modules = [f'model.layers.{layer}.mlp.{role}_proj' for layer, role, _ in CANDIDATES]
    comps = [cm.components[m] for m in modules]
    capture = {}
    def coordinate_hook(k, j):
        def hook(module, args, output):
            capture[k] = output[0, :, j].detach().float().cpu().numpy().copy()
        return hook
    handles = [cm.target_model.get_submodule(m).register_forward_hook(coordinate_hook(k, j))
               for k, (m, (_, _, j)) in enumerate(zip(modules, CANDIDATES))]
    def head_hook(module, args):
        capture['embedding'] = args[0].detach()
    handles.append(cm.target_model.lm_head.register_forward_pre_hook(head_hook))
    deltas = delta_weights(cm)
    row_ids = np.r_[discovery, evaluation]
    started = time.monotonic()
    try:
        with torch.no_grad():
            for index, row_id in enumerate(row_ids):
                destination = directory / f'row_{index:04d}.npz'
                context_path = destination.with_suffix('.json')
                if destination.exists() and context_path.exists():
                    # Validate actual load, not mere existence, before reuse.
                    with np.load(destination, allow_pickle=False) as saved:
                        if int(saved['row_id']) != row_id or saved['raw_act'].shape != (3, 2, 768):
                            raise RuntimeError('invalid saved e08 row')
                    continue
                idx = rows[int(row_id):int(row_id)+1].to('cuda')
                original = cm(idx, cache_type='input')
                activation = np.array([capture[k] for k in range(3)])
                positions = np.array([selected_position(a) for a in activation])
                features = capture['embedding'][0, positions].float().cpu().numpy()
                ci = cm.calc_causal_importances(pre_weight_acts=original.cache,
                    detach_inputs=True, sampling=info.config.sampling).lower_leaky
                # Check the retained delta reconstructs the original model.
                reconstruction_error = None
                if index == 0:
                    reconstructed = cm(idx, mask_infos=masked_info(ci, deltas, ones=True))
                    reconstruction_error = float((reconstructed.float() - original.output.float()).abs().max())
                    if not torch.allclose(reconstructed.float(), original.output.float(), atol=.003, rtol=.001):
                        raise RuntimeError(f'full-rank reconstruction failed: {reconstruction_error}')
                    del reconstructed
                baseline = cm(idx, mask_infos=masked_info(ci, deltas), cache_type='input')
                values = {'row_id': np.array(row_id), 'positions': positions, 'features': features,
                          'activation': activation, 'target_kl_to_ci_delta': np.array(kl_rows(original.output, baseline.output))}
                raw, gates, coefficients, fires, energy = [], [], [], [], []
                for k, (m, comp) in enumerate(zip(modules, comps)):
                    acts = comp.get_component_acts(baseline.cache[m]).float()[0]
                    gate = ci[m].float()[0]
                    weighted = gate * acts
                    where = torch.tensor([int(positions[k]), 0], device='cuda')
                    raw.append(acts[where].cpu().numpy())
                    gates.append(gate[where].cpu().numpy())
                    coefficients.append(torch.stack([weighted[0], weighted[1:].sum(0)]).cpu().numpy())
                    fires.append((gate > .1).sum(0).cpu().numpy())
                    energy.append((weighted.square().sum(0) * comp.U.float().square().sum(1)).cpu().numpy())
                values.update(raw_act=np.array(raw), ci=np.array(gates), coefficient_sums=np.array(coefficients),
                              n_fire=np.array(fires), energy_sum=np.array(energy))
                temporary = destination.with_suffix('.npz.tmp')
                with temporary.open('wb') as f:
                    np.savez_compressed(f, **values)
                temporary.replace(destination)
                contexts = []
                for k, pos in enumerate(positions):
                    p = int(pos)
                    contexts.append({'candidate': k, 'position': p,
                        'prefix': tok.decode(idx[0, :p+1].tolist()),
                        'next_token': tok.decode([int(idx[0, p+1])]),
                        'next_token_id': int(idx[0, p+1]),
                        'position0_prefix': tok.decode([int(idx[0, 0])])})
                atomic_json(context_path, {'row_id': int(row_id), 'contexts': contexts,
                    'full_rank_reconstruction_max_logit_error': reconstruction_error})
                if (index + 1) % 16 == 0:
                    print(f'seed{seed} context row {index+1}/1024, {time.monotonic()-started:.0f}s', flush=True)
                del original, baseline, ci
    finally:
        for handle in handles:
            handle.remove()
    atomic_json(directory / 'collect_done.json', {'rows': 1024, 'seed': seed,
        'manifest_sha256': digest(manifest_path), 'checkpoint_sha256': evidence['sha256']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    collect(args.seed)
