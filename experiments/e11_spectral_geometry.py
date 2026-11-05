"""E11 CPU-only selection from VPD component response geometry.

Consumes a precomputed calibration-split response export and a frozen E10
selector manifest. Does not compute spectroscopy or perform model inference.
The output is an E10-compatible *new* manifest, with spectral and conditional
permutation selectors appended. It must be causally evaluated on held-out rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'percinterp.e11.responses.v1'
KINDS = {'sgld_susceptibility', 'vpd_response_proxy', 'synthetic_fixture'}


def _bins(x, n=3):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.zeros(len(x), dtype=int)
    cuts = np.unique(np.quantile(x, np.arange(1, n)/n))
    return np.searchsorted(cuts, x, side='right')


def _z(x):
    s = np.std(x)
    return (x - np.mean(x))/s if s > 1e-12 else np.zeros_like(x)


def _residual_unit_vectors(response, nuisance, modules):
    """Residualize component profiles on nuisance covariates before geometry.

    Each component is a row in the resulting matrix; scaling to unit length
    makes the logdet selector insensitive to its overall response magnitude.
    """
    r = np.asarray(response, dtype=float)
    z = np.asarray(nuisance, dtype=float)
    m = np.asarray(modules).astype(str)
    n = len(m)
    if r.ndim != 2 or z.ndim != 2 or r.shape[1] != n or z.shape[0] != n:
        raise ValueError('response [perturbation,component], nuisance [component,feature] required')
    if r.shape[0] < 3 or not np.isfinite(r).all() or not np.isfinite(z).all():
        raise ValueError('need >=3 finite calibration perturbations and finite covariates')
    # Remove per-component response intercepts across calibration stimuli.
    y = (r - r.mean(axis=0, keepdims=True)).T
    dummies = np.column_stack([(m == k).astype(float) for k in sorted(set(m))])
    z_standardized = np.column_stack([_z(z[:, j]) for j in range(z.shape[1])])
    design = np.column_stack((dummies, z_standardized))
    fitted = design @ np.linalg.lstsq(design, y, rcond=1e-9)[0]
    residual = y - fitted
    raw_norm = np.linalg.norm(y, axis=1)
    norm = np.linalg.norm(residual, axis=1)
    # Treat numerical regression noise as zero response.
    valid = norm > 1e-7 * np.maximum(raw_norm, 1.0)
    unit = np.zeros_like(residual)
    unit[valid] = residual[valid] / norm[valid, None]
    return unit, {'nonzero_residual_profiles': int(valid.sum()),
                  'n_components': n,
                  'residual_energy_fraction': float(np.sum(residual**2) /
                                                     max(np.sum(y**2), 1e-20))}


def _groups(modules, degree, ci, n_bins=3):
    m = np.asarray(modules).astype(str)
    deg_bin, ci_bin = np.zeros(len(m), dtype=int), np.zeros(len(m), dtype=int)
    for mod in sorted(set(m)):
        ix = np.flatnonzero(m == mod)
        deg_bin[ix] = _bins(degree[ix], n_bins)
        ci_bin[ix] = _bins(ci[ix], n_bins)
    return [(str(m[i]), int(deg_bin[i]), int(ci_bin[i])) for i in range(len(m))]


def _joint_baseline(modules, local_ids, degree, ci, quotas):
    m = np.asarray(modules).astype(str)
    selected = []
    for mod, quota in sorted(quotas.items()):
        ix = np.flatnonzero(m == mod)
        if len(ix) < quota:
            raise ValueError(f'quota exceeds eligible pool for {mod}')
        score = _z(np.log1p(degree[ix])) + _z(np.log1p(ci[ix]))
        chosen = sorted(ix, key=lambda k: (-float(score[np.where(ix == k)[0][0]]),
                                          int(local_ids[k])))[:quota]
        selected.extend(chosen)
    return sorted(selected)


def _select(K, groups, targets):
    """Greedy logdet(I+K[S,S]), with exact joint degree/CI stratum quotas."""
    n = len(groups)
    selection, used = [], Counter()
    while len(selection) < sum(targets.values()):
        eligible = [i for i in range(n) if i not in selection and
                    used[groups[i]] < targets[groups[i]]]
        if not eligible:
            raise ValueError('stratum quota cannot be met')
        if selection:
            A = np.eye(len(selection)) + K[np.ix_(selection, selection)]
            cross = K[np.ix_(selection, eligible)]
            marginal = np.diag(K)[eligible] - np.sum(cross * np.linalg.solve(A, cross), axis=0)
        else:
            marginal = np.diag(K)[eligible]
        marginal = np.maximum(marginal, 0)
        best = eligible[int(np.argmax(marginal))]
        selection.append(best)
        used[groups[best]] += 1
    return sorted(selection)


def geometry_plan(response, nuisance, modules, local_ids, quotas, n_null=20,
                  seed=20261102, bins=3):
    m = np.asarray(modules).astype(str)
    i = np.asarray(local_ids, dtype=int)
    z = np.asarray(nuisance, dtype=float)
    if len(set(zip(m, i))) != len(m) or np.any(i < 0):
        raise ValueError('duplicate or negative component identities')
    if set(quotas) != set(m) or any(v < 0 for v in quotas.values()):
        raise ValueError('module quotas and response export disagree')
    if z.shape != (len(m), 5) or np.any(z < 0):
        raise ValueError('expected five nonnegative nuisance columns: in/out-degree, CI, attribution, rate')
    if not (0 <= n_null <= 99) or not (2 <= bins <= 6):
        raise ValueError('invalid number of nulls or strata')
    degree, ci = z[:, 0]+z[:, 1], z[:, 2]
    base = _joint_baseline(m, i, degree, ci, quotas)
    groups = _groups(m, degree, ci, bins)
    targets = Counter(groups[k] for k in base)
    unit, diag = _residual_unit_vectors(response, z, m)
    K = unit @ unit.T
    picks = {'joint_degree_ci': base, 'spectral_geometry': _select(K, groups, targets)}
    rng = np.random.default_rng(seed)
    for j in range(n_null):
        perm = np.arange(len(m))
        for key in sorted(set(groups)):
            members = np.asarray([k for k, group in enumerate(groups) if group == key])
            perm[members] = rng.permutation(members)
        k_perm = unit[perm] @ unit[perm].T
        picks[f'spectral_perm_{j:03d}'] = _select(k_perm, groups, targets)
    def format_set(indexes):
        return {mod: sorted(int(i[j]) for j in indexes if m[j] == mod)
                for mod in sorted(quotas)}
    diag['strata_count'] = len(targets)
    diag['spectral_replacements'] = len(set(base)-set(picks['spectral_geometry']))
    diag['same_as_joint_baseline'] = base == picks['spectral_geometry']
    diag['n_nulls'] = n_null
    return {key: format_set(vals) for key, vals in picks.items()}, diag


def load_export(path):
    with np.load(path, allow_pickle=False) as o:
        for key in ('responses', 'nuisance', 'modules', 'local_ids', 'response_source',
                    'calibration_split', 'graph_sha256', 'vpd_run'):
            if key not in o:
                raise ValueError(f'missing {key} in response export')
        data = {key: o[key].copy() for key in o.files}
    kind = str(data['response_source'].item())
    if kind not in KINDS:
        raise ValueError('unrecognized response source')
    return data, kind


def derive_manifest(e10, data, kind, *, digest, n_null, seed, bins):
    if e10.get('schema') != 'percinterp.e10.selector.v1':
        raise ValueError('a frozen E10 selector manifest is required')
    if str(data['graph_sha256'].item()) != e10['graph_sha256']:
        raise ValueError('response graph fingerprint disagrees with E10')
    if str(data['vpd_run'].item()) != e10.get('provenance', {}).get('vpd_run'):
        raise ValueError('response VPD run differs from E10')
    real = e10.get('source_type') == 'E03_VPD_SAVED_INPUTS'
    if real != (kind != 'synthetic_fixture'):
        raise ValueError('synthetic and real provenance cannot be mixed')
    if real and str(data['calibration_split'].item()) != 'train':
        raise ValueError('real calibration must come from train split, E10 evaluates test')
    mods = data['modules'].astype(str)
    loc = data['local_ids'].astype(int)
    known = set(zip(mods, loc))
    for selector in e10['selection'].values():
        if any((mod, int(j)) not in known for mod, ids in selector.items() for j in ids):
            raise ValueError('export missing an existing E10 selected component')
    newsets, diag = geometry_plan(data['responses'], data['nuisance'], mods, loc,
                                   e10['module_quotas'], n_null=n_null, seed=seed, bins=bins)
    merged = json.loads(json.dumps(e10))
    if set(newsets) & set(merged['selection']):
        raise ValueError('selector name collision')
    merged['selection'].update(newsets)
    merged['e11_response_geometry'] = {
        'schema': SCHEMA, 'response_file_sha256': digest, 'response_source': kind,
        'calibration_split': str(data['calibration_split'].item()),
        'method': 'nuisance-residualized unit component profiles; conditional logdet selection',
        'nuisance_order': ['in_degree', 'out_degree', 'mean_ci', 'total_abs_attr', 'firing_rate'],
        'strata': 'module x degree-quantile x CI-quantile',
        'strata_bins': bins, 'seed': seed, 'diagnostics': diag,
        'claim_scope': 'selector calibration; no causal inference performed',
    }
    return merged


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--e10-manifest', type=Path, required=True)
    p.add_argument('--responses', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n-nulls', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261102)
    p.add_argument('--bins', type=int, default=3)
    args = p.parse_args()
    if args.output.exists():
        p.error('refusing to overwrite a frozen selection manifest')
    raw = args.responses.read_bytes()
    e10 = json.loads(args.e10_manifest.read_text(encoding='utf-8'))
    data, kind = load_export(args.responses)
    plan = derive_manifest(e10, data, kind, digest=hashlib.sha256(raw).hexdigest(),
                           n_null=args.n_nulls, seed=args.seed, bins=args.bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False)+'\n',
                           encoding='utf-8')
    print(json.dumps(plan['e11_response_geometry']['diagnostics'], indent=2))
    print('Saved new pre-causal manifest:', args.output)


if __name__ == '__main__':
    main()
