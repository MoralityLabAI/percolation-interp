"""E11 held-out, paired sequence KL audit for a frozen geometry selector manifest.

Input is E10's per-sequence causal JSON with E11 selectors appended to the
manifest BEFORE evaluation. Never accepts synthetic selector provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PRIMARY = ('joint_degree_ci', 'local_degree', 'local_weighted_degree',
           'top_ci', 'top_attr', 'top_rate', 'sieve')
MIN_EFFECT = .02


def score(record, manifest, *, draws=2000, seed=20261102):
    if manifest.get('e11_response_geometry', {}).get('response_source') == 'synthetic_fixture':
        raise ValueError('synthetic calibration cannot be scored as real model evidence')
    if record.get('schema') != 'percinterp.e10.causal.v1' or record.get('source_type') != 'E03_VPD_SAVED_INPUTS':
        raise ValueError('requires actual E10 model inference record')
    mh = hashlib.sha256(json.dumps(manifest, sort_keys=True,
                                   separators=(',', ':')).encode()).hexdigest()
    if record.get('manifest_sha256') != mh:
        raise ValueError('causal output does not match frozen selection manifest')
    if not 100 <= draws <= 100000:
        raise ValueError('bootstrap draws must be 100..100000')
    kl = {k: np.asarray(v, dtype=float) for k,v in record['per_sequence_kl'].items()}
    wanted = set(PRIMARY) | {'spectral_geometry', 'all', 'none'}
    if not wanted.issubset(kl):
        raise ValueError(f'missing frozen selectors: {wanted-set(kl)}')
    length = len(kl['spectral_geometry'])
    if length < 2 or any(x.shape != (length,) or not np.isfinite(x).all() for x in kl.values()):
        raise ValueError('all selectors must have finite paired sequence KL')
    baseline = kl['spectral_geometry']
    gap = float(np.mean(kl['none'])-np.mean(kl['all']))
    ratio = float(np.mean(kl['all'])/np.mean(kl['none'])) if np.mean(kl['none'])>0 else float('inf')
    if gap <= 0:
        return {'status':'INVALID_ABLATION_GAP','gap':gap}
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,length,size=(draws,length))
    bootgap=(kl['none'][idx]-kl['all'][idx]).mean(axis=1)
    good=bootgap>0
    if not np.any(good):
        raise ValueError('no bootstrap resample with positive ablation gap')
    results={}
    for key in sorted(k for k in kl if k in PRIMARY or k.startswith('spectral_perm_')):
        deltas=kl[key]-baseline
        means=deltas[idx].mean(axis=1)
        vals=means[good]/bootgap[good]
        lo,hi=np.quantile(vals,[.025,.975])
        effect=float(np.mean(deltas)/gap)
        exact_same={m:manifest['selection'][m] == manifest['selection']['spectral_geometry']
                    for m in PRIMARY if m in manifest['selection']}
        equivalent=bool(exact_same.get(key,False))
        if equivalent and not np.array_equal(kl[key],baseline):
            raise ValueError(f'identical component sets produced unequal KL: {key}')
        results[key]={'extra_gap_closed':effect,'ci_95':[float(lo),float(hi)],
                      'exact_mask_equivalence':equivalent,
                      'passes':bool(effect>=MIN_EFFECT and lo>0)}
    passes=bool(ratio<=.25 and all(results[k]['passes'] for k in PRIMARY))
    perms=[k for k in results if k.startswith('spectral_perm_')]
    conditional_p=(1+sum(np.mean(kl[k])<=np.mean(baseline) for k in perms))/(len(perms)+1) if perms else None
    return {'status':'WITHIN_TARGET_SEED_PASS' if passes else 'GATE_NOT_MET',
            'quality_gate_pass':bool(ratio<=.25),
            'n_sequences':length,'ablation_gap':gap,'kl_all_to_none':ratio,
            'primary_min_extra_gap':MIN_EFFECT, 'results':results,
            'conditional_permutation_p':conditional_p,'n_permutations':len(perms),
            'bootstrap_unit':'512-token sequence', 'bootstrap_draws':draws,
            'scope':'within-target diagnostic; source-document independence unverified'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--causal',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--draws',type=int,default=2000)
    args=p.parse_args()
    if args.output.exists():
        p.error('refusing overwrite')
    m=json.loads(args.manifest.read_text())
    r=json.loads(args.causal.read_text())
    report=score(r,m,draws=args.draws)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(report['status'], 'conditional permutation p=',report.get('conditional_permutation_p'))


if __name__=='__main__':
    main()
