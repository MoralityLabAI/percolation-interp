"""Score all six fixed e08 cases, including unsupported cases in Holm's family."""
from pathlib import Path
import numpy as np
from percinterp.e08_contextual import dissociation, holm, SEED
from percinterp.e08_io import atomic_json, read_json, digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = {}
    for seed in (0, 1):
        for k in range(3):
            name = f'seed{seed}_candidate{k}'
            plan_path = ROOT / f'out/e08/{name}_groups.json'
            plan = read_json(plan_path)
            gate = read_json(ROOT / f'out/e05b_seed{seed}/gate.json')
            result = {'status': plan['status'], 'p': 1., 'e05_gate_pass': gate['pass'],
                      'gate': gate, 'layer': plan['layer'], 'role': plan['role'], 'neuron': plan['neuron']}
            if plan['status'] != 'ready':
                results[name] = result; continue
            directory = ROOT / f'out/e08/seed{seed}/effects_candidate{k}'
            settings = read_json(directory / 'variants.json')
            names = [v['name'] for v in settings]
            labels = np.array(plan['evaluation_labels'])
            ids = np.flatnonzero(labels >= 0)
            matrices, identity = [], []
            plan_hash = digest(plan_path)
            for i in ids:
                with np.load(directory / f'row_{i:04d}.npz', allow_pickle=False) as row:
                    if str(row['plan_sha256']) != plan_hash:
                        raise RuntimeError('effect/plan provenance mismatch')
                    matrices.append(row['metrics'].copy()); identity.append(float(row['identity_max_logit_error']))
            metrics = np.array(matrices)
            rng = np.random.default_rng(SEED + 100 + seed * 3 + k)
            score = dissociation(metrics[:, [names.index('group0_alpha0'), names.index('group1_alpha0')], 1],
                                  labels[ids], np.array(plan['evaluation_strata'])[ids], rng)
            result.update(score)
            result.update(families=plan['families'], group_ids=plan['groups'],
                group_coordinate_mass=plan['absolute_coordinate_contribution_mass'],
                identity_max_logit_error=max(identity),
                effects_by_family={}, random_controls={})
            for family in (0, 1):
                observations = metrics[labels[ids] == family]
                result['effects_by_family'][str(family)] = {name: {
                    'mean_next_token_logp_change': float(observations[:, j, 0].mean()),
                    'mean_kl': float(observations[:, j, 1].mean())}
                    for j, name in enumerate(names) if not name.startswith('random')}
                for group in (0, 1):
                    random_ids = [j for j, n in enumerate(names) if n.startswith(f'random{group}_')]
                    draws = observations[:, random_ids, 1].mean(0)
                    real = observations[:, names.index(f'group{group}_alpha0'), 1].mean()
                    result['random_controls'][f'family{family}_group{group}'] = {
                        'draws': len(draws), 'draw_mean_kls': draws.tolist(), 'observed_mean_kl': float(real),
                        'p_random_at_least_observed': float((1 + (draws >= real).sum()) / (1 + len(draws)))}
            results[name] = result
    adjusted = holm([r['p'] for r in results.values()])
    for value, p in zip(results.values(), adjusted):
        value['p_holm_six_cases'] = float(p)
        value['exploratory_selective_disruption'] = bool(value.get('both_positive', False) and p < .05)
        value['semantic_interference_established'] = False
    atomic_json(ROOT / 'results/e08/contextual.json', {'cases': results,
        'labels_sha256': digest(ROOT / 'out/e08/labels.json'),
        'interpretation': 'Contextual labels need blinded review; KL measures disruption, not feature meaning. '
            'Even a positive dissociation does not establish semantic interference or generality across models.'})
    print('All six e08 cases scored, with unsupported cases assigned p=1.', flush=True)


if __name__ == '__main__':
    main()
