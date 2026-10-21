"""Freeze discovery-only cluster proposals; require blinded label review."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from percinterp.checkpoint_arrays import CheckpointArrays
from percinterp.e08_contextual import (CANDIDATES, context_clusters, contrast_groups,
    matched_groups, stratify, SEED)
from percinterp.e08_io import atomic_json, read_json, digest
from percinterp.polysemanticity import direction_energy, support_edges

ROOT = Path(__file__).resolve().parents[1]


def data(seed, candidate):
    directory = ROOT / f'out/e08/seed{seed}'
    result = {k: [] for k in ('positions', 'activation', 'features', 'raw_act', 'ci',
                             'coefficient_sums', 'n_fire', 'energy_sum')}
    for i in range(1024):
        with np.load(directory / f'row_{i:04d}.npz', allow_pickle=False) as row:
            for key in result:
                value = row[key][candidate]
                if key == 'activation':
                    value = value[int(row['positions'][candidate])]
                result[key].append(value.copy())
    return {k: np.array(v) for k, v in result.items()}


def proposals():
    destination = ROOT / 'out/e08/proposals.json'
    review_path = ROOT / 'out/e08/blinded_contexts.json'
    if destination.exists() and review_path.exists():
        return
    frozen, review = {}, {}
    for seed in (0, 1):
        for candidate in range(3):
            name = f'seed{seed}_candidate{candidate}'
            d = data(seed, candidate)
            # Stable activation rank; row index is the deterministic tie break.
            selected = np.argsort(-np.abs(d['activation'][:512]), kind='stable')[:128]
            cluster = context_clusters(d['features'][selected], d['features'][512:],
                                       seed=81024 + seed * 3 + candidate)
            if cluster['status'] != 'proposed':
                frozen[name] = {'status': cluster['status']}
                continue
            labels = cluster['discovery']
            frozen[name] = {'status': 'proposed', 'selected_discovery_rows': selected.tolist(),
                'discovery_labels': labels.tolist(), 'evaluation_labels': cluster['evaluation'].tolist(),
                'centroids': cluster['centroids'].tolist()}
            # Opaque case identifiers; review sees prefixes only, no activations,
            # component IDs, interventions, next-token targets, or evaluation rows.
            opaque = f'case_{len(review):02d}'
            frozen[name]['review_case'] = opaque
            review[opaque] = {}
            similarity = d['features'][selected] / np.linalg.norm(d['features'][selected], axis=1, keepdims=True)
            for c in range(4):
                members = np.flatnonzero(labels == c)
                rank = members[np.argsort(-(similarity[members] @ cluster['centroids'][c]), kind='stable')]
                # Eight central and eight dispersed examples expose incoherence.
                examples = np.unique(np.r_[rank[:8], rank[np.linspace(0, len(rank)-1, min(8, len(rank)), dtype=int)]])
                if len(examples) < min(16, len(members)):
                    examples = np.r_[examples, [j for j in rank if j not in examples]][:16]
                review[opaque][str(c)] = {'discovery_count': int(len(members)), 'examples': []}
                for j in examples:
                    row = int(selected[j])
                    context = read_json(ROOT / f'out/e08/seed{seed}/row_{row:04d}.json')['contexts'][candidate]
                    review[opaque][str(c)]['examples'].append({'discovery_index': row, 'prefix': context['prefix']})
    atomic_json(destination, frozen)
    atomic_json(review_path, {'policy': 'Discovery prefixes only; clusters are unvalidated proposals.',
        'required_review': 'For each cluster supply family, coherent boolean, confirmed example indices, rationale. '
            'Require >=12 coherent examples of the 16 reviewed; lexical format alone is not a semantic family. '
            'Use coherent=false when uncertain. No independent inter-annotator agreement is available.',
        'cases': review})


def freeze_groups():
    """Called only after labels.json is supplied by a reviewer, never invented."""
    labels_path = ROOT / 'out/e08/labels.json'
    if not labels_path.exists():
        return False
    labels = read_json(labels_path)
    if not labels.get('reviewer') or not labels.get('method') or not isinstance(labels.get('cases'), dict):
        raise ValueError('missing label reviewer/method/cases')
    proposal = read_json(ROOT / 'out/e08/proposals.json')
    review = read_json(ROOT / 'out/e08/blinded_contexts.json')['cases']
    evidence = read_json(ROOT / 'results/e05b_retrieval.json')
    for seed in (0, 1):
        run = evidence['runs'][str(seed)]['run_id']
        checkpoint = CheckpointArrays(Path.home() / 'param_decomp_out/decompositions' / run / 'model_20000.pth')
        for k, (layer, role, neuron) in enumerate(CANDIDATES):
            name = f'seed{seed}_candidate{k}'
            destination = ROOT / f'out/e08/{name}_groups.json'
            if destination.exists():
                if read_json(destination)['label_sha256'] != digest(labels_path):
                    raise RuntimeError('labels changed after groups were frozen')
                continue
            base = {'seed': seed, 'candidate': k, 'layer': layer, 'role': role, 'neuron': neuron,
                    'label_sha256': digest(labels_path), 'proposal_sha256': digest(ROOT / 'out/e08/proposals.json')}
            p = proposal[name]
            if p['status'] != 'proposed':
                atomic_json(destination, dict(base, status=p['status'])); continue
            case = p['review_case']
            assignments = labels['cases'].get(case)
            if not isinstance(assignments, dict) or set(assignments) != {'0', '1', '2', '3'}:
                raise ValueError(f'incomplete blinded review for {case}')
            good = []
            for cluster, annotation in assignments.items():
                if (not isinstance(annotation.get('coherent'), bool) or not annotation.get('rationale')
                        or not isinstance(annotation.get('confirmed_indices'), list)):
                    raise ValueError(f'incomplete cluster annotation {case}/{cluster}')
                known = {v['discovery_index'] for v in review[case][cluster]['examples']}
                confirmed = set(annotation['confirmed_indices'])
                if not confirmed <= known:
                    raise ValueError('annotation cites an unseen discovery context')
                if annotation['coherent']:
                    family = annotation.get('family', '').strip()
                    if not family or len(confirmed) < 12:
                        raise ValueError('coherent family needs a name and >=12 supporting examples')
                    count = review[case][cluster]['discovery_count']
                    if count >= 32:
                        good.append((int(cluster), family, count))
            good.sort(key=lambda v: (-v[2], v[1], v[0]))
            if len(good) < 2:
                atomic_json(destination, dict(base, status='insufficient_semantic_families')); continue
            a, b = good[:2]
            if a[1] == b[1]:
                atomic_json(destination, dict(base, status='families_not_distinct')); continue
            d = data(seed, k)
            selected = np.array(p['selected_discovery_rows'])
            disc, test = np.array(p['discovery_labels']), np.array(p['evaluation_labels'])
            test_labels = np.full(512, -1, dtype=int)
            for family, cluster in enumerate((a[0], b[0])):
                test_labels[test == cluster] = family
            if min(np.sum(test_labels == j) for j in (0, 1)) < 32:
                atomic_json(destination, dict(base, status='insufficient_evaluation_families')); continue
            u = checkpoint.array(f'_components.model-layers-{layer}-mlp-{role}_proj.U')
            q = direction_energy(u)
            left, right = support_edges(q, 8)
            contacts = left[right == neuron]
            used = np.isin(disc, [a[0], b[0]])
            family_labels = (disc[used] == b[0]).astype(int)
            signed = (d['raw_act'][selected[used], 0] * d['ci'][selected[used], 0]) * u[:, neuron]
            groups, contrast = contrast_groups(signed, family_labels, contacts)
            rate = d['n_fire'][:512].sum(0) / (512 * 128)
            energy = d['energy_sum'][:512].sum(0) / (512 * 128)
            random = matched_groups(groups, rate, energy, np.random.default_rng(SEED + seed * 3 + k))
            if random['status'] != 'matched':
                atomic_json(destination, dict(base, status=random['status'], matching=random)); continue
            mu = d['coefficient_sums'][:512].sum(0) / np.array([[512], [512 * 127]])
            strata = stratify(d['positions'][512:], d['activation'][512:], d['activation'][:512])
            full_signed = (d['raw_act'][selected, 0] * d['ci'][selected, 0]) * u[:, neuron]
            group_mass = [float(np.abs(full_signed[:, g]).sum() / max(np.abs(full_signed).sum(), 1e-30)) for g in groups]
            atomic_json(destination, dict(base, status='ready', families=[a[1], b[1]],
                groups=[g.tolist() for g in groups], random_groups=[g.tolist() for g in random['groups']],
                contrast_all_components=contrast.tolist(), contact_ids=contacts.tolist(),
                absolute_coordinate_contribution_mass=group_mass,
                means_position_class=mu.tolist(), evaluation_labels=test_labels.tolist(),
                evaluation_strata=strata.tolist(), matching_bins=random['bins']))
    return True


if __name__ == '__main__':
    proposals()
    print('labels ready' if freeze_groups() else 'awaiting blinded semantic review', flush=True)
