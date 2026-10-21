import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from percinterp.e08_contextual import (context_clusters, contrast_groups, decile_bins,
    dissociation, holm, matched_groups, selected_position, split_row_ids,
    weighted_curve, weighted_expectations)
from percinterp.e08_io import atomic_json, driver_lock, prerequisite_files
from percinterp.polysemanticity import bond_curve


def test_split_never_wraps_and_selected_token_has_a_next_token():
    permutation = np.random.default_rng(1).permutation(5910)
    a, b = split_row_ids(5910, permutation)
    assert np.array_equal(a, permutation[4000:4512])
    assert np.array_equal(b, permutation[4512:5024])
    assert not set(a) & set(b)
    with pytest.raises(ValueError):
        split_row_ids(5000, np.arange(5000))
    with pytest.raises(ValueError):
        split_row_ids(5910, np.zeros(5910, dtype=int))
    assert selected_position([100, 3, -3, 200]) == 1


def test_cluster_centers_do_not_see_evaluation_and_degenerate_proposals_fail():
    rng = np.random.default_rng(4)
    x = np.r_[rng.normal([1, 0], .03, (40, 2)), rng.normal([0, 1], .03, (40, 2))]
    a = context_clusters(x, [[1, 0]], k=2)
    b = context_clusters(x, [[0, -1]], k=2)
    assert a['status'] == 'proposed'
    assert np.array_equal(a['centroids'], b['centroids'])
    assert np.array_equal(a['discovery'], b['discovery'])
    assert context_clusters(np.ones((10, 2)), [[1, 1]], k=2)['status'] == 'degenerate_representations'


def test_groups_are_disjoint_and_full_signed_products_are_gauge_invariant():
    z = np.r_[np.tile([3, 2, 0, -1], (32, 1)), np.tile([-1, 0, 2, 3], (32, 1))]
    labels = np.repeat([0, 1], 32)
    groups, contrast = contrast_groups(z, labels, np.arange(4), max_size=1)
    assert groups[0].tolist() == [0] and groups[1].tolist() == [3]
    raw, direction = np.array([2., -3]), np.array([-4., 5])
    assert np.array_equal(raw * direction, (-raw) * (-direction))


def test_matching_preserves_both_bins_without_including_selected_components():
    rate = np.ones(100)
    energy = np.repeat(np.arange(10) + 1, 10)
    groups = [np.array([0, 1]), np.array([10, 11])]
    matched = matched_groups(groups, rate, energy, np.random.default_rng(0))
    assert matched['status'] == 'matched'
    bins = np.array(matched['bins'])
    for source, draws in zip(groups, matched['groups']):
        for ids in draws:
            assert len(np.unique(ids)) == len(ids)
            assert not set(ids) & {0, 1, 10, 11}
            assert np.array_equal(np.sort(bins[ids]), np.sort(bins[source]))
    assert np.unique(decile_bins(np.ones(100))).tolist() == [0]
    scarce = matched_groups([np.array([0]), np.array([1])], [0, 1], [0, 1], np.random.default_rng(0))
    assert scarce['status'] == 'insufficient_exact_matches'


def test_double_dissociation_requires_both_directions_and_row_support():
    labels = np.repeat([0, 1], 40)
    effects = np.c_[3 * (labels == 0), 3 * (labels == 1)]
    score = dissociation(effects, labels, np.zeros(80), np.random.default_rng(3), resamples=99)
    assert score['directed_interactions'] == [3, 3]
    assert score['p'] == .01 and score['both_positive']
    assert score['interaction_ci95'] == [[3, 3], [3, 3]]
    shared = dissociation(np.c_[effects[:, 0], effects[:, 0]], labels, np.zeros(80),
                         np.random.default_rng(3), resamples=99)
    assert not shared['both_positive']
    assert dissociation(effects[:50], labels[:50], np.zeros(50), np.random.default_rng(3))['status'] == 'insufficient_matched_rows'
    # No cross-stratum comparisons when families occupy disjoint positions.
    assert dissociation(effects, labels, labels, np.random.default_rng(3))['status'] == 'insufficient_matched_rows'


def test_holm_accounts_for_all_six_cases_including_unscored_cases():
    assert np.allclose(holm([.001, .01, 1, 1, 1, 1]), [.006, .05, 1, 1, 1, 1])
    with pytest.raises(ValueError):
        holm([float('nan')])


def test_weighted_uniform_reduces_to_exact_binary_bernoulli_draw():
    rng = np.random.default_rng(17)
    q = np.full((7, 11), 1 / 11)
    u = rng.random(q.shape)
    ts = np.linspace(0, 4, 41)
    got = weighted_curve(q, ts, u)
    left, right = np.indices(q.shape)
    want = bond_curve(left.ravel(), right.ravel(), 7, 11, -np.expm1(-ts), u.ravel())
    for key in got:
        assert np.array_equal(got[key], want[key])
    expected = weighted_expectations(q, ts)
    assert np.allclose(expected['expected_edge_fraction'], -np.expm1(-ts))
    assert np.allclose(expected['expected_retained_energy'], -np.expm1(-ts))
    assert expected['contacts_for_50pct_energy'] == [6] * 7
    assert expected['contacts_for_90pct_energy'] == [10] * 7


def test_nonuniform_curve_matches_independent_breadth_first_search():
    rng = np.random.default_rng(18)
    q = rng.random((8, 9)); q[q < .3] = 0; q /= q.sum(1, keepdims=True)
    u = rng.random(q.shape)
    times = [0., .03, .2, 1, 4, 100]
    got = weighted_curve(q, times, u)
    for j, t in enumerate(times):
        keep = u < -np.expm1(-t * 9 * q)
        graph = {i: set() for i in range(17)}
        for a, b in zip(*np.nonzero(keep)):
            graph[a].add(8 + b); graph[8 + b].add(a)
        seen, masses = set(), []
        for root in range(17):
            if root in seen:
                continue
            pending, mass = [root], 0
            seen.add(root)
            while pending:
                node = pending.pop(); mass += int(node < 8)
                for nxt in graph[node] - seen:
                    seen.add(nxt); pending.append(nxt)
            if mass:
                masses.append(mass)
        largest = max(masses)
        assert got['giant_left_fraction'][j] == largest / 8
        assert got['left_cluster_count'][j] == len(masses)
        chi = (sum(m*m for m in masses) - largest*largest) / (8-largest) if largest < 8 else 0
        assert got['susceptibility_left'][j] == chi
    isolated = weighted_curve(np.eye(4), [0, 1, 100], np.zeros((4, 4)))
    assert np.all(isolated['giant_left_fraction'] == .25)
    with pytest.raises(ValueError):
        weighted_curve([[1., -1.]], [1], [[.1, .1]])


def test_atomic_outputs_and_os_driver_lock(tmp_path):
    path = tmp_path / 'nested/result.json'
    atomic_json(path, {'valid': True})
    with pytest.raises(ValueError):
        atomic_json(path, {'invalid': float('nan')})
    assert json.loads(path.read_text()) == {'valid': True}
    lock = tmp_path / 'driver.lock'
    with driver_lock(lock):
        with pytest.raises(OSError):
            with driver_lock(lock):
                pass
    with driver_lock(lock):
        pass


def test_prior_work_completion_requires_all_appropriate_outputs_and_success(tmp_path):
    assert not prerequisite_files(tmp_path)
    for seed in (0, 1):
        out = tmp_path / f'out/e05b_seed{seed}'
        atomic_json(out / 'gate.json', {'pass': bool(seed)})
        for name in ('stats.pt', 'dataset_attributions.pt'):
            (out / name).write_bytes(b'data')
        if seed:
            for name in ('analysis.json', 'causal.json'):
                atomic_json(out / name, {})
    for name in ('e03_seed1', 'e03_seed2', 'e05b_seed0', 'e05b_seed1'):
        atomic_json(tmp_path / f'results/e06/heldout_{name}.json', {})
    logs = tmp_path / 'out/e05b'; logs.mkdir()
    (logs / 'e06_score.log').write_text('scored')
    (logs / 'followthrough.log').write_text('still running')
    assert not prerequisite_files(tmp_path)
    (logs / 'followthrough.log').write_text('All analysis stages finished; reports and local commit need final review')
    assert prerequisite_files(tmp_path)
    (tmp_path / 'out/e05b_seed1/causal.json').write_text('{truncated')
    assert not prerequisite_files(tmp_path)


def test_signed_hook_preserves_delta_other_tokens_and_cleans_up_after_failure():
    torch = pytest.importorskip('torch')
    from percinterp.e08_hooks import contribution_hook
    torch.manual_seed(2)
    x = torch.randn(2, 4, 5)
    u, v, target = torch.randn(3, 7), torch.randn(5, 3), torch.randn(5, 7)
    delta = target - v @ u
    class Projection(torch.nn.Module):
        def forward(self, x):
            return (x @ v) @ u + x @ delta
    component = Projection()
    baseline = component(x)
    assert torch.allclose(baseline, x @ target, atol=2e-6)
    vector = (x[:, 2] @ v)[:, :1] @ u[:1]
    mean = torch.ones_like(vector) * .2
    outcomes = {}
    for alpha in (0, 1, 2):
        with contribution_hook(component, 2, vector, mean, alpha=alpha):
            outcomes[alpha] = component(x)
    assert torch.equal(outcomes[1], baseline)
    assert torch.allclose(outcomes[0][:, 2] + outcomes[2][:, 2], 2 * baseline[:, 2], atol=2e-6)
    assert torch.equal(outcomes[0][:, [0, 1, 3]], baseline[:, [0, 1, 3]])
    with contribution_hook(component, 2, vector, mean, coordinate=3):
        local = component(x)
    assert torch.equal(local[:, 2, [0, 1, 2, 4, 5, 6]], baseline[:, 2, [0, 1, 2, 4, 5, 6]])
    with pytest.raises(RuntimeError):
        with contribution_hook(component, 2, vector, mean):
            raise RuntimeError('test cleanup')
    assert torch.equal(component(x), baseline)


def test_effect_variant_counts_and_prior_driver_fail_closed(tmp_path, monkeypatch):
    # Import orchestration without constructing a model or GPU context.
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('e08_pipeline', root / 'experiments/e08_pipeline.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.VRAM.name == 'vramorama.exe'
    assert module.PD_PY.name == 'python.exe'
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    def forbidden():
        raise AssertionError('process or GPU action before prerequisite files')
    monkeypatch.setattr(module, 'active_predecessors', forbidden)
    module.run(False)
    monkeypatch.syspath_prepend(str(root / 'experiments'))
    import e08_effects
    settings = e08_effects.variants({'groups': [[1], [2]],
                                   'random_groups': [[[3]] * 100, [[4]] * 100]})
    assert len(settings) == len({v['name'] for v in settings}) == 207
    assert sum(v['name'].startswith('random') for v in settings) == 200
    assert sum(v['coordinate'] for v in settings) == 2


def test_semantic_review_is_required_and_unseen_evidence_is_rejected(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'experiments'))
    import e08_prepare
    monkeypatch.setattr(e08_prepare, 'ROOT', tmp_path)
    assert not e08_prepare.freeze_groups()
    atomic_json(tmp_path / 'out/e08/labels.json', {'reviewer': 'test'})
    with pytest.raises(ValueError, match='reviewer/method/cases'):
        e08_prepare.freeze_groups()
    monkeypatch.setattr(e08_prepare, 'CheckpointArrays', lambda _: None)
    atomic_json(tmp_path / 'results/e05b_retrieval.json', {'runs': {'0': {'run_id': 'a'}, '1': {'run_id': 'b'}}})
    atomic_json(tmp_path / 'out/e08/proposals.json', {'seed0_candidate0': {'status': 'proposed', 'review_case': 'case'}})
    atomic_json(tmp_path / 'out/e08/blinded_contexts.json', {'cases': {
        'case': {str(k): {'examples': [{'discovery_index': 1}], 'discovery_count': 32} for k in range(4)}}})
    annotation = {'coherent': True, 'family': 'unverified', 'confirmed_indices': [999], 'rationale': 'test'}
    atomic_json(tmp_path / 'out/e08/labels.json', {'reviewer': 'test', 'method': 'blinded',
        'cases': {'case': {str(k): annotation for k in range(4)}}})
    with pytest.raises(ValueError, match='unseen discovery context'):
        e08_prepare.freeze_groups()
