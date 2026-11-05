import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('e11',ROOT / 'experiments' / 'e11_spectral_geometry.py')
e11=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e11)


def fixture():
    modules=np.array(['block0']*6)
    local=np.arange(6)
    nuisance=np.tile([1.,1.,1.,1.,.3],(6,1))
    # 3 distinct task responses, paired with identical nuisance features.
    signals=np.eye(3)[:,[0,0,1,1,2,2]]
    return modules,local,nuisance,signals


def test_spectral_separates_useful_directions_beyond_identical_degree_ci():
    modules,local,nuisance,signals=fixture()
    selects,diag=e11.geometry_plan(signals,nuisance,modules,local,{'block0':2},n_null=5)
    base=selects['joint_degree_ci']['block0']
    new=selects['spectral_geometry']['block0']
    assert base==[0,1]
    assert len({x//2 for x in new})==2
    assert len({x//2 for x in base})==1
    assert diag['spectral_replacements']==1
    assert len({tuple(v['block0']) for v in selects.values()})>=2


def test_null_indistinguishable_profiles_yield_baseline_selector():
    modules,local,nuisance,signals=fixture()
    flat=np.tile(np.array([[1.],[2.],[3.]]),(1,6))
    selects,diag=e11.geometry_plan(flat,nuisance,modules,local,{'block0':2},n_null=0)
    assert diag['nonzero_residual_profiles']==0
    assert selects['spectral_geometry']==selects['joint_degree_ci']


def test_rejects_mismatched_vpd_and_leaky_split():
    modules,local,nuisance,signals=fixture()
    base={'schema':'percinterp.e10.selector.v1','graph_sha256':'frozen',
          'source_type':'E03_VPD_SAVED_INPUTS','module_quotas':{'block0':2},
          'provenance':{'vpd_run':'same'},
          'selection':{'sieve':{'block0':[0,1]}}}
    data={'responses':signals,'nuisance':nuisance,'modules':modules,'local_ids':local,
          'response_source':np.array('vpd_response_proxy'),
          'calibration_split':np.array('test'), 'graph_sha256':np.array('frozen'),
          'vpd_run':np.array('same')}
    with pytest.raises(ValueError,match='train split'):
        e11.derive_manifest(base,data,'vpd_response_proxy',digest='x',n_null=0,seed=1,bins=3)
    data['calibration_split']=np.array('train')
    data['vpd_run']=np.array('other')
    with pytest.raises(ValueError,match='VPD run'):
        e11.derive_manifest(base,data,'vpd_response_proxy',digest='x',n_null=0,seed=1,bins=3)


def test_valid_manifest_adds_selection_without_mutating_e10():
    modules,local,nuisance,signals=fixture()
    base={'schema':'percinterp.e10.selector.v1','graph_sha256':'frozen',
          'source_type':'E03_VPD_SAVED_INPUTS','module_quotas':{'block0':2},
          'provenance':{'vpd_run':'same'},
          'selection':{'sieve':{'block0':[0,1]}}}
    data={'responses':signals,'nuisance':nuisance,'modules':modules,'local_ids':local,
          'response_source':np.array('vpd_response_proxy'),
          'calibration_split':np.array('train'), 'graph_sha256':np.array('frozen'),
          'vpd_run':np.array('same')}
    result=e11.derive_manifest(base,data,'vpd_response_proxy',digest='abc',n_null=2,seed=1,bins=3)
    assert len(result['selection'])==5
    assert len(base['selection'])==1
    for v in result['selection'].values():
        assert len(v['block0'])==2

SPEC2=importlib.util.spec_from_file_location('e11_score',ROOT / 'experiments' / 'e11_spectral_score.py')
scorer=importlib.util.module_from_spec(SPEC2)
SPEC2.loader.exec_module(scorer)


def test_scorer_rejects_synthetic_and_unmatched_manifest():
    m={'e11_response_geometry':{'response_source':'synthetic_fixture'},'selection':{}}
    with pytest.raises(ValueError,match='synthetic'):
        scorer.score({},m)
    m['e11_response_geometry']['response_source']='vpd_response_proxy'
    with pytest.raises(ValueError,match='actual E10'):
        scorer.score({},m)


def test_scorer_reports_incremental_gain_with_quality_gate():
    import hashlib,json
    selectors={k:{'block0':[0]} for k in scorer.PRIMARY}
    selectors['spectral_geometry']={'block0':[1]}
    selectors['spectral_perm_000']={'block0':[2]}
    m={'e11_response_geometry':{'response_source':'vpd_response_proxy'},'selection':selectors}
    mh=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    kl={k:[1.4]*20 for k in scorer.PRIMARY}
    kl.update({'spectral_geometry':[1.0]*20,'spectral_perm_000':[1.6]*20,
               'all':[.1]*20,'none':[5.0]*20})
    record={'schema':'percinterp.e10.causal.v1', 'source_type':'E03_VPD_SAVED_INPUTS',
            'manifest_sha256':mh,'per_sequence_kl':kl}
    report=scorer.score(record,m,draws=200)
    assert report['status']=='WITHIN_TARGET_SEED_PASS'
    assert report['conditional_permutation_p']==.5
    assert all(report['results'][key]['passes'] for key in scorer.PRIMARY)
