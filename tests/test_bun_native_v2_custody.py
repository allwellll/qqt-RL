"""Six mandatory custody cases, plus version isolation and terminal-tail tests."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.population_run_state import sha

ROOT=Path('runs/bun_native_v2_20261030/custody_v2_tests')


@pytest.fixture(scope='module')
def evidence():
    result=json.loads((ROOT/'ACCEPTANCE.json').read_text())
    for r in result['records']:
        assert not r['mismatch'] and r['ticks']==r['expected_ticks']
        assert sha(Path(r['raw_rewards']))==r['raw_sha256']
        assert sha(ROOT/f"P{r['seat']}_{r['mode']}.json")==r['fixture_sha256']
    return result


def test_1_real_acquisition_intercept_return_cycle(evidence):
    for r in evidence['records']:
        if r['mode']=='dropcycles':
            assert r['interceptions']
            assert abs(r['restores'][0]['custody'])<=1e-6
            assert r['restores'][0]['raw_exact_cycle']<=1e-6
            assert all(x['raw_exact_cycle']<=1e-6 for x in r['restores'])


def test_2_five_cycles_no_raw_gain(evidence):
    for r in evidence['records']:
        if r['mode'] in ('cycles','dropcycles'):
            assert len(r['restores'])==5
            assert abs(r['custody_delta'][r['seat']])<=5e-6
            assert r['real_composed_reward'][r['seat']]<=5e-6


def test_3_enemy_origin_delivery_positive(evidence):
    for r in evidence['records']:
        if r['mode']=='delivery':
            assert r['real_composed_reward'][r['seat']]>0
            assert r['terminal'][r['seat']]==20
            assert r['terminal'][1-r['seat']]==-20


def test_4_far_empty_ordinary_kill_unchanged(evidence):
    assert evidence['ordinary_kill_new']==evidence['ordinary_kill_R1']==16
    reference=json.loads((ROOT/'ORDINARY_old.json').read_text());new=json.loads((ROOT/'ORDINARY_new.json').read_text())
    assert reference['status']==new['status']=='PASS'
    assert reference['records']==new['records']


def test_5_late_interception_more_immediate_gain(evidence):
    m=evidence['monotonic']
    assert m['q08_intercept']>m['q02_intercept']>0


def test_6_allow_delivery_worse_than_intercept(evidence):
    for seat in (0,1):
        loss=next(r for r in evidence['records'] if r['seat']==seat and r['mode']=='loss')
        defense=next(r for r in evidence['records'] if r['seat']==seat and r['mode']=='dropcycles')
        assert loss['terminal'][seat]==-20
        assert loss['real_composed_reward'][seat]<defense['restores'][0]['raw_exact_cycle']


def test_conservative_delta_no_waiting_drift():
    from scripts.bun_native_v2_custody_reward import delta
    phi=np.array([0.,-.3,-.55,-.55,-.15,-.15,0.],np.float32)
    assert abs(float(np.asarray(delta(phi[:-1],phi[1:])).sum()))<1e-7
    assert np.asarray(delta(-.55,-.55))==0
    assert np.asarray(delta(-.8,0.,True))==0


def test_new_worker_is_opt_in_and_old_seals_intact():
    from scripts.bun_native_v2_custody_train import worker_source
    compile(worker_source(),'<custody-v2>','exec')
    assert 'bun_native_v2_custody_collector as M' in worker_source()
    p=json.loads(Path('runs/bun_native_v2_20261030/protocols/r2_a_c000000.json').read_text())
    assert all(sha(Path(f))==h for f,h in p['source_sha256'].items())
