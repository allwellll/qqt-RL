"""R5 serial boundary, same raw parent, single-factor and runtime wiring."""
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import retention_phase_screen as screen


def prior_fixture(monkeypatch, tmp_path, candidate='A0_or_no_improving_candidate'):
    monkeypatch.setattr(screen.engine, 'ROOT', tmp_path)
    monkeypatch.setattr(screen, 'ROOT', tmp_path)
    monkeypatch.setattr(screen, 'BASE', tmp_path/'runs/r5')
    monkeypatch.setattr(screen, 'PRIOR', tmp_path/'runs/r4')
    monkeypatch.setattr(screen, 'build_probe', lambda source:None)
    screen.PRIOR.mkdir(parents=True)
    source = screen.PRIOR/'A0_source'; (source/'inputs').mkdir(parents=True)
    p = {'state_w': np.ones((24, 8), np.float32), 'state_b': np.zeros(8)}
    for name in ['initial.pkl', 'retention_parent.pkl']:
        (source/'inputs'/name).write_bytes(pickle.dumps(dict(actor=p, critic=p)))
    for name in ['initial_pool.pkl', 'frozen.pkl']:
        (source/'inputs'/name).write_bytes(pickle.dumps(dict(parent=p)))
    for name in ['return_replay.pkl', 'RETURN_REPLAY.json']:
        (source/'inputs'/name).write_bytes(name.encode())
    (source/'plan.json').write_text(json.dumps(dict(seed=0, generations=32,
        checkpoints=[0,8,16,24,32], state_schema='old', retention_core=True,
        core_error_priority=False, core_history_fraction=.5, certified_return_ppo_factor=1.,
        parent_selection=dict(checkpoint='raw_parent',policy_sha256='raw_sha'))))
    (source/'regions.json').write_text('{}')
    runs=[]
    for i in range(6):
        run=source if i==0 else screen.PRIOR/f'run{i}'
        run.mkdir(exist_ok=True)
        (run/'FIXED_RUN_TERMINAL.json').write_text('{"generation":32,"atomic_stop":false}')
        (run/'GPU_RELEASE_RECEIPT.json').write_text('{"released":true}')
        for folder,points in [('authority_je10',[0,8,16,24,32]),('authority_retention_test',[0,32])]:
            for g in points:
                out=run/folder/f'g{g:06d}';out.mkdir(parents=True)
                (out/'COMPLETE.json').write_text('{"passed":true}')
        runs.append(dict(namespace=str(run.relative_to(tmp_path))))
    (screen.PRIOR/'DECLARATION.json').write_text(json.dumps(dict(runs=runs)))
    (screen.PRIOR/'EVALUATION_COMPLETE.json').write_text('{"passed":true,"completed":'+json.dumps(list(range(30)))+'}')
    (screen.PRIOR/'SCREEN_COMPLETE.json').write_text('{"passed":true}')
    (screen.PRIOR/'FULL_COMPARISON_20261021.json').write_text(json.dumps(dict(candidate=candidate)))
    return source


def test_no_r5_namespace_until_all_r4_training_and_holdout_complete(monkeypatch,tmp_path):
    source=prior_fixture(monkeypatch,tmp_path)
    (screen.PRIOR/'EVALUATION_COMPLETE.json').unlink()
    with pytest.raises(AssertionError):screen.prepare()
    assert not screen.BASE.exists()
    (screen.PRIOR/'EVALUATION_COMPLETE.json').write_text('{"passed":true,"completed":[]}')
    with pytest.raises(AssertionError):screen.prepare()
    assert not screen.BASE.exists()


@pytest.mark.parametrize('candidate,priority',[
    ('A0_or_no_improving_candidate',False),('A1_pending_confirmation',True)])
def test_matrix_has_only_phase_factor_and_shared_raw_parent(monkeypatch,tmp_path,candidate,priority):
    source=prior_fixture(monkeypatch,tmp_path,candidate)
    screen.prepare()
    d=json.loads((screen.BASE/'DECLARATION.json').read_text())
    assert d['actual_fresh_ticks_budget']==6*32*2097152
    assert d['shared_core_error_priority']==priority
    assert len(d['train_seeds'])==3 and len(set(d['train_seeds']))==3
    assert not set(d['route_test_seeds'])&set(d['route_dev_seeds'])
    assert d['parent']=='raw_parent'
    for a,b in zip(d['runs'][::2],d['runs'][1::2]):
        assert a['seed']==b['seed'] and a['parent']==b['parent']
        assert not a['phase_conditioning'] and b['phase_conditioning']
        pair=[]
        for spec in [a,b]:
            run=tmp_path/spec['namespace'];plan=json.loads((run/'plan.json').read_text())
            assert plan['core_error_priority']==priority
            assert all(not list((run/x).iterdir()) for x in ['state','formal','cache'])
            pair.append({k:v for k,v in plan.items() if k not in
                         ['phase_conditioning','state_schema','experiment_id']})
            for name in ['return_replay.pkl','RETURN_REPLAY.json']:
                assert (run/'inputs'/name).read_bytes()==(source/'inputs'/name).read_bytes()
            for name in ['initial.pkl','retention_parent.pkl','initial_pool.pkl','frozen.pkl']:
                raw=pickle.loads((run/'inputs'/name).read_bytes())
                weights=raw.get('actor',raw.get('parent'))['state_w']
                assert weights.shape[-2]==(29 if spec['phase_conditioning'] else 24)
                np.testing.assert_array_equal(weights[:24],1)
                if spec['phase_conditioning']:np.testing.assert_array_equal(weights[24:],0)
        assert pair[0]==pair[1]
    with pytest.raises(AssertionError):screen.prepare()


def test_worker_uses_new_process_without_old_compile_cache(monkeypatch,tmp_path):
    monkeypatch.setattr(screen,'ROOT',tmp_path);monkeypatch.setattr(screen,'BASE',tmp_path/'r5')
    monkeypatch.setattr(screen.engine,'require_training_authorized',lambda run:None)
    monkeypatch.setattr(screen.engine,'env_for',lambda run,cpu:dict(CUDA_VISIBLE_DEVICES='' if cpu else 'all'))
    calls=[]
    monkeypatch.setattr(screen.subprocess,'Popen',lambda args,**kw:calls.append((args,kw)) or SimpleNamespace())
    run=tmp_path/'r5/A1';run.mkdir(parents=True)
    screen.launch_worker(run)
    args,kw=calls[0]
    assert screen.SCRIPT in args and kw['start_new_session']
    assert 'NODE_OPTIONS' in kw['env'] and 'retention_phase_conditioning.js' in kw['env']['NODE_OPTIONS']
    assert not (run/'cache').exists()


def test_evaluation_child_requires_cpu_and_keeps_new_holdout(monkeypatch,tmp_path):
    monkeypatch.setattr(screen,'ROOT',tmp_path)
    monkeypatch.setattr(screen,'declaration',lambda:dict(dev_rng=20261019,test_rng=20261106,
        route_dev_seeds=[2026102100],route_test_seeds=[2026110700],evaluation_report='reports/r5.md'))
    monkeypatch.setattr(screen.engine,'env_for',lambda run,cpu:dict(CUDA_VISIBLE_DEVICES='' if cpu else 'all'))
    calls=[]
    monkeypatch.setattr(screen.subprocess,'run',lambda args,**kw:calls.append((args,kw)) or SimpleNamespace(returncode=0))
    run=tmp_path/'A1';run.mkdir()
    assert screen.evaluation_job(run,'test',32)==0
    args,kw=calls[0]
    assert screen.SCRIPT in args and kw['env']['CUDA_VISIBLE_DEVICES']==''
    assert kw['env']['JE10_EVAL_SEED']=='20261106'
    assert kw['env']['JE10_ROUTE_SEED_START']=='2026110700'
    assert 'retention_phase_conditioning.js' in kw['env']['NODE_OPTIONS']
