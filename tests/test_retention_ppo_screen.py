"""Protect paired fresh initialization and CPU-only authority handoff."""
import json
from types import SimpleNamespace
import pytest
from scripts import retention_ppo_screen as screen


def test_prepare_requires_frozen_result_and_never_reuses_namespace(monkeypatch,tmp_path):
    monkeypatch.setattr(screen,'ROOT',tmp_path)
    monkeypatch.setattr(screen,'BASE',tmp_path/'runs/r3')
    monkeypatch.setattr(screen,'R2',tmp_path/'runs/r2')
    with pytest.raises(AssertionError):screen.prepare()
    assert not screen.BASE.exists()
    screen.BASE.mkdir(parents=True)
    with pytest.raises(AssertionError,match='fresh R3'):screen.prepare()


@pytest.mark.parametrize('candidate,fraction',[
    ('A1_pending_confirmation',1.),('A0_or_no_improving_candidate',.5)])
def test_prepared_matrix_is_paired_single_factor_and_preserves_raw_parent(monkeypatch,tmp_path,candidate,fraction):
    monkeypatch.setattr(screen,'ROOT',tmp_path)
    monkeypatch.setattr(screen,'BASE',tmp_path/'runs/retention_20261021/r3')
    monkeypatch.setattr(screen,'R2',tmp_path/'runs/retention_20261020/r2')
    source=screen.R2/'A0_s2026102001'
    (source/'inputs').mkdir(parents=True)
    for name in ['initial.pkl','retention_parent.pkl','initial_pool.pkl','return_replay.pkl','RETURN_REPLAY.json']:
        (source/'inputs'/name).write_bytes(name.encode())
    (source/'plan.json').write_text(json.dumps(dict(generations=32,checkpoints=[0,8,16,24,32],
        seed=2026102001,retention_core=True,core_history_fraction=.5,native_return_retention=True)))
    (source/'regions.json').write_text('{}')
    (screen.R2/'SCREEN_COMPLETE.json').write_text('{}')
    (screen.R2/'FULL_COMPARISON_20261021.json').write_text(json.dumps(dict(candidate=candidate)))
    screen.prepare()
    d=json.loads((screen.BASE/'DECLARATION.json').read_text())
    assert d['actual_fresh_ticks_budget']==6*32*2097152
    assert not d['reward_hypotheses_installed'] and not d['synthetic_training_carry']
    assert d['test_rng']==20261102 and len(d['route_test_seeds'])==64
    assert not set(d['route_test_seeds'])&set(d['route_dev_seeds'])
    for a,b in zip(d['runs'][::2],d['runs'][1::2]):
        assert a['seed']==b['seed'] and a['parent']==b['parent']
        assert a['core_history_fraction']==b['core_history_fraction']==fraction
        assert a['certified_return_ppo_factor']==1. and b['certified_return_ppo_factor']==.5
        plans=[]
        for spec in [a,b]:
            run=tmp_path/spec['namespace']
            assert all(not list((run/folder).iterdir()) for folder in ['state','formal','cache'])
            for p in (source/'inputs').iterdir():assert (run/'inputs'/p.name).read_bytes()==p.read_bytes()
            plan=json.loads((run/'plan.json').read_text())
            assert plan['generations']==32 and plan['native_return_retention']
            plans.append({k:v for k,v in plan.items() if k not in ['experiment_id','state_schema','certified_return_ppo_factor']})
        assert plans[0]==plans[1]


def test_holdout_evaluation_is_cpu_only_and_uses_fresh_fixed_seeds(monkeypatch,tmp_path):
    monkeypatch.setattr(screen,'ROOT',tmp_path)
    monkeypatch.setattr(screen,'declaration',lambda:dict(evaluation_report='reports/evaluation.md',
        dev_rng=20261019,test_rng=20261102,route_dev_seeds=[2026102100],route_test_seeds=[2026110300]))
    monkeypatch.setattr(screen.engine,'env_for',lambda run,cpu:dict(CUDA_VISIBLE_DEVICES='' if cpu else 'all'))
    calls=[]
    monkeypatch.setattr(screen.subprocess,'run',lambda args,**kw:calls.append((args,kw)) or SimpleNamespace(returncode=0))
    run=tmp_path/'run';run.mkdir()
    assert screen.evaluation_job(run,'test',32)==0
    args,kw=calls[0]
    assert args[1]==screen.SCRIPT and kw['env']['CUDA_VISIBLE_DEVICES']==''
    assert kw['env']['JE10_EVAL_SEED']=='20261102'
    assert kw['env']['JE10_ROUTE_SEED_START']=='2026110300'
    assert kw['env']['JE10_ROUTE_GAMES']=='64'


@pytest.mark.parametrize('corruption',[None,'state','policy','adam','schema','plan'])
def test_atomic_checkpoint_receipt_rejects_corruption(tmp_path,corruption):
    cp=tmp_path/'formal'/screen.ARM/'gen000001';cp.mkdir(parents=True)
    plan=tmp_path/'plan.json';plan.write_text('{"state_schema":"new-r3"}')
    (cp/'state.pkl').write_bytes(b'state');(cp/'policy.pt').write_bytes(b'policy')
    m=dict(passed=True,replicas_exact=True,devices=8,generation=1,adam_steps=[128]*8,
        state_schema='new-r3',plan_sha256=screen.sha(plan),
        state_sha256=screen.sha(cp/'state.pkl'),policy_sha256=screen.sha(cp/'policy.pt'))
    if corruption in ['state','policy']:
        (cp/('state.pkl' if corruption=='state' else 'policy.pt')).write_bytes(b'corrupt')
    elif corruption=='adam':m['adam_steps'][3]=0
    elif corruption=='schema':m['state_schema']='excluded-old'
    elif corruption=='plan':plan.write_text('{"state_schema":"new-r3","modified":true}')
    (cp/'COMMITTED.json').write_text(json.dumps(m))
    if corruption:
        with pytest.raises(AssertionError):screen.checkpoint_receipt(tmp_path,1)
    else:assert screen.checkpoint_receipt(tmp_path,1)['atomic_sha_verified']


def test_smoke_reuse_copies_only_executables_into_independent_arm_cache(monkeypatch,tmp_path):
    monkeypatch.setattr(screen,'ROOT',tmp_path);monkeypatch.setattr(screen,'BASE',tmp_path/'runs/r3')
    smoke=screen.BASE/'smoke_A1';gpu=smoke/'cache/gpu';gpu.mkdir(parents=True)
    (smoke/'TECHNICAL_COMPLETE.json').write_text('{"passed":true}')
    (gpu/'jit-example-cache').write_bytes(b'executable')
    (gpu/'state.pkl').write_bytes(b'never copy learner state')
    run=screen.BASE/'A1_seed';run.mkdir()
    screen.reuse_smoke_compilation(run,'A1')
    assert (run/'cache/gpu/jit-example-cache').read_bytes()==b'executable'
    assert not (run/'cache/gpu/state.pkl').exists()
    assert not (run/'state').exists() and not (run/'formal').exists()
    (gpu/'jit-example-cache').write_bytes(b'changed source')
    assert (run/'cache/gpu/jit-example-cache').read_bytes()==b'executable'
    assert not json.loads((run/'SMOKE_COMPILE_CACHE_REUSE.json').read_text())['learner_state_copied']


def test_baseline_cache_warming_never_copies_state_or_overwrites(monkeypatch,tmp_path):
    monkeypatch.setattr(screen,'ROOT',tmp_path);monkeypatch.setattr(screen,'R2',tmp_path/'r2')
    source=screen.R2/'smoke_A0/cache/gpu';source.mkdir(parents=True)
    (source/'jit-example-cache').write_bytes(b'old-executable')
    (source/'state.pkl').write_bytes(b'forbidden')
    run=tmp_path/'r3/smoke_A1';(run/'cache/gpu').mkdir(parents=True)
    (run/'cache/gpu/jit-example-cache').write_bytes(b'new-executable')
    screen.seed_baseline_compilation(run)
    assert (run/'cache/gpu/jit-example-cache').read_bytes()==b'new-executable'
    assert not (run/'cache/gpu/state.pkl').exists()
    assert not json.loads((run/'BASELINE_COMPILE_CACHE_REUSE.json').read_text())['learner_state_copied']
