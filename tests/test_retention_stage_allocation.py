"""R7 changes only accepted ordinary route stage allocation, never row authority."""
import pickle
import jax
import jax.numpy as jnp
import numpy as np


def rows():
    n = 32
    return dict(valid=jnp.ones(n, bool), identity=jnp.arange(n),
                kind=jnp.array([1]*4+[2]*4+[3]*8+[2]*16),
                preparation=jnp.arange(n)>=16, opening=jnp.arange(n)==31,
                round=jnp.full(n, 4),
                actual_custody=jnp.arange(n)<16,
                joint_target=jnp.ones((n,5,3),bool))


def test_half_return_changes_only_ordinary_quota_and_payload_is_authoritative():
    from scripts.retention_stage_allocation import choose
    from scripts.phase_balanced_joint_replay import choose as baseline
    d=rows();key=jax.random.PRNGKey(2)
    a=baseline(d,key,64);b=jax.jit(lambda x,k:choose(x,k,64))(d,key)
    assert int((a['valid']&~a['preparation']&(a['kind']==3)).sum())==24
    assert int((b['valid']&~b['preparation']&(b['kind']==3)).sum())==16
    for kind in [1,2]:
        assert int((b['valid']&~b['preparation']&(b['kind']==kind)).sum())==8
    assert int(b['opening'].sum())==2
    assert int((b['preparation']&~b['opening']).sum())==30
    for k,v in b.items():np.testing.assert_array_equal(v,d[k][b['identity']])
    for k in a:np.testing.assert_array_equal(a[k][32:],b[k][32:])


def test_absent_stage_fallback_invalid_rows_and_restore():
    from scripts.retention_stage_allocation import choose
    from scripts.phase_balanced_joint_replay import choose as baseline
    d=rows();k=jax.random.PRNGKey(3)
    d['valid']=d['kind']!=3
    got=choose(d,k,64)
    assert bool(got['valid'].all()) and not bool((got['kind']==3).any())
    d['preparation']=jnp.zeros(32,bool);d['opening']=jnp.zeros(32,bool)
    for name,v in choose(d,k,64).items():np.testing.assert_array_equal(v,baseline(d,k,64)[name])
    saved=pickle.loads(pickle.dumps(d))
    for name,v in choose(saved,k,64).items():np.testing.assert_array_equal(v,choose(d,k,64)[name])
    d['valid']=jnp.zeros(32,bool)
    assert not bool(choose(d,k,64)['valid'].any())


def test_install_targets_actual_sampler_without_loss_or_capacity_changes(monkeypatch):
    from scripts.retention_stage_allocation import install,choose
    from scripts import return_error_replay as replay
    from scripts import je09_stage_h5c_dagger as dagger
    monkeypatch.setattr(replay,'baseline_choose',replay.baseline_choose)
    before=(dagger.CAPACITY,dagger.ADDITIONS,dagger.COEFFICIENT,dagger.replay_loss)
    receipt=install(.5)
    assert replay.baseline_choose is choose
    assert before==(dagger.CAPACITY,dagger.ADDITIONS,dagger.COEFFICIENT,dagger.replay_loss)
    assert receipt['reward_changed'] is False and receipt['loss_changed'] is False


def test_actual_error_replay_call_keeps_single_factor_quota(monkeypatch):
    from scripts.retention_stage_allocation import install
    from scripts import return_error_replay as replay
    monkeypatch.setattr(replay,'baseline_choose',replay.baseline_choose)
    monkeypatch.setattr(replay,'target_mass',lambda actor,d:jnp.full(d['valid'].size,.5))
    install(.5)
    got=replay.choose({},rows(),jax.random.PRNGKey(10),64)
    assert int((got['valid']&~got['preparation']&(got['kind']==3)).sum())==16
    for kind in [1,2]:
        assert int((got['valid']&~got['preparation']&(got['kind']==kind)).sum())==8


def test_queue_skips_completed_budget_without_evaluation_gate(tmp_path,monkeypatch):
    from scripts import retention_continuous_screen as screen
    monkeypatch.setattr(screen,'ROOT',tmp_path)
    (tmp_path/'done').mkdir();(tmp_path/'done/TRAINING_COMPLETE.json').write_text('{}')
    items=[dict(namespace='done'),dict(namespace='next')]
    assert screen.pending(items)==[dict(namespace='next')]


def test_prepare_script_entry_imports_without_pythonpath():
    import os, subprocess, sys
    env=dict(os.environ)
    env.pop('PYTHONPATH',None)
    subprocess.run([sys.executable,'scripts/retention_continuous_prepare.py'],
                   env=env,check=True,capture_output=True)


def test_parent_holdout_reuse_refuses_different_initial_weights(tmp_path,monkeypatch):
    import pytest
    from scripts import retention_continuous_prepare as preparation
    monkeypatch.setattr(preparation,'ROOT',tmp_path)
    source=tmp_path/'runs/retention_20261022/r6/A0_s2026102210/inputs'
    source.mkdir(parents=True);(source/'initial.pkl').write_bytes(b'parent-a')
    run=tmp_path/'candidate';(run/'inputs').mkdir(parents=True)
    (run/'inputs/initial.pkl').write_bytes(b'different-parent-b')
    with pytest.raises(AssertionError):preparation.share_parent_holdout(run)
    assert not (run/'authority_retention_test').exists()


def test_owner_claims_next_run_without_waiting_for_evaluator(tmp_path,monkeypatch):
    import json
    import pytest
    from scripts import retention_continuous_screen as screen
    from scripts import retention_continuous_prepare as preparation
    from scripts.population_run_state import atomic_json
    monkeypatch.setattr(screen,'ROOT',tmp_path)
    monkeypatch.setattr(screen,'BASE',tmp_path)
    items=[dict(namespace=n,kind='observation',generations=1) for n in ['r0','r1']]
    for item in items:
        r=tmp_path/item['namespace'];r.mkdir();atomic_json(r/'PREPARED.json',dict(passed=True))
    atomic_json(tmp_path/'RESOURCE.json',dict(idle=True))
    monkeypatch.setattr(screen,'specs',lambda:items)
    monkeypatch.setattr(screen,'guard',lambda *a:None)
    monkeypatch.setattr(screen,'verify_seal',lambda *a:None)
    monkeypatch.setattr(screen.recipe.engine,'wait_idle',lambda:None)
    monkeypatch.setattr(screen.recipe.engine,'validate_update',lambda r:None)
    monkeypatch.setattr(screen.recipe,'_checkpoint_receipt',lambda *a:dict(passed=True))
    monkeypatch.setattr(screen.recipe.engine,'env_for',lambda *a:{})
    notices=[];launched=[]
    monkeypatch.setattr(screen,'notify',notices.append)
    class Proc:
        returncode=0
        def poll(self):return 0
        def wait(self):raise AssertionError('CPU evaluation must never block training succession')
    monkeypatch.setattr(screen.subprocess,'Popen',lambda *a,**kw:Proc())
    def spawn(run,**kw):
        launched.append(run.name)
        atomic_json(run/'TRAINING_COMPLETE.json',dict(passed=True))
        out=run/'formal'/screen.ARM
        atomic_json(out/'TRAINER_EXIT.json',dict(generation=1,atomic_stop=False))
        atomic_json(out/'STATE.json',dict(finite=True))
        return Proc()
    monkeypatch.setattr(screen,'spawn',spawn)
    def exhausted():raise RuntimeError('two fixed runs consumed')
    monkeypatch.setattr(preparation,'append_observation',exhausted)
    with pytest.raises(RuntimeError,match='two fixed runs consumed'):screen.owner()
    assert launched==['r0','r1'] and len(notices)==2
    assert json.loads((tmp_path/'r1/GPU_RELEASE_RECEIPT.json').read_text())['released']
