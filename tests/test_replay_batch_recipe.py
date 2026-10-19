"""Exercise the real aggregator in isolation; never patch a live worker."""
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def candidate_namespace():
    from scripts.replay_batch_recipe import patched_replay
    source=Path('scripts/return_error_replay.py').read_text()
    namespace={}
    exec(compile(patched_replay(source),'isolated-replay-batch','exec'),namespace)
    namespace['target_mass']=lambda actor,data:jnp.full(data['valid'].shape,.5)
    return namespace


def mixed_batch(n=256):
    from test_prestock_joint_replay import batch
    original=batch()
    obs=jnp.arange(n,dtype=jnp.float32).reshape(1,n,1,1,1)
    gv=jnp.stack([jnp.arange(n),jnp.arange(n)+10],-1)[None].astype(jnp.float32)
    mm=jnp.ones((1,n,5),bool);am=jnp.ones((1,n,3),bool)
    groups=jnp.arange(n)%8
    kind=jnp.where(groups<3,3,jnp.where(groups==3,1,2))
    ctx=jnp.tile(original[-1][:,:1],(1,n,1)).at[:,:,54].set(kind)
    prep=groups>=4;opening=groups==7
    metadata=jnp.stack([prep,jnp.arange(n),kind,opening],-1)[None]
    return (obs,gv,None,None,None,None,(mm,am),jnp.concatenate([ctx,metadata],-1))


def test_production_branch_doubles_slots_preserves_fifo_labels_phases_and_restore(monkeypatch):
    from scripts import je09_stage_h5c_dagger as legacy,return_error_replay as prior
    from scripts.prestock_joint_replay import buffer
    # Exercise the production capacity branch with small observations only.
    monkeypatch.setattr(legacy,'CAPACITY',256)
    monkeypatch.setattr(legacy,'ADDITIONS',128)
    monkeypatch.setattr(prior,'target_mass',lambda actor,data:jnp.full(data['valid'].shape,.5))
    aggregate=jax.jit(candidate_namespace()['aggregate'])
    old=buffer(capacity=256,additions=128,obs_shape=(1,1,1),global_size=2)
    for generation in [1,2,3]:
        key=jax.random.PRNGKey(generation)
        expected,_,_=prior.aggregate(old,mixed_batch(),key,jnp.int32(generation),{})
        updated,replay,stats=aggregate(old,mixed_batch(),key,jnp.int32(generation),{})
        assert replay['valid'].shape==(128,), '64 current plus 64 history native slots required'
        for name in expected:np.testing.assert_array_equal(updated[name],expected[name])
        for half,part in [('current',slice(0,64)),('history',slice(64,128))]:
            for group,count in [('valid',64),('ordinary',32),('returning',24),('opening',2),('navigation',30)]:
                assert int(stats['replay_'+half+'_'+group])==count
            for row in range(part.start,part.stop):
                index=int(replay['obs'][row,0,0,0])
                np.testing.assert_array_equal(replay['joint_target'][row],mixed_batch()[-1][0,index,95:110].reshape(5,3)>0)
        assert bool((replay['round'][:64]==generation).all())
        if generation==1:
            for leaf in replay.values():np.testing.assert_array_equal(leaf[:64],leaf[64:])
        else:assert bool((replay['round'][64:]<generation).all())
        restored=pickle.loads(pickle.dumps(jax.device_get(old)))
        repeated=aggregate(restored,mixed_batch(),key,jnp.int32(generation),{})
        for a,b in zip(jax.tree.leaves((updated,replay,stats)),jax.tree.leaves(repeated)):
            np.testing.assert_array_equal(a,b)
        old=updated


def test_patch_is_idempotent_and_rejects_unknown_dispatch():
    from scripts.replay_batch_recipe import patched_replay
    source=Path('scripts/return_error_replay.py').read_text()
    patched=patched_replay(source)
    assert patched_replay(patched)==patched
    with pytest.raises(AssertionError):patched_replay('def aggregate(): pass')


def test_missing_current_labels_fall_back_to_valid_history_at_expanded_shape(monkeypatch):
    from scripts import je09_stage_h5c_dagger as legacy
    from scripts.prestock_joint_replay import buffer
    monkeypatch.setattr(legacy,'CAPACITY',256);monkeypatch.setattr(legacy,'ADDITIONS',128)
    aggregate=jax.jit(candidate_namespace()['aggregate'])
    old=buffer(capacity=256,additions=128,obs_shape=(1,1,1),global_size=2)
    key=jax.random.PRNGKey(14)
    old,_,_=aggregate(old,mixed_batch(),key,jnp.int32(1),{})
    batch=list(mixed_batch());batch[-1]=batch[-1].at[:,:,53].set(0)
    _,replay,_=aggregate(old,tuple(batch),key,jnp.int32(2),{})
    assert replay['valid'].shape==(128,)
    assert bool(replay['valid'].all()) and bool((replay['round']==1).all())
    for value in replay.values():np.testing.assert_array_equal(value[:64],value[64:])


@pytest.mark.parametrize('generation,released,atomic_stop',[(127,True,False),(128,False,False),(128,True,True),(128,True,False)])
def test_install_only_after_normal_terminal_and_all_eight_cards_release(tmp_path,generation,released,atomic_stop):
    import json
    from scripts.replay_batch_recipe import install,patched_replay
    path=tmp_path/'scripts/return_error_replay.py';path.parent.mkdir()
    before=Path('scripts/return_error_replay.py').read_text();path.write_text(before)
    parent=tmp_path/'runs/prior/formal/arm/gen000128';parent.mkdir(parents=True)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps(dict(generation=generation,passed=True,atomic_stop=atomic_stop)))
    (parent.parents[2]/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(released=released,compute_clients=0 if released else 1,devices=list(range(8)))))
    if generation!=128 or not released or atomic_stop:
        with pytest.raises(AssertionError):install(tmp_path,parent)
        assert path.read_text()==before
    else:
        receipt=install(tmp_path,parent)
        assert path.read_text()==patched_replay(before)
        assert receipt['current_slots']==receipt['history_slots']==64
        install(tmp_path,parent);assert path.read_text()==patched_replay(before)


def test_doubled_batch_does_not_double_mean_loss_or_actor_gradient():
    from scripts.return_dagger_weight import weighted_joint_loss
    move=jnp.array([[1.,0.,-1.,2.,0.],[0.,2.,1.,0.,-1.]])
    ability=jnp.array([[1.,0.,-1.],[0.,1.,0.]])
    mm=jnp.ones((2,5),bool);am=jnp.ones((2,3),bool)
    joint=jnp.zeros((2,5,3),bool).at[0,0,0].set(True).at[1,1,0].set(True)
    valid=jnp.ones(2,bool);returning=jnp.array([True,False])
    def loss(m,a,mm,am,j,v,r):return weighted_joint_loss(m,a,mm,am,j,v,r,2.,.05)[0]
    args=(move,ability,mm,am,joint,valid,returning)
    doubled=tuple(jnp.concatenate([x,x]) for x in args)
    np.testing.assert_allclose(loss(*args),loss(*doubled),rtol=1e-6)
    grad=jax.grad(loss)(*args);dgrad=jax.grad(loss)(*doubled)
    np.testing.assert_allclose(grad,dgrad[:2]+dgrad[2:],rtol=1e-6)


def test_terminal_relay_preserves_nested_proposal_and_explicit_batch_metadata(tmp_path):
    import inspect
    from scripts.je10_supervise import choose_recipe,predeclare_continuity,supervise
    plan=dict(combat_fraction=.35,bridge_fraction=.2,return_dagger_weight=True)
    proposal=dict(recipe='return_dagger_weight',replay_batch=True,
        reason='Actual bounded return replay coverage is limited',changed_factor='Native replay half batch 32 to 64')
    assert choose_recipe(plan,[],proposal)['replay_batch']
    assert choose_recipe(dict(plan,replay_batch=True),[],None)['replay_batch']
    source=inspect.getsource(supervise)
    assert "decision.get('proposal',{}).get('replay_batch')" in source
    assert source.index('replay_batch=install_replay_batch(')<source.index('c.prepare(')
    assert "plan['replay_batch']=replay_batch" in source
    assert "declaration['replay_batch']=replay_batch" in source
    assert "error_replay['current_history_and_phase_slot_counts_unchanged']=False" in source
    assert "'scripts/replay_batch_recipe.py'" in source
    assert 'replay_batch' in inspect.getsource(predeclare_continuity)
