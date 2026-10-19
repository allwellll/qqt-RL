"""Successor quota rebalance preserves fixed-return and native-label authority."""
from pathlib import Path
import json,pickle,types
import pytest
import jax,jax.numpy as jnp
import numpy as np


def implementation():
    from scripts.navigation_replay_recipe import patched
    module=types.ModuleType('isolated_navigation_replay_candidate')
    exec(compile(patched(Path('scripts/phase_balanced_joint_replay.py').read_text()),'isolated-candidate','exec'),module.__dict__)
    return module


def test_navigation_share_increases_without_erasing_return_or_opening():
    from test_phase_balanced_joint_replay import fixture
    d=fixture();m=implementation();got=jax.jit(lambda d,k:m.choose(d,k,32))(d,jax.random.PRNGKey(81))
    ordinary=got['valid']&~got['preparation']
    assert int(ordinary.sum())==16
    assert int((ordinary&(got['kind']==3)).sum())==12
    for k in [1,2]:assert int((ordinary&(got['kind']==k)).sum())==2
    assert int((got['preparation']&~got['opening']).sum())==15
    assert int(got['opening'].sum())==1
    assert bool(got['valid'].all())


def test_no_preparation_and_next_draw_after_restore_remain_exact():
    from test_phase_balanced_joint_replay import fixture
    from scripts.phase_balanced_joint_replay import choose
    d=fixture();d['preparation']=jnp.zeros(100,bool);d['opening']=jnp.zeros(100,bool)
    d['valid']=jnp.arange(100)%3!=0;k=jax.random.PRNGKey(82);m=implementation()
    a=choose(d,k,32);b=m.choose(pickle.loads(pickle.dumps(d)),k,32)
    for name in a:np.testing.assert_array_equal(a[name],b[name])


def test_absent_groups_single_opening_and_invalid_padding_preserve_authority():
    from test_phase_balanced_joint_replay import fixture
    m=implementation();d=fixture();d['valid']=jnp.arange(100)==99
    got=m.choose(d,jax.random.PRNGKey(83),32)
    np.testing.assert_array_equal(got['identity'],jnp.full(32,99))
    d['valid']=jnp.zeros(100,bool);got=m.choose(d,jax.random.PRNGKey(84),32)
    assert not bool(got['valid'].any())


def test_aggregate_keeps_fifo_updates_and_exact_restore_with_telemetry():
    from test_prestock_joint_replay import batch
    from scripts import prestock_joint_replay as uniform
    m=implementation();old=uniform.buffer(capacity=4,additions=2,obs_shape=(1,1,1),global_size=2)
    data=list(batch());meta=jnp.array([[[1,4,2,0],[0,0,0,0],[1,5,2,1],[0,0,0,0]]])
    data[-1]=jnp.concatenate([data[-1],meta],-1);data=tuple(data)
    for g in [1,2,3]:
        k=jax.random.PRNGKey(g);expected=uniform.aggregate(old,data,k,jnp.int32(g))
        got=jax.jit(m.aggregate)(old,data,k,jnp.int32(g))
        for name in expected[0]:np.testing.assert_array_equal(expected[0][name],got[0][name])
        replay=got[1]
        assert bool((replay['round'][:2]==g).all())
        if g>1:assert bool((replay['round'][2:]<g).all())
        again=m.aggregate(pickle.loads(pickle.dumps(old)),data,k,jnp.int32(g))
        for a,b in zip(jax.tree.leaves(got),jax.tree.leaves(again)):np.testing.assert_array_equal(a,b)
        old=got[0]


def root_fixture(tmp_path,generation=128,released=True,stop=False):
    root=tmp_path/'root';path=root/'scripts/phase_balanced_joint_replay.py';path.parent.mkdir(parents=True)
    path.write_text(Path('scripts/phase_balanced_joint_replay.py').read_text())
    parent=root/'runs/prior/formal/arm/gen000128';parent.mkdir(parents=True)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps(dict(generation=generation,passed=True,atomic_stop=stop)))
    (parent.parents[2]/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(released=released,compute_clients=0 if released else 1)))
    return root,parent,path


@pytest.mark.parametrize('generation,released,stop',[(127,True,False),(128,False,False),(128,True,True)])
def test_installer_rejects_before_terminal_release_or_interrupted(tmp_path,generation,released,stop):
    from scripts.navigation_replay_recipe import install
    root,parent,path=root_fixture(tmp_path,generation,released,stop);old=path.read_text()
    with pytest.raises(AssertionError):install(root,parent)
    assert path.read_text()==old


def test_terminal_installer_idempotent_and_does_not_change_other_factors(tmp_path):
    from scripts.navigation_replay_recipe import install,patched
    root,parent,path=root_fixture(tmp_path);d=install(root,parent);first=path.read_text()
    assert d['per_half']==dict(ordinary=16,returning=12,drop=2,stock_ready=2,navigation=15,opening=1)
    assert not d['reward_coefficient_changed'] and not d['query_budget_changed'] and not d['actor_observation_changed']
    assert d['current_history_half_preserved'] and d['total_replay_per_device']==64
    install(root,parent);assert path.read_text()==first and patched(first)==first
