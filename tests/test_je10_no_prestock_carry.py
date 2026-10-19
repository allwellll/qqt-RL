import os

os.environ.setdefault("JAXBOMB_RULE", "bun")

import jax
import jax.numpy as jnp
import pytest
import numpy as np

from jax_bomb import je07_bun_env as env
from scripts import je09_bun_scenes as scenes


def test_route_scene_reset_never_grants_learner_enemy_custody():
    env.prepare()
    sample=jax.jit(lambda key,course: scenes.init_batch(key,1024,course))
    rows=jnp.arange(1024)
    # 16,384 fixed-seed starts, both seats, including stale forced courses.
    for seed in range(16):
        course=(jax.nn.one_hot((2,3,4,6)[seed%4],10) if seed<8 else jnp.ones(10)/10)
        state=sample(jax.random.PRNGKey(2026101800+seed),course)
        assert bool(jnp.all(state.bun_carried[rows,rows%2]<0))
        assert not bool(jnp.isin(state.scene,jnp.array([2,3,4,6])).any())


def test_every_scene_weight_entrypoint_excludes_invalid_carrier_scenes():
    from scripts import je05_bun_scenes,je07_bun_scenes,je09_bun_ops,je09_stage_h6_ops
    for module in [je05_bun_scenes,je07_bun_scenes,scenes]:
        w=module.curriculum_weights(jnp.zeros((len(module.NAMES),2)))
        np.testing.assert_array_equal(np.asarray(w)[[2,3,4,6]],0)
        assert np.isclose(float(w.sum()),1)
    for module in [je09_bun_ops,je09_stage_h6_ops]:
        w=module.curriculum_weights(jnp.zeros((10,len(module.FUNNEL))),.35,.2)
        np.testing.assert_array_equal(np.asarray(w)[[2,3,4,6]],0)
        assert np.isclose(float(w.sum()),1)


def test_dagger_does_not_grant_custody_for_legacy_scene_id():
    from scripts import je10_ops,je10_route_oracle
    state=scenes.init_batch(jax.random.PRNGKey(41),2,jax.nn.one_hot(8,10))
    state=state._replace(scene=jnp.full(2,2,jnp.int8),
        bun_carried=state.bun_carried.at[:,0].set(1))
    aux=je10_ops.reward_state(2)
    result=je10_route_oracle.labels(state,aux,jnp.ones((2,5),bool),
        jnp.zeros(2,jnp.int32),jnp.zeros((2,13,15)))
    assert not bool((result[:,7]>0).any())


def test_real_enemy_cage_steal_is_the_only_stock_custody_transition():
    env.prepare()
    state = scenes.init_batch(jax.random.PRNGKey(701), 2,
                              jax.nn.one_hot(8, len(scenes.NAMES)))
    base = env._BUN_BASES[0]
    pos = state.pos.at[:, 1].set(base.astype(jnp.float32) + jnp.array([1.5, 3.05]))
    state = state._replace(
        core=state.core._replace(pos=pos, brick=jnp.zeros_like(state.brick),
                                 wall=jnp.zeros_like(state.wall)),
        scene=jnp.full(2, 8, jnp.int8),
    )
    rows = jnp.arange(2)
    assert bool(jnp.all(state.bun_carried[rows, rows % 2] < 0))
    joint = jnp.zeros((2, 2, 2), jnp.int32).at[:, 0, 0].set(4).at[:, 1, 0].set(2)
    after, _, info = jax.vmap(
        lambda s, a, k: env.step(s, a, k, auto_reset=False, return_info=True)
    )(state, joint, jax.random.split(jax.random.PRNGKey(702), 2))
    assert bool(jnp.all(info["steal"][:, 1]))
    assert bool(jnp.all(after.bun_carried[:, 1] == 0))


def test_fixed_return_replay_requires_native_event_proof(tmp_path):
    import hashlib
    import json
    import pickle
    from scripts.je10_causal_replay import load_verified
    replay=tmp_path/'return_replay.pkl';receipt=tmp_path/'RETURN_REPLAY.json'
    payload={group:tuple(np.zeros((n,2)) for _ in range(4)) for group,n in [('short',3),('long',2)]}
    replay.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError,match='proof'):
        load_verified(replay,receipt)
    proof=dict(passed=True,sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),
        real_native_drop_pickup=True,action_takeover=False,future_information=False,
        short=dict(rows=3),long=dict(rows=2))
    receipt.write_text(json.dumps(proof));assert len(load_verified(replay,receipt)['short'][0])==3
    for field,value in [('real_native_drop_pickup',False),('sha256','wrong'),('short',dict(rows=9))]:
        receipt.write_text(json.dumps(dict(proof,**{field:value})))
        with pytest.raises(ValueError,match='proof'):
            load_verified(replay,receipt)


@pytest.mark.parametrize('module_name',['bun_env','je05_bun_env','je07_bun_env'])
def test_legacy_lesson_entrypoints_never_initialize_custody(module_name):
    import importlib
    module=importlib.import_module('jax_bomb.'+module_name);module.prepare()
    for lesson in [1,11]:
        module.configure_training(('carry_home' if lesson==1 else 'carry_return')+'=1',1,retention_replay=0)
        sample=jax.jit(lambda key:module.init_batch(key,128))
        state=sample(jax.random.PRNGKey(2026101800+lesson))
        assert bool((state.bun_carried<0).all())
        assert not bool(jnp.isin(state.lesson,jnp.array([1,11])).any())


@pytest.mark.parametrize('contact',[False,True])
def test_retained_native_drop_pickup_and_contact_transfer_create_real_custody(contact):
    env.prepare()
    state=jax.tree.map(lambda x:x[0],scenes.init_batch(jax.random.PRNGKey(901),2,jax.nn.one_hot(9,10)))
    positions=jnp.array([[5.5,4.5],[5.5,5.3 if contact else 6.5]])
    state=state._replace(core=state.core._replace(pos=positions,brick=jnp.zeros_like(state.brick),
        wall=jnp.zeros_like(state.wall),pushable=jnp.zeros_like(state.core.pushable)),
        trapped=state.trapped.at[1].set(10 if contact else 1))
    assert int(state.bun_carried[0])==-1
    tick=jax.jit(lambda s,move:env.step(s,jnp.array([[move,0],[4,0]]),
        jax.random.PRNGKey(902),auto_reset=False,return_info=True))
    state,_,info=tick(state,4)
    if contact:
        assert bool(info['native_transfer'][0]) and int(state.bun_carried[0])==1
    else:
        assert bool(info['drop'][1]) and int(state.bun_carried[0])==-1
        picked=False
        for _ in range(24):
            state,_,info=tick(state,3)
            if bool(info['pickup_loose'][0]):
                picked=True;break
        assert picked and int(state.bun_carried[0])==1
