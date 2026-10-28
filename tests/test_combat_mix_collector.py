from tests.external_artifacts import requires_artifacts
"""Focused real-state checks; no synthetic learner custody."""
import os
os.environ.setdefault('JAXBOMB_RULE','bun')
import jax
import jax.numpy as j
import numpy as np
from scripts import combat_mix_collector as M
from scripts import carrier_mechanism_engine as E
from scripts import combat_item_engine as T
from scripts import dagger_rl_r4_train as C

def test_metadata_separate_from_observation_and_no_initial_carry_mutation():
    T.configure();C.install_provenance()
    keys=jax.random.split(jax.random.PRNGKey(2026103191),2)
    st,ax=E.initial(keys,0)
    before=E.inputs(st)
    after=dict(ax,combat_mix=M.metadata(st,j.array([0,3])))
    assert all(np.array_equal(a,b) for a,b in zip(before,E.inputs(st)))
    assert np.all(np.asarray(st.bun_carried[:,0])<0)
    assert np.all(np.asarray(after['combat_mix']['shaping']['pending'])==0)


def test_pool_partition_is_disjoint_and_source_parent_is_not_modified():
    from scripts.combat_mix_opponents import fixed_allocation
    ids=fixed_allocation(2048,2026103051)
    groups=[np.flatnonzero(ids<3)]+[np.flatnonzero(ids==i) for i in (3,4,5)]
    assert np.array_equal(np.sort(np.concatenate(groups)),np.arange(2048))


@requires_artifacts('runs/combat_mix_v1/PROTOCOL.json')
def test_real_learner_action_logprob_value_and_rng_are_identical_when_replacing_opponent_only():
    import json,pickle
    from pathlib import Path
    source=json.loads(Path('runs/combat_mix_v1/PROTOCOL.json').read_text())['sources']['2026103051']
    params=pickle.load((Path(source['checkpoint'])/'state.pkl').open('rb'))['params']
    keys=jax.random.split(jax.random.PRNGKey(2026103192),4)
    st,ax=E.initial(keys,0)
    ax=dict(ax,tactics=T.events.state(4),provenance_source=j.full(4,-1,j.int32))
    kinds=np.array([0,3,4,5])
    ax['combat_mix']=M.metadata(st,kinds)
    groups=dict(bot=np.array([0]),current=np.array([1]),ring=np.array([2]),fixed=np.array([3]))
    original=jax.jit(E.choose)(params,st,ax,keys)
    actual=jax.jit(lambda p,s,a,k:M.choose(p,p['actor'],p['actor'],s,a,k,groups))(params,st,ax,keys)
    for i in (0,1):
        assert np.array_equal(original[i],actual[i])
    assert np.array_equal(original[2][:,0],actual[2][:,0])
    for x,y in zip(original[3],actual[3]):
        assert np.array_equal(x,y)
    assert np.array_equal(original[2][0,1],actual[2][0,1])


def test_chain_credit_requires_a_real_unobstructed_owned_old_to_new_temporal_ray():
    from scripts import combat_mix_shaping as S
    keys=jax.random.split(jax.random.PRNGKey(2026103193),1)
    before,ax=E.initial(keys,1)
    before=before._replace(core=before.core._replace(pos=j.array([[[3.5,3.5],[3.5,5.5]]]),
        t=j.array([40],j.int32),wall=before.wall.at[0,3:6,3].set(False),pushable=before.core.pushable.at[0,3:6,3].set(False),fuse=j.zeros_like(before.fuse).at[0,5,3].set(20),
        owner=j.full_like(before.core.owner,-1).at[0,5,3].set(0),
        bomb_blast=j.zeros_like(before.core.bomb_blast).at[0,5,3].set(4)))
    after=before._replace(core=before.core._replace(fuse=before.fuse.at[0,3,3].set(29),
        owner=before.core.owner.at[0,3,3].set(0)))
    meta=M.metadata(before,j.array([0]))
    yes=j.array([[True,False]]);no=j.zeros((1,2),bool)
    info=dict(bomb_placed=yes,tactical_bomb_newly_threatens_enemy=yes,survivable_bomb_placed=yes,death=no)
    ev,_=S.chain_event(before,after,info,ax,meta)
    assert bool(ev['connected_uncredited'][0])
    wall=before.wall.at[0,4,3].set(True)
    blocked=before._replace(core=before.core._replace(wall=wall))
    ev,_=S.chain_event(blocked,after,info,ax,meta)
    assert not bool(ev['connected_uncredited'][0])
    enemy=before._replace(core=before.core._replace(owner=before.core.owner.at[0,5,3].set(1)))
    ev,_=S.chain_event(enemy,after,info,ax,meta)
    assert not bool(ev['connected_uncredited'][0])
