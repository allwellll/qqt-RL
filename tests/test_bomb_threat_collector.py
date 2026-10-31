import jax
import jax.numpy as j
import numpy as np
from scripts import bomb_threat_collector as M
from scripts import carrier_mechanism_engine as E

def test_metadata_reward_is_checkpointed_but_never_actor_input():
    M.T.configure()
    keys=jax.random.split(jax.random.PRNGKey(20261029),1)
    st,ax=E.initial(keys,1);a=E.inputs(st)
    meta=M.metadata(st,j.array([0]));assert 'bomb_threat' in meta
    np.testing.assert_array_equal(a[0],E.inputs(st)[0]);assert int(st.bun_carried[0,0])==-1

def test_reward_added_to_actual_transition_and_parts(monkeypatch):
    # A finite arbitrary reward payload tests actual collector wiring, not ability.
    def base(st,ax,actions,keys,arm):return st,ax,j.zeros((1,2)),j.array([False]),{},j.zeros((1,2,3))
    def extra(st,after,info,base,meta):return meta['bomb_threat'],j.array([.02]),{},meta['birth']
    monkeypatch.setattr(M.T,'transition',base);monkeypatch.setattr(M.threat,'transition',extra)
    class S:t=j.array([0])
    meta=dict(birth=j.zeros((1,13,15)),bomb_threat={},shaping={})
    _,ax,r,_,_,parts,_=M.transition(S(),dict(combat_mix=meta),None,None,.3)
    np.testing.assert_allclose(r,[[.02,0]]);np.testing.assert_allclose(parts[:,:,-1],r)
