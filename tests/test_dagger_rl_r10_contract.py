"""R10 is a supervised-set experiment, not an action ban or optimizer sweep."""
import json
from pathlib import Path
from typing import NamedTuple
import jax
import jax.numpy as j
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def test_frozen_pair_keeps_optimizer_reward_opponent_and_bank():
    p=json.loads((ROOT/'runs/dagger_rl_20261024/r10_release_qualification/PROTOCOL.json').read_text())
    prior=json.loads((ROOT/'runs/dagger_rl_20261024/r9_step_scale/PROTOCOL.json').read_text())
    assert p['learning_rate_by_arm']==dict(control=7.5e-5,qualified_route=7.5e-5)
    assert p['dagger_coefficient_by_arm']==dict(control=.2,qualified_route=.2)
    for k in ['parent','parent_input_sha256','optimizer','train_mixture','bank_sha256','oracle',
              'aggregation','envs','steps','updates','checkpoints','gamma','gae_lambda',
              'clip','value','entropy','gradient_clip','epochs','minibatch','anchor_coefficient']:
        assert p[k]==prior[k],k
    assert p['total_environment_steps']==1572864 and p['total_optimizer_steps']==6144
    assert set(p['paired_seeds']).isdisjoint(prior['paired_seeds'])
    assert abs(p['evaluation']['natural_holdout_seed']-prior['evaluation']['natural_holdout_seed'])>=64


def test_actual_collector_only_changes_supervised_labels_and_passive_counts(monkeypatch):
    from scripts import dagger_rl_r10_train as m
    class State(NamedTuple):
        t: object
        alive: object
        trapped: object
        bun_carried: object
        held_item: object
        pos: object
    st=State(j.zeros(2,int),j.ones((2,2),bool),j.zeros((2,2),int),
             j.array([[1,-1],[-1,-1]]),j.array([[1,0],[2,0]]),j.ones((2,2,2)))
    ax={'terminal':j.array([False,False]),'provenance_source':j.array([-1,0]),
        'opponent_enabled':j.zeros(2,bool),'opponent_candidate':j.zeros(2,bool)}
    keys=jax.random.split(jax.random.PRNGKey(0),2)
    o=j.zeros((2,24,13,15));g=j.zeros((2,24));mm=j.ones((2,5),bool);bm=j.ones((2,3),bool)
    own=j.array([[2,2],[1,2]]);actions=j.stack([own,own],1)
    route=j.zeros((2,5,3),bool).at[:,2,:].set(True);item=j.zeros_like(route)
    monkeypatch.setattr(m.jax,'pmap',lambda f,**kwargs:f)
    monkeypatch.setattr(m.C.E,'choose',lambda p,s,a,k:(k,k,actions,(o,g,mm,bm,own,j.zeros(2),j.zeros(2))))
    monkeypatch.setattr(m.C.T,'route_teacher',lambda s,a:route)
    monkeypatch.setattr(m.C.O,'teacher',lambda s,a:(item,j.zeros((2,6))))
    names=['trap_hit','capture','death','steal','pickup_loose','native_transfer','own_bomb_defeat']
    info={n:j.zeros((2,2),bool) for n in names}
    monkeypatch.setattr(m.C.T,'transition',lambda s,a,ac,k,arm:(s,a,j.zeros((2,2)),j.zeros(2,bool),info,j.zeros((2,2,1))))
    monkeypatch.setattr(m.C.E,'prediction',lambda p,s:(None,None,j.zeros(2)))
    monkeypatch.setattr(m.C.T,'fresh',lambda k,bs,ba:(st,ax))
    monkeypatch.setattr(m.C.E.train,'net_forward',lambda *args:(j.zeros((2,5)),j.zeros((2,3)),None,None))
    monkeypatch.setattr(m.C,'combat_rows',lambda o,g:j.zeros(2,bool))
    control=m.collector(1,False)(None,st,ax,keys,None,None)
    treatment=m.collector(1,True)(None,st,ax,keys,None,None)
    for i,(a,b) in enumerate(zip(control[1],treatment[1])):
        if i!=8:np.testing.assert_array_equal(a,b)
    # PPO executed ability2 stays unchanged, as do the physical masks/reward/GAE.
    assert treatment[1][2][0,0,1]==2
    assert not treatment[1][8][0,0,2,2]
    assert treatment[1][8][0,1,2,2]
    np.testing.assert_array_equal(treatment[2][8],[[[1,1,0],[0,0,0]]])
