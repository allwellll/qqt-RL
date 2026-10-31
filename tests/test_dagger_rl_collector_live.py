"""Exercise collector metadata using the actual BunState API shape (no done)."""
from typing import NamedTuple
import jax
import jax.numpy as j
import numpy as np


def test_collector_live_uses_persisted_terminal_aux_not_missing_state_field(monkeypatch):
    from scripts import dagger_rl_r2_train as train

    class State(NamedTuple):
        t: object
        alive: object
        trapped: object
        bun_carried: object
        held_item: object

    st=State(j.zeros(2,int),j.ones((2,2),bool),j.zeros((2,2),int),-j.ones((2,2),int),j.zeros((2,2),int))
    ax={'terminal':j.array([False,True]),'provenance_source':j.array([-1,0])}
    keys=jax.random.split(jax.random.PRNGKey(0),2)
    o=j.zeros((2,24,13,15));g=j.zeros((2,24));mm=j.ones((2,5),bool);bm=j.ones((2,3),bool)
    own=j.zeros((2,2),int);actions=j.zeros((2,2,2),int);mask=j.zeros((2,5,3),bool)
    monkeypatch.setattr(train.jax,'pmap',lambda function,**kwargs:function)
    monkeypatch.setattr(train.E,'choose',lambda p,s,a,k:(k,k,actions,(o,g,mm,bm,own,j.zeros(2),j.zeros(2))))
    monkeypatch.setattr(train.T,'route_teacher',lambda s,a:mask)
    monkeypatch.setattr(train.O,'teacher',lambda s,a:(mask,j.zeros((2,5))))
    names=['trap_hit','capture','death','steal','pickup_loose','native_transfer','own_bomb_defeat']
    info={name:j.zeros((2,2),bool) for name in names}
    monkeypatch.setattr(train.T,'transition',lambda s,a,act,k,arm:(s,a,j.zeros((2,2)),j.zeros(2,bool),info,j.zeros((2,2,1))))
    monkeypatch.setattr(train.E,'prediction',lambda p,s:(None,None,j.zeros(2)))
    monkeypatch.setattr(train.T,'fresh',lambda k,bs,ba:(st,ax))
    monkeypatch.setattr(train.E.train,'net_forward',lambda p,arch,o,g:(j.zeros((2,5)),j.zeros((2,3)),None,None))
    _,batch,_=train.collector(1)(None,st,ax,keys,None,None)
    np.testing.assert_array_equal(batch[10],[[True,False]])
