"""Schedule and inactive Adam semantics, not capability admission."""
import jax
import jax.numpy as j
import numpy as np
from scripts import dagger_rl_train as T


def test_aggregated_labels_stay_out_of_ppo_objective(monkeypatch):
    p=dict(actor=j.zeros(8),critic=j.zeros(1))
    def forward(p,arch,o,g):return j.tile(p['actor'][:5],(len(o),1)),j.tile(p['actor'][5:],(len(o),1)),j.zeros(len(o)),j.tile(p['critic'],(len(o),1))
    monkeypatch.setattr(T.E.train,'net_forward',forward)
    monkeypatch.setattr(T.E.train,'hl_gauss_value_loss',lambda v,r:j.mean((v[:,0]-r)**2))
    mask=j.zeros((2,5,3),bool).at[:,2,2].set(True)
    b=(j.zeros((2,1)),j.zeros((2,1)),j.zeros((2,2),int),j.zeros(2),j.ones(2),j.ones(2),j.ones((2,5),bool),j.ones((2,3),bool),mask,j.ones(2,bool))
    replay=(b[0],b[1],b[6],b[7],mask,b[-1])
    changed=replay[:4]+(j.zeros_like(mask),replay[-1])
    a=T.objective(p,b,replay,'ppo')[0];c=T.objective(p,b,changed,'ppo')[0]
    np.testing.assert_array_equal(a,c)
    ga=jax.grad(lambda q:T.objective(q,b,replay,'dagger')[0])(p)
    np.testing.assert_array_equal(ga['critic'],j.zeros(1))
    assert ga['actor'][7]<0


def test_actual_fifo_is_bounded_and_contains_student_ability_labels():
    # Mimics genuine collector tensor axes only; no environment/cargo fixture.
    n=(8,2,2);z=j.zeros(n);o=j.zeros(n+(1,));mask=j.zeros(n+(5,3),bool).at[...,2,2].set(True)
    b=(o,o,j.zeros(n+(2,),int),z,z,z,j.ones(n+(5,),bool),j.ones(n+(3,),bool),mask,j.ones(n,bool))
    hist,rb,stats=T.aggregate(b,None,1,1)
    assert stats['new_item_rows']==32 and rb[-1].shape==(8,1024)
    np.testing.assert_array_equal(rb[-2][...,2,2],True)
    assert len(hist[0])==32
