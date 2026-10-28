"""Inactive successor candidate: phase-scoped parent retention semantics."""
import jax
import jax.numpy as j
import numpy as np
from scripts.dagger_rl_anchor import combat_rows, reference_kl


def test_anchor_excludes_real_cargo_windows_loose_and_inactive():
    obs=j.zeros((8,24,13,15));g=j.zeros((8,24)).at[:,9:11].set(1)
    g=g.at[1,13:15].set(1).at[2,10].set(0).at[3,10].set(1.5)
    obs=obs.at[4,17,6,6].set(1)
    g=g.at[5,9].set(0).at[6,9].set(1.5)
    # Ordinary slow is not a native control window.
    g=g.at[7,23].set(1)
    np.testing.assert_array_equal(combat_rows(obs,g),[True,False,False,False,False,False,False,True])


def test_anchor_has_no_offphase_or_teacher_gradient_and_respects_masks():
    move=j.array([[1.,2.,3.,4.,5.],[2.,1.,0.,3.,4.]])
    ability=j.array([[0.,1.,2.],[1.,0.,2.]])
    pm=j.zeros_like(move);pa=j.zeros_like(ability)
    mm=j.ones_like(move,dtype=bool).at[:,4].set(False)
    bm=j.ones_like(ability,dtype=bool).at[:,2].set(False)
    gate=j.array([True,False])
    f=lambda m,a,t,u:reference_kl(m,a,t,u,mm,bm,gate)
    grad=jax.grad(f,(0,1,2,3))(move,ability,pm,pa)
    assert np.linalg.norm(grad[0][0])>0 and np.linalg.norm(grad[1][0])>0
    for x in grad[:2]:np.testing.assert_array_equal(x[1],0)
    np.testing.assert_array_equal(grad[0][:,4],0)
    np.testing.assert_array_equal(grad[1][:,2],0)
    for x in grad[2:]:np.testing.assert_array_equal(x,0)
    zero=reference_kl(move,ability,pm,pa,mm,bm,j.zeros(2,bool))
    assert float(zero)==0
    for x in jax.grad(lambda m,a:reference_kl(m,a,pm,pa,mm,bm,j.zeros(2,bool)),(0,1))(move,ability):np.testing.assert_array_equal(x,0)


def test_zero_coefficient_is_exact_old_objective():
    m=j.arange(10,dtype=j.float32).reshape((2,5));a=j.arange(6,dtype=j.float32).reshape((2,3))
    old=lambda x,y:j.sum(x*x)+j.sum(y*y)
    new=lambda x,y:old(x,y)+0.*reference_kl(x,y,j.zeros_like(x),j.zeros_like(y),j.ones_like(x,bool),j.ones_like(y,bool),j.ones(2,bool))
    np.testing.assert_array_equal(new(m,a),old(m,a))
    for x,y in zip(jax.grad(new,(0,1))(m,a),jax.grad(old,(0,1))(m,a)):np.testing.assert_array_equal(x,y)
