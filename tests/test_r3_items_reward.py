import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import jax.numpy as j
import numpy as np
from scripts import r3_items_reward as R


def simulate(candidates,bad_at=(),terminal_at=()):
    s=R.state(1);paid=np.zeros(8)
    for t,c in enumerate(candidates):
        s,p,_=R.settle(s,j.asarray([c]),bad=j.array([t in bad_at]),terminal=j.array([t in terminal_at]),active=j.array([True]))
        paid+=np.asarray(p[0])
    return s,paid


def test_glue_effect_only_and_outcome_precedence():
    _,pay=simulate(np.zeros((100,8)));assert pay.sum()==0
    np.testing.assert_allclose(R.glue_reward(j.array([False,False,True]),j.array([False,True,True])),[.01,.04,.06])


def test_single_factual_kill_cannot_upgrade_multiple_glue_contacts():
    age=j.array([[45,20,5,-1,-1,-1]])
    z=j.zeros((1,6),bool)
    kill,pickup=R.glue_outcomes(age,z,z,z,z,j.array([True]),j.array([True]))
    np.testing.assert_array_equal(kill,[[False,False,True,False,False,False]])
    np.testing.assert_array_equal(pickup,kill)


def test_banana_carry_speed_only():
    np.testing.assert_allclose(R.banana_reward(j.array([False,True,True]),j.array([20.,20.,20.]),j.array([False,False,True])),[0.,.04,-.02])


def test_delay_30_and_blast_at_maturity_voids():
    c=np.zeros((31,8));c[0,R.BANANA]=.04
    _,paid=simulate(c[:30]);assert paid.sum()==0
    _,paid=simulate(c);np.testing.assert_allclose(paid[R.BANANA],.04)
    s,paid=simulate(c,bad_at=(30,));assert paid.sum()==0;assert float(s['void'][0,R.BANANA])>.039


def test_cross_batch_real_chain_and_voids():
    np.testing.assert_allclose(R.chain_reward(j.array([2,2,2,2,2]),same_batch=j.array([False,True,False,False,False]),
        bad=j.array([False,False,True,False,False]),empty=j.array([False,False,False,True,False]),home_stack=j.array([False,False,False,False,True])),[.03,0,0,0,0])


def test_road_highwater_repeat_opponent_and_partial_cause():
    pay,best,improved=R.road_credit(j.array([20,20,20,20]),j.array([15,15,15,15]),j.array([20,20,17,20]),
        j.array([True,False,True,True]),j.array([True,True,True,False]))
    np.testing.assert_allclose(pay,[.15,0,.06,0]);np.testing.assert_array_equal(best,[15]*4)
    pay,_,_=R.road_credit(best,j.array([15]*4),j.array([20]*4),j.array([True]*4),j.array([True]*4))
    np.testing.assert_array_equal(pay,0)


def test_stock_three_bombs_without_events_pays_zero():
    _,pay=simulate(np.zeros((2400,8)));assert pay.sum()==0


def test_joint_caps_and_negative_penalty_no_refill():
    s=R.state(1);s,p=R.capped_payment(s,j.ones((1,8))*10)
    assert float(p[0,:5].sum())<=.600001 and float(p[0,5:].sum())<=.450001
    assert np.all(np.asarray(p[0])<=np.asarray(R.CAPS)+1e-6)
    s,_=R.capped_payment(s,-j.ones((1,8)))
    s,p=R.capped_payment(s,j.ones((1,8))*10);assert float(p.sum())<1e-6


def test_inherited_attack_budget_included():
    _,p=R.capped_payment(R.state(1),j.ones((1,8))*10,attack_gross=j.array([.43]))
    np.testing.assert_allclose(p[0,5:].sum(),.02,atol=1e-7)


def test_existing_event_settles_in_tail_no_new_post_terminal_event():
    c=np.zeros((60,8));c[0,R.ROAD]=.2
    s=R.state(1);paid=0.
    for t,row in enumerate(c):
        s,p,_=R.settle(s,j.asarray([row]),bad=j.array([False]),terminal=j.array([t>=5]),active=j.array([t<5]))
        paid+=float(p.sum())
    np.testing.assert_allclose(paid,.2)
    phi=np.array([0.,-.0075,-.005,-.0025,0.]);assert abs(np.diff(phi).sum())<1e-12
