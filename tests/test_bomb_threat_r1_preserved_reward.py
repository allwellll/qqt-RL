import jax
import jax.numpy as j
import numpy as np
from scripts import bomb_threat_r1_preserved_reward as R
from scripts.bomb_threat_iter2_recipe import settlement


def event(**changes):
    e=dict(active=j.array([True]),bad=j.array([False]),terminal=j.array([False]),placed=j.array([True]),safe=j.array([True]),threat=j.array([True]),enemy_cell=j.array([30]),old_cell=j.array([21]),old_id=j.array([7]),chain=j.array([True]),owned_removed=j.zeros((1,13,15),bool).at[0,4,4].set(True),improved=j.array([True]),settled=j.array([True]),old_ids=j.array([[7,9,10,-1]]),multibatch=j.array([True]),batch_id=j.array([100]),pickup_types=j.zeros((1,3),bool),danger_after=j.array([False]),phi_before=j.array([-.02]),phi_after=j.array([-.018]),phase_before=j.array([True]),phase_after=j.array([True]))
    e.update(changes);return e


def test_legacy_settlement_bitwise_unchanged_and_independent():
    s=R.state(1);e=event()
    expected=settlement(5.,.6)(e,s['legacy'])
    got=R.legacy_settle(e,s['legacy'])
    for x,y in zip(jax.tree.leaves(expected),jax.tree.leaves(got)):np.testing.assert_array_equal(x,y)
    np.testing.assert_allclose(got[1],[.35],atol=1e-7)
    a,_,_=R.settle(e,s)
    for x,y in zip(jax.tree.leaves(a['legacy']),jax.tree.leaves(s['legacy'])):np.testing.assert_array_equal(x,y)


def test_shared_new_positive_cap_includes_items_and_cannot_recharge():
    s=R.state(1)
    for i in range(30):
        s,r,m=R.settle(event(batch_id=j.array([100+i]),old_ids=j.array([[7,100+i,200+i,-1]]),phi_before=j.array([-.3]),phi_after=j.array([-.1])),s)
        assert float(s['new_reserved'][0])<=.1000001
    np.testing.assert_allclose(s['new_reserved'],.1,atol=1e-7)
    a,ip,m=R.bounded_item_step(s['items'],j.array([[2.,2.,1.3]]),j.array([[10.,8.,2.4]]),j.ones((1,3),bool),active=j.array([True]),self_death=j.array([False]),terminal=j.array([False]),control_lost=j.array([False]),budget=j.maximum(0.,.1-s['new_reserved']))
    assert float(m['candidate'][0])==0 and float(a['pending'].sum())==0
    np.testing.assert_array_equal(a['highwater'],j.ones((1,3)))


def test_selfharm_terminal_and_control_void_new_only():
    for key in ('bad','terminal'):
        s=R.state(1);s,_,_=R.settle(event(settled=j.array([False])),s)
        assert float(s['pending'][0])>0
        before=s['legacy'];spent=s['new_reserved']
        s,r,m=R.settle(event(**{key:j.array([True])}),s)
        assert float(r[0])==0 and float(s['pending'][0])==0
        np.testing.assert_array_equal(s['new_reserved'],spent)
        for x,y in zip(jax.tree.leaves(before),jax.tree.leaves(s['legacy'])):np.testing.assert_array_equal(x,y)


def test_item_thirtieth_tick_void_and_budget_reservation():
    s=R.state(1)['items'];a=j.array([[2.,2.,1.3]]);b=j.array([[10.,8.,2.4]])
    flags=dict(active=j.array([True]),self_death=j.array([False]),terminal=j.array([False]),control_lost=j.array([False]),budget=j.array([.04]))
    s,r,m=R.bounded_item_step(s,a,b,b>a,**flags)
    np.testing.assert_allclose(m['candidate'],.04,atol=1e-7)
    for _ in range(29):s,_,_=R.bounded_item_step(s,b,b,b>b,**flags)
    flags['self_death']=j.array([True]);s,r,m=R.bounded_item_step(s,b,a,a>b,**flags)
    assert float(r[0])==0 and float(s['pending'].sum())==0
    np.testing.assert_allclose(m['void'],.04,atol=1e-7)
    np.testing.assert_array_equal(s['highwater'],j.ones((1,3)))


def test_new_item_and_approach_share_one_positive_reservation_budget():
    s=R.state(1);initial=j.array([[2.,2.,1.3]]);full=j.array([[10.,8.,2.4]])
    flags=dict(active=j.array([True]),self_death=j.array([False]),terminal=j.array([False]),control_lost=j.array([False]))
    item,p,m=R.bounded_item_step(s['items'],initial,full,full>initial,budget=j.array([.1]),**flags)
    s=dict(s,items=item,new_reserved=m['candidate'])
    np.testing.assert_allclose(m['candidate'],.1,atol=1e-7)
    s,p,m=R.settle(event(),s)
    assert float(j.maximum(p,0.).sum())==0
    assert float(s['new_reserved'][0])<=.1000001
    # The old bundle remains eligible despite the addition budget being full.
    _,oldpay,_=R.legacy_settle(event(),s['legacy'])
    np.testing.assert_allclose(oldpay,[.35],atol=1e-7)
