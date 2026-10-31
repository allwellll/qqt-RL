import pickle
import numpy as np
import jax
import jax.numpy as j
from scripts import bomb_threat_r1_item_progress as P


def evolve(s,before,after,**kw):
    before=j.array([before],j.float32);after=j.array([after],j.float32)
    flags=dict(active=j.array([True]),self_death=j.array([False]),terminal=j.array([False]),control_lost=j.array([False]))
    flags.update({k:j.array([v]) for k,v in kw.items()})
    return jax.jit(P.step)(s,before,after,after>before,**flags)


def mature(s,stats,n=30):
    total=0.
    for _ in range(n):
        s,r,m=evolve(s,stats,stats);total+=float(r[0])
    return s,total


def test_three_full_attributes_exact_024_independent_of_attack_cap():
    s=P.state(j.array([P.INITIAL]));s,r,m=evolve(s,P.INITIAL,P.MAXIMUM)
    assert float(r[0])==0
    s,total=mature(s,P.MAXIMUM)
    np.testing.assert_allclose(total,.24,atol=1e-7)
    assert float(s['pending'].sum())==0


def test_super_actual_clipping_and_seven_speed_steps():
    s=P.state(j.array([P.INITIAL]));s,_,_=evolve(s,P.INITIAL,[7,7,1.3+5*1.1/7])
    s,total=mature(s,[7,7,1.3+5*1.1/7])
    np.testing.assert_allclose(total,.08*(5/8+5/6+5/7),atol=1e-7)
    s,_,_=evolve(s,[7,7,1.3+5*1.1/7],P.MAXIMUM);s,t=mature(s,P.MAXIMUM)
    np.testing.assert_allclose(total+t,.24,atol=1e-7)


def test_death_repickup_overflow_and_scene_changes_cannot_pay():
    s=P.state(j.array([P.INITIAL]));s,_,_=evolve(s,P.INITIAL,[4,4,1.3])
    s,total=mature(s,[4,4,1.3]);np.testing.assert_allclose(total,.08*(2/8+2/6),atol=1e-7)
    s,_,_=evolve(s,[4,4,1.3],P.INITIAL)
    s,_,m=evolve(s,P.INITIAL,[4,4,1.3]);assert float(m['candidate'][0])==0
    # False native growth flag excludes an arbitrary changed snapshot.
    _,r,m=P.step(s,j.array([[4,4,1.3]]),j.array([P.MAXIMUM]),j.zeros((1,3),bool),active=j.array([True]),self_death=j.array([False]),terminal=j.array([False]),control_lost=j.array([False]))
    assert float(m['candidate'][0])==0
    s=P.state(j.array([P.MAXIMUM]));s,_,m=evolve(s,P.MAXIMUM,P.MAXIMUM)
    assert float(m['candidate'][0])==0


def test_exact30_selfdeath_voids_and_highwater_does_not_rewind():
    s=P.state(j.array([P.INITIAL]));s,_,_=evolve(s,P.INITIAL,[3,2,1.3]);s,total=mature(s,[3,2,1.3],29)
    assert total==0
    s,r,m=evolve(s,[3,2,1.3],P.INITIAL,self_death=True)
    assert float(r[0])==0 and float(m['void'][0])>0
    s,_,m=evolve(s,P.INITIAL,[3,2,1.3]);assert float(m['candidate'][0])==0


def test_terminal_and_control_tail_is_explicitly_void_no_bootstrap_reward():
    for invalid in ('terminal','control_lost'):
        s=P.state(j.array([P.INITIAL]));s,_,_=evolve(s,P.INITIAL,[3,2,1.3])
        s,r,m=evolve(s,[3,2,1.3],[3,2,1.3],**{invalid:True})
        assert float(r[0])==0 and float(s['pending'].sum())==0
        assert float(m['void'][0])>0


def test_real_bank_baseline_no_retroactive_and_restore_exact():
    s=P.state(j.array([[7,4,2.0]]));s,_,m=evolve(s,[7,4,2.0],[7,4,2.0])
    assert float(m['candidate'][0])==0
    s,_,_=evolve(s,[7,4,2.0],[8,4,2.0]);copy=pickle.loads(pickle.dumps(s))
    a=evolve(s,[8,4,2.0],[8,4,2.0]);b=evolve(copy,[8,4,2.0],[8,4,2.0])
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)


def test_independent_weight_halving_only_changes_future_events():
    s=P.state(j.array([P.INITIAL]));s=dict(s,scale=j.array([.5]))
    s,_,_=evolve(s,P.INITIAL,P.MAXIMUM);s,total=mature(s,P.MAXIMUM)
    np.testing.assert_allclose(total,.12,atol=1e-7)


def test_bank_historical_highwater_survives_past_death_not_just_current_level():
    # Restored current bombs3 but actual source episode previously reached8.
    # Recollecting to8 must remain zero even though current snapshot is lower.
    s=P.state(j.array([[3.,2.,1.3]]));s=dict(s,highwater=P.fraction(j.array([[8.,2.,1.3]])))
    s,_,m=evolve(s,[3,2,1.3],[8,2,1.3]);assert float(m['candidate'][0])==0
    s,_,m=evolve(s,[8,2,1.3],[9,2,1.3])
    np.testing.assert_allclose(m['candidate'],.08/8,atol=1e-7)


def test_zero_weight_scale_has_exact_zero_payout_and_keeps_truthful_history():
    s=P.state(j.array([P.INITIAL]));s=dict(s,scale=j.zeros(1))
    s,r,m=evolve(s,P.INITIAL,P.MAXIMUM);s,total=mature(s,P.MAXIMUM)
    assert total==0 and float(s['pending'].sum())==0
    np.testing.assert_array_equal(s['highwater'],j.ones((1,3)))
