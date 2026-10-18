"""Control attribution must pay once at entry, never on sustained syrup."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts import je10_control_credit as credit


def event(players=2, **changes):
    e=dict(tick=jnp.array([1]), alive=jnp.ones((1,players),bool),
        next_alive=jnp.ones((1,players),bool), controlled=jnp.zeros((1,players),bool),
        next_controlled=jnp.zeros((1,players),bool), source=jnp.zeros((1,players,players),bool),
        trigger=jnp.zeros((1,players,players),bool), death=jnp.zeros((1,players),bool),
        terminal=jnp.zeros(1,bool), teams=jnp.arange(players)[None])
    e.update(changes);return e


def wrap(players=2, attacker=0, victim=1, **changes):
    return event(players,next_controlled=jnp.zeros((1,players),bool).at[0,victim].set(True),
        source=jnp.zeros((1,players,players),bool).at[0,victim,attacker].set(True),**changes)


def test_entry_is_primary_and_persistent_wrap_cannot_pay_again():
    s=credit.state(1); e=wrap();s,r,m=credit.transition(e,s)
    assert float(r[0,0])==4.; assert int(m['event_id'][0,1])==1
    again=dict(e,tick=jnp.array([2]),controlled=e['next_controlled'])
    for _ in range(5):
        s,r,m=credit.transition(again,s); assert float(r.sum())==0


def test_same_tick_replayed_edge_is_deduplicated_and_jittable():
    e=wrap();s,r,_=jax.jit(credit.transition)(e,credit.state(1))
    s,r,_=jax.jit(credit.transition)(e,s); assert float(r.sum())==0


def test_chain_uses_unique_causal_owner_not_trigger_owner():
    e=wrap();e['trigger']=e['source'][:, :, ::-1]
    s,r,m=credit.transition(e,credit.state(1));np.testing.assert_array_equal(r,[[4.,0.]])
    assert int(m['owner'][0,1])==0


@pytest.mark.parametrize('attacker,victim,teams',[(0,0,[0,1]),(0,1,[0,0]),(2,1,[0,1,1])])
def test_self_and_teammate_control_are_not_paid(attacker,victim,teams):
    n=len(teams);e=wrap(n,attacker,victim,teams=jnp.array([teams]))
    _,r,_=credit.transition(e,credit.state(1,n));assert float(r.sum())==0


def test_ambiguous_chain_attribution_is_rejected():
    e=wrap(3);e['source']=e['source'].at[0,1,2].set(True)
    _,r,_=credit.transition(e,credit.state(1,3));assert float(r.sum())==0


def test_death_gets_only_small_residual_once_and_preserves_original_owner():
    s,_,_=credit.transition(wrap(),credit.state(1))
    e=event(tick=jnp.array([2]),controlled=jnp.array([[False,True]]),
        next_alive=jnp.array([[True,False]]),death=jnp.array([[False,True]]),
        trigger=jnp.zeros((1,2,2),bool).at[0,1,0].set(True))
    s,r,m=credit.transition(e,s);assert float(r[0,0])==.25;assert float(r.sum())<4
    s,r,_=credit.transition(e,s);assert float(r.sum())==0


def test_death_fixture_or_unattributed_pop_does_not_invent_entry_credit():
    e=event(controlled=jnp.array([[False,True]]),next_alive=jnp.array([[True,False]]),
        death=jnp.array([[False,True]]))
    _,r,_=credit.transition(e,credit.state(1));assert float(r.sum())==0


def test_respawn_starts_new_life_but_does_not_refund_episode_cap():
    s,_,_=credit.transition(wrap(),credit.state(1))
    s,_,_=credit.transition(event(tick=jnp.array([2]),controlled=jnp.array([[False,True]]),
        death=jnp.array([[False,True]]),next_alive=jnp.array([[True,False]])),s)
    s,r,_=credit.transition(event(tick=jnp.array([3]),alive=jnp.array([[True,False]])),s)
    assert float(r.sum())==0;assert int(s['life'][0,1])==1
    s,r,m=credit.transition(wrap(tick=jnp.array([4])),s);assert float(r[0,0])==4
    assert int(m['event_id'][0,1])==2
    s,r,_=credit.transition(wrap(tick=jnp.array([5])),s);assert float(r.sum())==0


def test_death_and_terminal_invalidate_payments_reset_changes_episode_identity():
    e=wrap(death=jnp.array([[True,False]]),next_alive=jnp.array([[False,True]]))
    s,r,_=credit.transition(e,credit.state(1));assert float(r.sum())<=0
    s=credit.reset(s,jnp.array([True]));assert int(s['episode'][0])==1
    s,r,m=credit.transition(wrap(),s);assert float(r[0,0])==4
    assert int(m['episode'][0])==1


def test_rewrap_same_life_is_bounded_and_loss_claws_back_without_refund():
    s,r,_=credit.transition(wrap(),credit.state(1))
    s,r,_=credit.transition(event(tick=jnp.array([2]),death=jnp.array([[True,False]]),
        next_alive=jnp.array([[False,True]])),s)
    assert float(r[0,0])==-4
    assert float(s['gross'][0,0])==4
