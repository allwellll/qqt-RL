from types import SimpleNamespace
import jax.numpy as jnp
from scripts import je10_control_credit as credit


def test_native_contact_pop_has_explicit_active_source():
    before=SimpleNamespace(alive=jnp.array([[True,True]]),trapped=jnp.array([[0,9]]),t=jnp.array([4]))
    after=SimpleNamespace(alive=jnp.array([[True,False]]),trapped=jnp.array([[0,0]]),trapped_causal=jnp.array([[[False,False],[True,False]]]),trapped_source=jnp.array([[[False,False],[True,False]]]))
    info=dict(trigger_damage_source=jnp.zeros((1,2,2),bool),death=jnp.array([[False,True]]),native_trap_popper=jnp.array([[-1,0]]))
    events=credit.native_events(before,after,info,jnp.array([False]))
    assert bool(events['trigger'][0,1,0]), 'native popper must reach the residual ledger independently of explosion source'


def test_controlled_death_does_not_repay_legacy_kill():
    info=dict(surviving_kill=jnp.array([[True,False]]),death=jnp.array([[False,True]]))
    before=SimpleNamespace(trapped=jnp.array([[0,9]]))
    filtered=credit.parent_reward_events(before,info)
    assert not bool(filtered['surviving_kill'][0,0]), 'controlled death only pays residual; original control was already credited'
    assert bool(info['surviving_kill'][0,0]), 'native event metrics remain unmodified'


def test_native_timeout_is_not_active_residual():
    before=SimpleNamespace(alive=jnp.array([[True,True]]),trapped=jnp.array([[0,1]]),t=jnp.array([4]))
    after=SimpleNamespace(alive=jnp.array([[True,False]]),trapped=jnp.array([[0,0]]),trapped_causal=jnp.array([[[False,False],[True,False]]]),trapped_source=jnp.array([[[False,False],[True,False]]]))
    info=dict(trigger_damage_source=jnp.zeros((1,2,2),bool),death=jnp.array([[False,True]]),native_trap_popper=jnp.array([[-1,-1]]))
    events=credit.native_events(before,after,info,jnp.array([False]))
    assert not bool(events['trigger'].any())


def test_direct_uncontrolled_combat_kill_retained():
    info=dict(surviving_kill=jnp.array([[True,False]]),death=jnp.array([[False,True]]))
    before=SimpleNamespace(trapped=jnp.zeros((1,2),jnp.int16))
    assert bool(credit.parent_reward_events(before,info)['surviving_kill'][0,0])
