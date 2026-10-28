"""Stage B post-pickup window, crossing and counterfactual contracts."""
import copy
import pytest
from tests.test_je09_bridge import bridge_event
from scripts.je09_bun_bridge import jax_bridge_reward, jax_bridge_state


def picked(home=False):
    _, s, _ = jax_bridge_reward(bridge_event(drop=True), jax_bridge_state(1), window=120)
    _, s, _ = jax_bridge_reward(bridge_event(pickup_loose=True, carried=True, home_half=home), s, window=120)
    return s


def test_window_excludes_late_milestones_and_has_exact_boundary():
    s=picked();s['ticks']=s['ticks']+119
    r,_,_=jax_bridge_reward(bridge_event(carried=True,after_distance=5,home_half=True),s,window=120)
    assert float(r[0,0])==5.5
    s['ticks']=s['ticks']+1
    r,_,_=jax_bridge_reward(bridge_event(carried=True,after_distance=5,home_half=True),s,window=120)
    assert float(r[0,0])==0


def test_home_mark_requires_real_crossing_and_net_progress():
    r,_,m=jax_bridge_reward(bridge_event(carried=True,home_half=True,after_distance=5),picked(True),window=120)
    assert not bool(m['bridge_events'][0,0,5])
    r,_,m=jax_bridge_reward(bridge_event(carried=True,home_half=True,after_distance=12),picked(),window=120)
    assert not bool(m['bridge_events'][0,0,5])


def test_scaled_milestones_are_once_and_same_tick_clawback_is_exact():
    s=picked();event=bridge_event(carried=True,after_distance=5,home_half=True)
    r,s,_=jax_bridge_reward(event,s,window=120,scale=2.)
    assert float(r[0,0])==11
    r,s,_=jax_bridge_reward(event,s,window=120,scale=2.)
    assert float(r[0,0])==0
    saved=copy.deepcopy(s)
    r,s,_=jax_bridge_reward(bridge_event(lost=True),s,window=120,scale=2.)
    assert float(r[0,0])==-11
    r,_,_=jax_bridge_reward(bridge_event(timeout=True),saved,window=120,scale=2.)
    assert float(r[0,0])==-11
    r,_,_=jax_bridge_reward(event,picked(),window=120,scale=0.)
    assert float(r[0,0])==0


def test_stage_b_combat_replay_birth_reserve_is_explicit():
    import jax.numpy as jnp
    from scripts.je09_bun_ops import curriculum_weights
    weights=curriculum_weights(jnp.zeros((10,70)),combat_fraction=.60,bridge_fraction=.25)
    assert float(weights[0])==pytest.approx(.60)
    assert float(weights[9])==pytest.approx(.25)
    assert float(weights.sum())==pytest.approx(1.)
