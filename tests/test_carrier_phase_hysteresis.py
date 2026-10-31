"""Proposal-only phase memory from fixed production Review boundary jitter."""
import jax.numpy as jnp
import numpy as np
from scripts.carrier_phase_hysteresis import advance


def step(previous,distance,**overrides):
    fields=dict(carrying=True,alive=True,trapped=0,enemy_alive=True,enemy_trapped=0,distance=distance)
    fields.update(overrides)
    return advance(jnp.array([previous]),**{k:jnp.asarray([v]) for k,v in fields.items()})


def test_near_threat_enters_combat_immediately_then_resume_requires_clear_separation():
    previous=False;allowed=[]
    for distance in [4.4,3.9199,4.0099,3.8299,4.0099,3.8299,5.01]:
        previous,gate=step(previous,distance);previous=bool(previous[0]);allowed.append(bool(gate[0]))
    np.testing.assert_array_equal(allowed,[True,False,False,False,False,False,True])


def test_controlled_enemy_restores_route_without_distance_wait():
    latch,gate=step(True,1.,enemy_trapped=20)
    assert not bool(latch[0]) and bool(gate[0])
    latch,gate=step(True,1.,enemy_alive=False)
    assert not bool(latch[0]) and bool(gate[0])
    # Native recovery into close threat must immediately shut the route again.
    latch,gate=step(False,1.)
    assert bool(latch[0]) and not bool(gate[0])


def test_death_trap_or_custody_loss_resets_carrier_memory():
    for fields in [dict(alive=False),dict(trapped=5),dict(carrying=False)]:
        latch,gate=step(True,3.,**fields)
        assert not bool(latch[0]) and not bool(gate[0])
    # Empty hands can still target stock only in a native controlled window.
    latch,gate=step(True,3.,carrying=False,enemy_alive=False)
    assert not bool(latch[0]) and bool(gate[0])


def test_hysteresis_does_not_change_initial_distance_four_boundary_or_free_far_return():
    _,gate=step(False,4.)
    assert not bool(gate[0])
    _,gate=step(False,4.01)
    assert bool(gate[0])
    _,gate=step(True,5.)
    assert not bool(gate[0])


def test_transition_pairs_before_after_gates_and_keeps_memory_in_checkpoint_aux():
    from types import SimpleNamespace
    from scripts.carrier_phase_hysteresis import transition_gates
    def state(distance):
        return SimpleNamespace(pos=jnp.array([[[0.,0.],[0.,distance]]]),
            alive=jnp.ones((1,2),bool),trapped=jnp.zeros((1,2),int),
            bun_carried=jnp.array([[1,-1]]))
    aux={'carrier_phase_latched':jnp.array([[False,False]]),'routing':{'sentinel':7}}
    new={'routing':{'sentinel':8}}
    before,after=transition_gates(state(3.8299),state(4.0099),aux,new)
    assert bool(before['carrier_phase_latched'][0,0])
    assert bool(after['carrier_phase_latched'][0,0])
    assert not bool(before['routing']['carrier_route_allowed'][0,0])
    assert not bool(after['routing']['carrier_route_allowed'][0,0])
    assert 'carrier_route_allowed' not in aux['routing']  # immutable input
    before,after=transition_gates(state(4.0099),state(5.01),after,{'routing':{}})
    assert not bool(after['carrier_phase_latched'][0,0])
    assert bool(after['routing']['carrier_route_allowed'][0,0])


def test_jitted_rollout_split_restore_and_episode_reset_preserve_only_live_memory():
    import jax
    import pickle
    from typing import NamedTuple
    from scripts.carrier_phase_hysteresis import transition_gates
    class State(NamedTuple):
        pos:object
        alive:object
        trapped:object
        bun_carried:object
    def state(distance):
        return State(jnp.array([[[0.,0.],[0.,distance]],[[0.,0.],[0.,distance]]]),
            jnp.ones((2,2),bool),jnp.zeros((2,2),int),jnp.array([[1,-1],[1,-1]]))
    initial={'carrier_phase_latched':jnp.zeros((2,2),bool),
        'routing':{'carrier_route_allowed':jnp.zeros((2,2),bool)}}
    @jax.jit
    def block(aux,distances,terminal):
        def body(values,pair):
            distance,ended=pair
            _,new=transition_gates(state(distance),state(distance),values,initial,ended)
            # Same per-environment reset as the production collector.
            new=jax.tree.map(lambda a,b:jnp.where(ended[:,None],b,a),new,initial)
            return new,new['carrier_phase_latched']
        return jax.lax.scan(body,aux,(distances,terminal))
    distances=jnp.array([3.9,4.1,4.1,4.2,5.1])
    ended=jnp.zeros((5,2),bool).at[2,0].set(True)
    whole,history=block(initial,distances,ended)
    split,_=block(initial,distances[:2],ended[:2])
    restored=jax.tree.map(jnp.asarray,pickle.loads(pickle.dumps(jax.device_get(split))))
    resumed,_=block(restored,distances[2:],ended[2:])
    jax.tree.map(np.testing.assert_array_equal,whole,resumed)
    np.testing.assert_array_equal(history[:,0,0],[True,True,False,False,False])
    np.testing.assert_array_equal(history[:,1,0],[True,True,True,True,False])
