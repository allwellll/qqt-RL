"""Production stock gates, including stale credit and expired native windows."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def fixture(seat=0):
    from tests.test_route_objective_consistency import native_case
    from scripts import je10_ops as ops
    s,aux,route=native_case()
    # No real loose target: inventory is the only empty-hand objective.
    s=s._replace(core=s.core._replace(alive=jnp.ones_like(s.alive)),
        trapped=jnp.zeros_like(s.trapped),bun_respawn=jnp.zeros_like(s.bun_respawn),
        bun_loose=jnp.zeros_like(s.bun_loose))
    aux=ops.reward_state(1)
    # Deliberately stale cache authority must not grant stock supervision.
    aux['dagger'].update(window=jnp.ones((1,2),bool),custody=jnp.ones((1,2),bool),
        blocked=s.wall|s.brick,distance=jnp.zeros_like(aux['dagger']['distance']))
    return s,aux,route,jnp.array([seat])


@pytest.mark.parametrize('seat',[0,1])
@pytest.mark.parametrize('stale_timer',[0,100])
def test_free_enemy_never_gets_stock_phase_or_teacher_even_with_stale_credit(seat,stale_timer):
    from scripts import je10_ops as ops,je10_route_oracle as oracle
    from scripts.prestock_training_state import update,state,potential
    s,aux,route,who=fixture(seat)
    s=s._replace(bun_respawn=jnp.full_like(s.bun_respawn,stale_timer),
        move_status=jnp.ones_like(s.move_status),status_ticks=jnp.full_like(s.status_ticks,100))
    phase=ops.safe_phase_values(s,route,aux['drop'])
    assert int(phase[0][0,seat])==0
    selected=jax.jit(update)(s,aux,state(1))
    assert not bool(selected['active'][0,seat])
    values=potential(s,selected,phase)
    assert int(values[0][0,seat])==0
    label=oracle.labels(s,aux,jnp.ones((1,5),bool),who,jnp.zeros((1,13,15)))
    assert int(label[0,6])==2  # candidate kind is not eligibility
    assert float(label[0,5])==0 and float(label[0,7])==0


@pytest.mark.parametrize('seat',[0,1])
@pytest.mark.parametrize('window',['dead','sugar'])
def test_current_native_window_preserves_stock_teacher(seat,window):
    from scripts import je10_ops as ops,je10_route_oracle as oracle
    s,aux,route,who=fixture(seat)
    if window=='dead':
        s=s._replace(core=s.core._replace(alive=s.alive.at[0,1-seat].set(False)),
            bun_respawn=s.bun_respawn.at[0,1-seat].set(50))
    else:s=s._replace(trapped=s.trapped.at[0,1-seat].set(50))
    assert int(ops.safe_phase_values(s,route,aux['drop'])[0][0,seat])==1
    label=oracle.labels(s,aux,jnp.ones((1,5),bool),who,jnp.zeros((1,13,15)))
    assert float(label[0,5])==1 and float(label[0,7])==1


def test_expired_window_cannot_start_stock_custody_but_real_carrier_keeps_return():
    from scripts import je10_ops as ops,je10_route_oracle as oracle
    s,aux,route,who=fixture()
    aux['dagger']['custody']=jnp.zeros((1,2),bool)
    after=s._replace(bun_carried=s.bun_carried.at[0,0].set(1),
        bun_stored=s.bun_stored.at[0,1,1].add(-1))
    info=dict(surviving_kill=jnp.zeros((1,2),bool),steal=jnp.array([[True,False]]),
        pickup_loose=jnp.zeros((1,2),bool),pickup_origin=jnp.full((1,2),-1),native_transfer=jnp.zeros((1,2),bool))
    updated=ops.update_custody(s,after,info,aux,aux['dagger'])
    assert not bool(updated['custody'][0,0])
    # Existing causal cargo remains eligible after enemy recovery at distance.
    aux['dagger']['custody']=jnp.array([[True,False]])
    assert int(ops.safe_phase_values(after,route,aux['drop'])[0][0,0]) in (3,4)
    label=oracle.labels(after,aux,jnp.ones((1,5),bool),who,jnp.zeros((1,13,15)))
    assert float(label[0,7])==1


def test_inactive_pbrs_residual_is_not_stock_directional_credit():
    from scripts.je09_route_pbrs import state,transition
    ledger=state(1)
    ledger['phi']=jnp.full((1,2),-.5)
    zero=jnp.zeros((1,2),jnp.int8);identity=jnp.full((1,2),-1)
    results=[]
    for invalid_stock_base in [-.9,-.1]:
        _,delta,_,_=transition(ledger,zero,identity,jnp.full((1,2),invalid_stock_base),
            zero,identity,jnp.full((1,2),invalid_stock_base),False)
        results.append(delta)
    np.testing.assert_array_equal(*results)
    np.testing.assert_allclose(results[0],.0025,atol=1e-7)


def test_direct_free_stock_take_has_no_transport_bonus_but_delivery_remains_terminal():
    from scripts.je05_bun_transport import transport_state,transport_reward,CAPTURE_REWARD
    z=jnp.zeros(1,bool)
    e=dict(token=jnp.ones(1,int),carried=jnp.ones(1,bool),capture=z,steal=~z,eligible=z,
        before_region=jnp.full(1,-1),after_region=jnp.full(1,-1),home_half=z,
        death=z,drop=z,trade=z,pickup=z,settled=~z)
    reward,_,metrics=transport_reward(e,transport_state(1))
    assert float(reward[0])==0 and bool(metrics['unconditional_steal'][0])
    reward,_,_=transport_reward(dict(e,steal=z,capture=~z),transport_state(1))
    assert float(reward[0])==CAPTURE_REWARD


def test_native_free_stock_pickup_is_a_game_action_without_stage_bonus(monkeypatch):
    from scripts import je10_ops as ops
    from jax_bomb import je07_bun_env as env
    from scripts import je09_bun_scenes as scenes
    monkeypatch.setattr(ops.pbrs,'phase_values',ops.safe_phase_values)
    env.prepare()
    s=scenes.init_batch(jax.random.PRNGKey(333),1,jax.nn.one_hot(8,10))
    pos=s.pos.at[0,0].set(env._BUN_BASES[1].astype(jnp.float32)+1.5).at[0,1].set(jnp.array([10.5,7.5]))
    s=s._replace(core=s.core._replace(pos=pos,alive=jnp.ones_like(s.alive),
        brick=jnp.zeros_like(s.brick),pushable=jnp.zeros_like(s.core.pushable),
        fuse=jnp.zeros_like(s.fuse),blast_linger=jnp.zeros_like(s.core.blast_linger)),
        trapped=jnp.zeros_like(s.trapped),bun_carried=jnp.full_like(s.bun_carried,-1),
        bun_respawn=jnp.zeros_like(s.bun_respawn),bun_loose=jnp.zeros_like(s.bun_loose))
    action=jnp.array([[[4,0],[4,0]]])
    after,done,info=jax.jit(jax.vmap(lambda s,a:env.step(s,a,jax.random.PRNGKey(1),False,True)))(s,action)
    assert bool(info['steal'][0,0]) and int(after.bun_carried[0,0])==1
    assert bool(after.alive[0,1]) and int(after.trapped[0,1])==0
    assert not bool(info['steal_window_dead'][0,0]|info['steal_window_trap'][0,0])
    info.update(h_joint_actions=action,h_route_terminal=done,blocked_own_route=jnp.zeros((1,2),bool))
    reward,_,_,metrics=jax.jit(ops.combined_reward)(s,after,info,ops.reward_state(1),jnp.ones(1,bool))
    assert float(metrics['shaping'][0,0])==0
    assert float(metrics['route_shaped_reward'][0,0])==0
    assert float(reward[0,0])==0
