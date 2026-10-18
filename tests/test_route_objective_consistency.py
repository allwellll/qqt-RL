"""Proposal-only alignment of attributed own recovery with the route teacher."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np


def fixture():
    from scripts.je08_bun_potential import shortest
    state=SimpleNamespace(pos=jnp.array([[[5.5,5.5],[5.5,9.5]]]),
        alive=jnp.ones((1,2),bool),trapped=jnp.full((1,2),20),
        bun_carried=jnp.full((1,2),-1),wall=jnp.zeros((1,13,15),bool),
        bun_loose=jnp.zeros((1,13,15,2),int).at[0,5,2,0].set(1).at[0,5,12,1].set(1))
    # Each player is controlled by its counterpart's native trap, but learner
    # itself must be free. Tests select which seat can act independently.
    targets=jnp.array([[[5,2],[5,12]]]);nonce=jnp.array([[3,7]])
    goals=jnp.zeros((1,2,13,15),bool).at[0,0,5,12].set(True).at[0,1,5,2].set(True)
    distance=shortest(state.wall[:,None],jnp.zeros((1,1,13,15),bool),goals)
    route=dict(target=targets,nonce=nonce,distance=jnp.concatenate([distance,distance],1))
    drop=dict(nonce=nonce,credit=jnp.array([[0,1]]))
    values=(jnp.ones((1,2),jnp.int8),jnp.full((1,2),100003),
        jnp.full((1,2),-.5),jnp.full((1,2),32),jnp.full((1,2),2))
    return state,route,drop,values


def test_both_seats_own_recovery_replaces_stock_and_selects_own_origin_map():
    from scripts.route_objective_consistency import align_recovery
    for seat in range(2):
        state,route,drop,values=fixture();state.trapped=state.trapped.at[0,seat].set(0)
        phase,identity,base,distance,provenance=align_recovery(state,route,drop,values)
        assert int(phase[0,seat])==2 and int(identity[0,seat])< -1
        assert int(distance[0,seat])==3 and float(base[0,seat])<0
        assert int(provenance[0,seat])==int(drop['nonce'][0,seat])
        other=1-seat
        assert int(phase[0,other])==1


def test_enemy_drop_and_carry_remain_higher_priority_than_own_recovery():
    from scripts.route_objective_consistency import align_recovery
    state,route,drop,values=fixture();state.trapped=state.trapped.at[0,0].set(0)
    for phase in [2,3,4]:
        parent=(jnp.full((1,2),phase),)+values[1:]
        result=align_recovery(state,route,drop,parent)
        for a,b in zip(result,parent):np.testing.assert_array_equal(a,b)


def test_no_recovery_for_stale_foreign_absent_or_closed_native_window():
    from scripts.route_objective_consistency import align_recovery
    for case in ['stale','foreign','absent','window','dead','carrying']:
        state,route,drop,values=fixture();state.trapped=state.trapped.at[0,0].set(0)
        if case=='stale':route=dict(route,nonce=route['nonce'].at[0,0].set(2))
        if case=='foreign':drop=dict(drop,credit=drop['credit'].at[0,0].set(1))
        if case=='absent':state.bun_loose=jnp.zeros_like(state.bun_loose)
        if case=='window':state.trapped=jnp.zeros_like(state.trapped)
        if case=='dead':state.alive=state.alive.at[0,0].set(False)
        if case=='carrying':state.bun_carried=state.bun_carried.at[0,0].set(1)
        result=align_recovery(state,route,drop,values)
        for a,b in zip(result,values):np.testing.assert_array_equal(a,b)


def test_origin_identity_is_distinct_and_phase_switch_has_no_boundary_payment():
    from scripts.route_objective_consistency import align_recovery
    from scripts.je09_route_pbrs import transition
    state,route,drop,values=fixture();state.trapped=state.trapped.at[0,0].set(0)
    result=align_recovery(state,route,drop,values)
    ledger=dict(phi=jnp.full((1,2),-.2),offset=jnp.zeros((1,2)),
        phase=values[0],identity=values[1],retired=jnp.full((1,2),-1))
    new,delta,before,after=transition(ledger,*result[:2],result[2],*result[:2],result[2],False)
    np.testing.assert_allclose(before,after)
    # Only seat 0 changes its objective; its old -.2 potential is preserved.
    np.testing.assert_allclose(before[0,0],-.2,atol=1e-7)
    np.testing.assert_allclose(delta[0,0],.001,atol=1e-7)
    new,delta,before,after=transition(new,*result[:2],result[2],*result[:2],result[2],True)
    np.testing.assert_array_equal(after,0)


def native_case():
    import jax
    from scripts import je09_bun_scenes as scenes,je10_ops as ops,je10_route_oracle as oracle
    from scripts.je08_bun_potential import shortest
    from jax_bomb import je07_bun_env as env
    state=scenes.init_batch(jax.random.PRNGKey(20261018),1,jax.nn.one_hot(8,10))
    state=state._replace(core=state.core._replace(
        pos=jnp.array([[[5.5,5.5],[1.5,12.5]]]),alive=jnp.array([[True,False]]),
        wall=jnp.zeros_like(state.wall),brick=jnp.zeros_like(state.brick),
        fuse=jnp.zeros_like(state.fuse),pushable=jnp.zeros_like(state.core.pushable)),
        trapped=jnp.zeros_like(state.trapped),bun_respawn=jnp.array([[0,50]],state.bun_respawn.dtype),
        bun_carried=jnp.full_like(state.bun_carried,-1),
        bun_loose=jnp.zeros_like(state.bun_loose).at[0,5,2,0].set(1))
    aux=ops.reward_state(1)
    aux['drop']['credit']=aux['drop']['credit'].at[0,0].set(0)
    aux['drop']['nonce']=aux['drop']['nonce'].at[0,0].set(1)
    loose=jnp.zeros((1,2,13,15),bool).at[0,0,5,2].set(True)
    route=dict(aux['routing'],target=jnp.array([[[5,2],[-1,-1]]]),nonce=jnp.array([[1,0]]))
    bases=jnp.stack([env._base_mask(0),env._base_mask(1)])
    cached_goals=jnp.concatenate([loose[:,::-1],jnp.broadcast_to(bases,(1,2,13,15))],1)
    route['distance']=shortest(state.wall[:,None],state.brick[:,None],cached_goals)
    route['stock_distance']=shortest(state.wall[:,None],state.brick[:,None],jnp.broadcast_to(bases[::-1],(1,2,13,15)))
    oracle_goals=jnp.concatenate([loose,jnp.broadcast_to(bases[::-1],(1,2,13,15)),jnp.broadcast_to(bases,(1,2,13,15))],1)
    aux['dagger'].update(loose=loose,loose_origin=jnp.array([[0,1]]),
        blocked=state.wall|state.brick,distance=shortest(state.wall[:,None],state.brick[:,None],oracle_goals))
    return state,aux,route


def test_constructed_native_transition_progresses_the_existing_teacher_target():
    import jax
    from scripts import je10_ops as ops,je10_route_oracle as oracle
    from scripts.route_objective_consistency import align_recovery
    from jax_bomb import je07_bun_env as env
    state,aux,route=native_case()
    label=oracle.labels(state,aux,jnp.ones((1,5),bool),jnp.zeros(1,int),jnp.zeros((1,13,15)))
    assert int(label[0,6])==4 and float(label[0,2])==1
    parent=ops.safe_phase_values(state,route,aux['drop'])
    assert int(parent[0][0,0])==2
    aligned=align_recovery(state,route,aux['drop'],parent)
    for a,b in zip(aligned,parent):np.testing.assert_array_equal(a,b)
    joint=jnp.array([[[2,0],[4,0]]])
    after,_,_=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(
        state,joint,jax.random.split(jax.random.PRNGKey(23),1))
    after_aligned=align_recovery(after,route,aux['drop'],ops.safe_phase_values(after,route,aux['drop']))
    assert int(aligned[0][0,0])==int(after_aligned[0][0,0])==2
    assert float(after_aligned[2][0,0])>float(aligned[2][0,0])


def test_native_own_pickup_continues_home_target_instead_of_freezing_potential():
    import jax
    from scripts import je10_ops as ops,je10_route_oracle as oracle
    from scripts.route_objective_consistency import align_recovery
    from jax_bomb import je07_bun_env as env
    state,aux,route=native_case()
    state=state._replace(core=state.core._replace(pos=state.pos.at[0,0].set(jnp.array([5.5,2.5]))))
    after,_,info=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(
        state,jnp.array([[[4,0],[4,0]]]),jax.random.split(jax.random.PRNGKey(25),1))
    assert bool(info['pickup_loose'][0,0]) and int(after.bun_carried[0,0])==0
    aux['dagger']['custody']=aux['dagger']['custody'].at[0,0].set(True)
    aux['dagger']['loose']=jnp.zeros_like(aux['dagger']['loose'])
    label=oracle.labels(after,aux,jnp.ones((1,5),bool),jnp.zeros(1,int),jnp.zeros((1,13,15)))
    assert int(label[0,6])==3 and float(label[0,7])>0
    parent=ops.safe_phase_values(after,route,aux['drop'])
    assert int(parent[0][0,0])==3
    aligned=align_recovery(after,route,aux['drop'],parent)
    for a,b in zip(aligned,parent):np.testing.assert_array_equal(a,b)
    assert int(aligned[0][0,0])==3 and float(aligned[2][0,0])<0
    moved,_,_=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(
        after,jnp.array([[[0,0],[4,0]]]),jax.random.split(jax.random.PRNGKey(26),1))
    following=align_recovery(moved,route,aux['drop'],ops.safe_phase_values(moved,route,aux['drop']))
    assert float(following[2][0,0])>float(aligned[2][0,0])
