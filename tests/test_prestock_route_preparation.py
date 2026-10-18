"""Inventory pickup needs a connected return route while placement is legal."""
import jax
import jax.numpy as jnp
import numpy as np


def corridor():
    hard=jnp.ones((1,5,7),bool).at[:,1:4,1:6].set(False)
    brick=jnp.zeros_like(hard).at[:,1:4,3].set(True)
    home=jnp.zeros_like(hard).at[:,2,1].set(True)
    stock=jnp.zeros_like(hard).at[:,2,5].set(True)
    return hard,brick,home,stock


def test_near_enemy_stock_prepares_home_connection_before_picking_inventory():
    from scripts.prestock_route_preparation import plan
    hard,brick,home,stock=corridor()
    result=jax.jit(plan)(jnp.array([[2.5,4.5]]),hard,brick,home,stock,jnp.array([True]))
    assert not bool(result.ready[0]) and bool(result.preparing[0])
    assert int(result.objective[0])==2
    np.testing.assert_array_equal(np.argwhere(result.staging[0]),[[2,4]])
    np.testing.assert_array_equal(np.argwhere(result.first_bricks[0]),[[2,3]])
    assert not bool((result.staging&stock).any())


def test_from_home_component_prepares_forward_corridor_to_enemy_stock():
    from scripts.prestock_route_preparation import plan
    hard,brick,home,stock=corridor()
    result=plan(jnp.array([[2.5,1.5]]),hard,brick,home,stock,jnp.array([True]))
    assert int(result.objective[0])==3
    np.testing.assert_array_equal(np.argwhere(result.staging[0]),[[2,2]])


def test_existing_detour_or_real_opening_allows_direct_stock_and_return():
    from scripts.prestock_route_preparation import plan
    hard,brick,home,stock=corridor()
    for opened in [brick.at[:,1,3].set(False),brick.at[:,2,3].set(False)]:
        result=plan(jnp.array([[2.5,4.5]]),hard,opened,home,stock,jnp.array([True]))
        assert bool(result.ready[0]) and not bool(result.preparing[0])
        assert int(result.objective[0])==1 and not bool(result.first_bricks.any())
        assert int(result.distance[0,2,4])==1


def test_inactive_or_hard_disconnection_never_invents_an_opening_label():
    from scripts.prestock_route_preparation import plan
    hard,brick,home,stock=corridor()
    for blocked,boxes,active in [(hard,brick,False),(hard|brick,jnp.zeros_like(brick),True)]:
        result=plan(jnp.array([[2.5,4.5]]),blocked,boxes,home,stock,jnp.array([active]))
        assert int(result.objective[0])==0
        assert not bool(result.staging.any()) and not bool(result.first_bricks.any())
        assert int(result.distance[0,2,4])==32767


def test_native_adapter_requires_free_empty_learner_real_inventory_and_window():
    from scripts.prestock_route_preparation import native_plan
    from test_route_objective_consistency import native_case
    state,aux,_=native_case()
    for seat in [0,1]:
        current=state if seat==0 else jax.tree.map(lambda x:x,state)._replace(
            core=state.core._replace(alive=state.alive[:,::-1],pos=state.pos[:,::-1]),
            bun_respawn=state.bun_respawn[:,::-1])
        # The existing loose target has priority; the adapter cannot redirect it.
        aux['dagger']['loose']=jnp.zeros_like(aux['dagger']['loose'])
        result=native_plan(current,aux,jnp.array([seat]))
        assert bool(result.ready[0]) and int(result.objective[0])==1
        enemy=1-seat
        variants=[
            current._replace(bun_carried=current.bun_carried.at[0,seat].set(enemy)),
            current._replace(trapped=current.trapped.at[0,seat].set(20)),
            current._replace(core=current.core._replace(alive=current.alive.at[0,seat].set(False))),
            current._replace(bun_respawn=jnp.zeros_like(current.bun_respawn),
                core=current.core._replace(alive=jnp.ones_like(current.alive))),
            current._replace(bun_stored=current.bun_stored.at[0,enemy,enemy].set(0)),
        ]
        for modified in variants:
            rejected=native_plan(modified,aux,jnp.array([seat]))
            assert int(rejected.objective[0])==0 and not bool(rejected.preparing[0])
        for key in ['terminal','loose']:
            old=aux['terminal'] if key=='terminal' else aux['dagger']['loose']
            if key=='terminal':aux[key]=jnp.ones_like(old)
            else:aux['dagger'][key]=old.at[0,seat,5,2].set(True)
            assert int(native_plan(current,aux,jnp.array([seat])).objective[0])==0
            if key=='terminal':aux[key]=old
            else:aux['dagger'][key]=old


def test_preparation_movement_never_teaches_staging_wait_or_stock_contact():
    from scripts.prestock_route_preparation import movement_labels
    hard,brick,home,stock=corridor()
    danger=jnp.zeros_like(brick);fuse=jnp.zeros_like(brick,int)
    legal=jnp.ones((1,5),bool)
    # At the staging center, the ability witness owns the next step. A wait
    # movement label would conflict with joint opening/escape supervision.
    target,active,result=movement_labels(jnp.array([[2.5,4.5]]),hard,brick,home,stock,
        jnp.array([True]),legal,danger,fuse)
    assert int(result.objective[0])==2
    assert not bool(active[0]) and not bool(target.any())
    # Off-center approach must never walk right into inventory before home is ready.
    target,active,result=movement_labels(jnp.array([[2.5,4.8]]),hard,brick,home,stock,
        jnp.array([True]),legal,danger,fuse)
    assert bool(active[0]) and float(target[0,2])>0
    assert float(target[0,3])==float(target[0,4])==0


def test_native_pushable_barrier_is_blocked_until_a_real_transition_opens_it():
    from scripts.prestock_route_preparation import native_plan
    from test_route_objective_consistency import native_case
    state,aux,_=native_case()
    wall=jnp.zeros_like(state.wall).at[:,0].set(True).at[:,-1].set(True).at[:,:,0].set(True).at[:,:,-1].set(True)
    pushable=jnp.zeros_like(wall).at[:,1:12,7].set(True)
    state=state._replace(core=state.core._replace(wall=wall,pushable=pushable))
    aux['dagger']['loose']=jnp.zeros_like(aux['dagger']['loose'])
    result=native_plan(state,aux,jnp.array([0]))
    assert not bool(result.ready[0]) and int(result.objective[0])==0
    opened=state._replace(core=state.core._replace(pushable=jnp.zeros_like(pushable)))
    assert bool(native_plan(opened,aux,jnp.array([0])).ready[0])
