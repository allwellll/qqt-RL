"""Cached preparation goals need edge identities and continuous PBRS edges."""
import jax
import jax.numpy as jnp
import numpy as np


def fixture():
    from scripts.prestock_route_preparation import plan,selection_state
    from tests.test_prestock_route_preparation import corridor
    hard,brick,home,stock=corridor();pos=jnp.array([[2.5,4.5]])
    proposed=plan(pos,hard,brick,home,stock,jnp.array([True]))
    return hard,brick,home,stock,pos,proposed,selection_state(1,5,7)


def test_goal_selection_has_one_identity_per_edge_and_reset_clears_it():
    from scripts.prestock_route_preparation import select
    hard,brick,home,stock,pos,proposed,empty=fixture()
    chosen=jax.jit(select)(empty,proposed,hard,brick,pos,jnp.array([False]))
    assert int(chosen['revision'][0])==1 and int(chosen['objective'][0])==2
    continued=select(chosen,proposed,hard,brick,pos,jnp.array([False]))
    assert int(continued['revision'][0])==1
    reset=select(continued,proposed,hard,brick,pos,jnp.array([True]))
    for name in ['revision','objective','staging','first_bricks']:np.testing.assert_array_equal(reset[name],empty[name])
    np.testing.assert_array_equal(reset['distance'],empty['distance'])


def test_same_topology_retains_reachable_goal_set_until_real_opening():
    from scripts.prestock_route_preparation import select,plan
    hard,brick,home,stock,pos,proposed,empty=fixture()
    chosen=select(empty,proposed,hard,brick,pos,jnp.array([False]))
    # A newly proposed tied alternative cannot churn a still reachable goal.
    alternative=proposed._replace(staging=jnp.zeros_like(proposed.staging).at[0,1,4].set(True))
    kept=select(chosen,alternative,hard,brick,pos,jnp.array([False]))
    np.testing.assert_array_equal(kept['staging'],chosen['staging'])
    assert int(kept['revision'][0])==1
    opened=brick.at[:,2,3].set(False)
    ready=plan(pos,hard,opened,home,stock,jnp.array([True]))
    next_=select(kept,ready,hard,opened,pos,jnp.array([False]))
    assert int(next_['objective'][0])==0 and int(next_['revision'][0])==2
    assert not bool(next_['staging'].any())


def test_preparation_pbrs_target_matches_staging_and_has_no_phase_boundary_payment():
    from scripts.prestock_route_preparation import select,potential_values
    from scripts.je09_route_pbrs import transition
    hard,brick,home,stock,pos,proposed,empty=fixture()
    chosen=select(empty,proposed,hard,brick,pos,jnp.array([False]))
    parent=(jnp.array([1]),jnp.array([100001]),jnp.array([-.5]),jnp.array([32]),jnp.array([2]))
    values=potential_values(pos,hard,brick,chosen,parent)
    assert int(values[0][0])==5 and float(values[2][0])==0
    off=potential_values(pos+jnp.array([[0.,.3]]),hard,brick,chosen,parent)
    assert float(off[2][0])<float(values[2][0])
    ledger=dict(phi=jnp.array([-.2]),offset=jnp.zeros(1),phase=parent[0],identity=parent[1],retired=jnp.array([-1]))
    _,delta,before,after=transition(ledger,*values[:2],values[2],*values[:2],values[2],False)
    np.testing.assert_allclose(before,after)
    np.testing.assert_allclose(delta,.001,atol=1e-7)
    for phase in [0,2,3,4]:
        other=(jnp.array([phase]),)+parent[1:]
        for a,b in zip(potential_values(pos,hard,brick,chosen,other),other):np.testing.assert_array_equal(a,b)
