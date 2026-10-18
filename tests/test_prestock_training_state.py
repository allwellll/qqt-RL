"""Native preparation state must share one target and reset correctly."""
import jax
import jax.numpy as jnp
import numpy as np


def native(seat=0):
    from test_frontier_native_witness import native_barrier
    from scripts import je10_ops as ops
    state,_,_=native_barrier()
    pos=jnp.array([[6.5,8.5],[1.5,12.5]]) if seat==0 else jnp.array([[1.5,1.5],[6.5,6.5]])
    state=state._replace(core=state.core._replace(pos=pos,alive=jnp.array([seat==0,seat==1])),
        bun_respawn=jnp.array([100 if seat==1 else 0,100 if seat==0 else 0],state.bun_respawn.dtype),
        bun_loose=jnp.zeros_like(state.bun_loose))
    return jax.tree.map(lambda x:x[None],state),ops.reward_state(1)


def test_both_seats_have_stable_shared_preparation_and_opening_returns_to_stock():
    from scripts.prestock_training_state import state,update,potential
    for seat in [0,1]:
        native_state,aux=native(seat);old=state(1)
        selected=jax.jit(update)(native_state,aux,old)
        assert int(selected['objective'][0,seat])==2
        assert bool(selected['active'][0,seat]) and not bool(selected['ready'][0,seat])
        again=update(native_state,aux,selected)
        for key in selected:np.testing.assert_array_equal(selected[key],again[key])
        parent=(jnp.ones((1,2),jnp.int8),jnp.full((1,2),100001),jnp.full((1,2),-.3),
            jnp.full((1,2),30),jnp.full((1,2),2))
        values=potential(native_state,selected,parent)
        assert int(values[0][0,seat])==5
        assert int(values[3][0,seat])==int(selected['distance'][0,seat,6,8 if seat==0 else 6])
        opened=native_state._replace(core=native_state.core._replace(brick=native_state.brick.at[0,6,7].set(False)))
        direct=update(opened,aux,selected)
        assert bool(direct['ready'][0,seat]) and int(direct['objective'][0,seat])==0
        for actual,expected in zip(potential(opened,direct,parent),parent):np.testing.assert_array_equal(actual,expected)


def test_actual_attributed_loose_has_priority_even_with_stale_dagger_flags():
    from scripts.prestock_training_state import state,update
    native_state,aux=native();aux['dagger']['loose']=jnp.zeros_like(aux['dagger']['loose'])
    native_state=native_state._replace(bun_loose=native_state.bun_loose.at[0,6,9,0].set(1))
    aux['routing']['target']=aux['routing']['target'].at[0,0].set(jnp.array([6,9]))
    aux['routing']['nonce']=aux['routing']['nonce'].at[0,0].set(3)
    aux['drop']['nonce']=aux['drop']['nonce'].at[0,0].set(3)
    aux['drop']['credit']=aux['drop']['credit'].at[0,0].set(0)
    selected=update(native_state,aux,state(1))
    assert not bool(selected['active'][0,0]) and int(selected['objective'][0,0])==0
    aux['routing']['nonce']=aux['routing']['nonce'].at[0,0].set(2)
    selected=update(native_state,aux,state(1))
    assert bool(selected['active'][0,0]) and int(selected['objective'][0,0])==2


def test_death_trap_terminal_and_window_close_clear_authority_and_resume_exactly():
    from scripts.prestock_training_state import state,update,potential
    import pickle
    native_state,aux=native();selected=update(native_state,aux,state(1))
    restored=pickle.loads(pickle.dumps(jax.device_get(selected)))
    a=update(native_state,aux,selected);b=jax.jit(update)(native_state,aux,restored)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    variants=[native_state._replace(core=native_state.core._replace(alive=jnp.zeros_like(native_state.alive))),
        native_state._replace(trapped=native_state.trapped.at[0,0].set(20))]
    for changed in variants:
        result=update(changed,aux,selected)
        for key,empty in state(1).items():np.testing.assert_array_equal(result[key][0,0],empty[0,0])
    aux['terminal']=jnp.ones_like(aux['terminal']);result=update(native_state,aux,selected)
    for key,empty in state(1).items():np.testing.assert_array_equal(result[key],empty)
    aux['terminal']=jnp.zeros_like(aux['terminal'])
    recovered=native_state._replace(core=native_state.core._replace(alive=jnp.ones_like(native_state.alive)),
        bun_respawn=jnp.zeros_like(native_state.bun_respawn))
    result=update(recovered,aux,selected)
    assert not bool(result['active'].any()) and not bool(result['objective'].any())
