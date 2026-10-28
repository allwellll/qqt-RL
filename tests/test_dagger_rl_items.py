from tests.external_artifacts import requires_artifacts
import jax
import jax.numpy as j
import numpy as np
from scripts import dagger_rl_oracle as O


def test_joint_release_set_cannot_be_satisfied_by_no_release():
    base=j.ones((2,5,3),bool);item=j.zeros_like(base).at[0,3,2].set(True)
    out=O.combine(base,item)
    assert out[0,3,2] and not out[0,:,0].any()
    np.testing.assert_array_equal(out[1],base[1])


def test_set_mass_has_nonzero_ability_gradient_and_zero_outside_gate():
    mask=j.zeros((1,5,3),bool).at[0,2,2].set(True)
    f=lambda a,m:O.set_loss(j.zeros((1,5)),a[None],m)
    grad=jax.grad(f)(j.zeros(3),mask)
    assert grad[2]<0 and grad[0]>0
    np.testing.assert_array_equal(jax.grad(f)(j.zeros(3),j.zeros_like(mask)),j.zeros(3))


@requires_artifacts('runs/carrier_mechanism_20261023/v1/BANK_FROZEN.json')
def test_no_future_random_or_synthetic_custody_and_no_glue_teacher():
    from scripts import combat_item_engine as T,carrier_mechanism_engine as E
    from scripts.carrier_mechanism_analysis import bank,merged
    T.configure();items=[x for x in bank() if x['record']['split']=='train']
    st=jax.tree.map(j.asarray,merged(items,'states'));ax=merged(items,'aux')
    labels,diag=jax.jit(O.teacher)(st,ax)
    forbidden=(st.held_item[:,0]!=1)&~j.any((st.field_item==1)&(st.field_owner==0),(-1,-2))
    assert not np.asarray(labels)[np.asarray(forbidden)].any()
    bad=st._replace(bun_carried=st.bun_carried.at[:,0].set(-1))
    assert not np.asarray(jax.jit(O.teacher)(bad,ax)[0]).any()
    danger=st._replace(core=st.core._replace(fuse=st.fuse.at[:,5,5].set(3)))
    assert not np.asarray(jax.jit(O.teacher)(danger,ax)[0]).any()
