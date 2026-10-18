"""Verify the production-shaped bounded sampler and extended label payload."""
import jax
import jax.numpy as jnp
import numpy as np


def test_bounded_native_verifier_emits_shared_preparation_payload_for_disconnected_stock():
    from scripts.prestock_verified_labels import verified_labels
    from scripts.prestock_training_state import state,update
    from scripts.je10_route_oracle import labels
    from jax_bomb import je07_bun_env as env
    from test_prestock_native_teacher import fixture
    from scripts import je10_ops as ops
    st,_=fixture();n=4;st=jax.tree.map(lambda x:jnp.repeat(x,n,0),st)
    aux=ops.reward_state(n);aux['prestock']=update(st,aux,state(n))
    legal,_=jax.vmap(env.legal_mask)(st);seat=jnp.zeros(n,jnp.int32)
    danger=jnp.zeros((n,13,15))
    result=jax.jit(lambda s,a:verified_labels(s,a,legal[:,0],seat,danger,labels,1))(st,aux)
    assert result.shape==(n,31)  # original12 + joint15 + preparation/revision/objective/opening
    accepted=np.asarray(result[:,5]>0)
    assert accepted.sum()==1  # reuse the original one stock slot, no extra query budget
    assert np.asarray(result[:,7]>0).all() and np.asarray(result[:,27]>0).all()
    assert np.asarray(result[:,28]==aux['prestock']['revision'][:,0]).all()
    joint=np.asarray(result[:,12:27]).reshape(n,5,3)>0
    assert joint[accepted,:,0].any() and not joint[accepted,:,1:].any()
    assert not joint[~accepted].any()
    assert np.asarray(result[:,8]>0).sum()==1


def test_ordinary_fragment_labels_preserve_legacy_all_legal_ability_semantics():
    from scripts.prestock_verified_labels import verified_labels
    from scripts.prestock_training_state import state
    from scripts.je10_route_oracle import labels
    from scripts import je10_ops as ops
    from jax_bomb import je07_bun_env as env
    from test_prestock_native_teacher import fixture
    st,_=fixture();n=4;st=jax.tree.map(lambda x:jnp.repeat(x,n,0),st)
    # Carrying learner has native safe return permission; it cannot place a bomb.
    st=st._replace(core=st.core._replace(brick=jnp.zeros_like(st.brick)),
        bun_carried=st.bun_carried.at[:,0].set(1))
    aux=ops.reward_state(n);aux['prestock']=state(n)
    aux['dagger']['custody']=aux['dagger']['custody'].at[:,0].set(True)
    from scripts.je08_bun_potential import shortest
    bases=jnp.stack([env._base_mask(0),env._base_mask(1)])
    aux['dagger']['distance']=aux['dagger']['distance'].at[:,4].set(shortest(st.wall,st.brick,jnp.broadcast_to(bases[0],st.wall.shape)))
    aux['dagger']['blocked']=st.wall|st.brick|st.core.pushable
    legal,ability=jax.vmap(env.legal_mask)(st);seat=jnp.zeros(n,jnp.int32)
    result=jax.jit(lambda s,a:verified_labels(s,a,legal[:,0],seat,jnp.zeros((n,13,15)),labels,1))(st,aux)
    accepted=np.asarray(result[:,5]>0);assert accepted.sum()==1
    joint=np.asarray(result[:,12:27]).reshape(n,5,3)>0
    expected=(np.asarray(result[:,:5])>0)[:,:,None]&np.asarray(ability[:,0])[:,None,:]
    np.testing.assert_array_equal(joint,expected)
    assert not np.asarray(result[:,27]>0).any()
