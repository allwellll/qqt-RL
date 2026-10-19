"""Real-drop distance admission, expiry and exact excluded gradients."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np

def fixture():
    from scripts.je09_chase_expert import state
    n=10
    st=SimpleNamespace(scene=jnp.full(n,9),alive=jnp.ones((n,2),bool),
        bun_carried=jnp.full((n,2),-1),trapped=jnp.zeros((n,2),int),
        bun_respawn=jnp.full((n,2),99),bun_loose=jnp.zeros((n,7,7,2)).at[:,1,1,1].set(1),
        pos=jnp.tile(jnp.array([[[3.5,3.5],[1.5,1.5]]]),(n,1,1)))
    aux=dict(chase=state(n),terminal=jnp.zeros(n,bool),
        drop=dict(nonce=jnp.ones((n,2),int),credit=jnp.tile(jnp.array([[1,0]]),(n,1)),cycle_paid=jnp.zeros((n,2,2),bool)),
        routing=dict(target=jnp.ones((n,2,2),int),nonce=jnp.ones((n,2),int)),
        potential=dict(drop=dict(active=jnp.ones((n,2),bool),retired=jnp.zeros((n,2),int))))
    aux['chase'].update(nonce=jnp.ones((n,2),int),admitted=jnp.ones((n,2),bool),age=jnp.ones((n,2),int),initial=jnp.full((n,2),10))
    return st,aux

def test_gate_and_exact_zero_gradients():
    from scripts.je09_chase_expert import gate,supervised_loss
    st,a=fixture();seat=jnp.zeros(10,int)
    assert bool(gate(st,a,seat).all())
    st.bun_carried=st.bun_carried.at[1,0].set(1)
    st.bun_loose=st.bun_loose.at[2].set(0)
    st.bun_respawn=st.bun_respawn.at[3,1].set(0)
    st.scene=st.scene.at[4].set(0);a['terminal']=a['terminal'].at[5].set(True)
    a['drop']['credit']=a['drop']['credit'].at[6,1].set(-1)
    a['chase']['admitted']=a['chase']['admitted'].at[7,0].set(False)
    a['chase']['age']=a['chase']['age'].at[8,0].set(100)
    a['drop']['cycle_paid']=a['drop']['cycle_paid'].at[9,0,1].set(True)
    active=gate(st,a,seat);np.testing.assert_array_equal(active,jnp.arange(10)==0)
    target=jnp.tile(jnp.array([.5,0,.5,0,0]),(10,1));mask=jnp.ones((10,5),bool)
    grad=jax.grad(lambda q:supervised_loss(q,mask,target,active,.1,.5)[0])(jnp.zeros((10,5)))
    assert grad[0,0]==grad[0,2]<0;np.testing.assert_array_equal(grad[1:],0)

def test_distance_admission_native_nonce_and_recovery():
    from scripts.je09_chase_expert import admit
    d=jnp.array([2,8,10,12,13,32,32767]);ok=admit(d,jnp.array([8,12]))
    np.testing.assert_array_equal(ok,jnp.array([True,True,True,True,False,False,False]))
    # Bucket identity belongs to the drop, not changing current distance.
    st,a=fixture();from scripts.je09_chase_expert import gate
    a['chase']['initial']=a['chase']['initial'].at[0,0].set(10)
    st.pos=st.pos.at[0,0].set(jnp.array([5.5,5.5]))
    assert bool(gate(st,a,jnp.zeros(10,int))[0])

def test_safe_set_and_unreachable_exclusion():
    from scripts.je09_chase_expert import action_set
    from scripts.je08_bun_potential import shortest,INF
    wall=jnp.zeros((1,7,7),bool);goal=wall.at[:,1,1].set(True)
    d=shortest(wall,wall,goal);legal=jnp.ones((1,5),bool);danger=jnp.zeros_like(wall,float)
    target,ok=action_set(jnp.array([[3.5,3.5]]),d,legal,danger,wall,wall)
    np.testing.assert_array_equal(target,jnp.array([[.5,0,.5,0,0]]));assert bool(ok[0])
    for distance,hazard in [(jnp.full_like(d,INF),danger),(d,danger.at[:,3,3].set(1))]:
        assert not bool(action_set(jnp.array([[3.5,3.5]]),distance,legal,hazard,wall,wall)[1].any())

def test_real_drop_update_and_nonce_frozen_admission():
    from scripts.je09_chase_expert import update,state,gate
    st,a=fixture();n=10;seat=jnp.zeros(n,int)
    a['expert']=dict(distance=jnp.full((n,2,7,7),10))
    a['chase']=state(n)
    assert not bool(gate(st,a,seat).any())
    a['chase']=update(st,a,a['chase'],jnp.array([8,12]))
    assert bool(gate(st,a,seat).all())
    assert bool((a['chase']['age']==0).all())
    # Leaving the admitted distance bucket does not remove recovery labels.
    a['expert']['distance']=jnp.full((n,2,7,7),19)
    a['chase']=update(st,a,a['chase'],jnp.array([12,16]))
    assert bool(a['chase']['admitted'].all()) and bool((a['chase']['initial']==10).all())
    # A new nonce is independently admitted and cannot inherit old eligibility.
    a['drop']['nonce']=jnp.full((n,2),2)
    a['chase']=update(st,a,a['chase'],jnp.array([8,12]))
    assert not bool(a['chase']['admitted'].any())
