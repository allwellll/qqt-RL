"""Stage C scope, equal shortest edges, safe turns and exact loss gradients."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np

def test_equivalent_safe_edges_and_offroute_recovery():
    from scripts.je09_return_expert import action_set
    from scripts.je08_bun_potential import shortest,INF
    wall=jnp.zeros((1,7,7),bool).at[:,0].set(True).at[:,-1].set(True).at[:,:,0].set(True).at[:,:,-1].set(True)
    goal=jnp.zeros_like(wall).at[:,1,1].set(True)
    d=shortest(wall,jnp.zeros_like(wall),goal)
    legal=jnp.ones((1,5),bool);danger=jnp.zeros_like(wall,dtype=jnp.float32)
    for pos in [[[3.5,3.5]],[[4.5,4.5]]]:
        target,valid=action_set(jnp.array(pos),d,legal,danger,wall,jnp.zeros_like(wall))
        np.testing.assert_array_equal(target[0],jnp.array([.5,0,.5,0,0]));assert bool(valid[0])
    target,valid=action_set(jnp.array([[3.5,3.5]]),d,legal,danger.at[:,2,3].set(1),wall,jnp.zeros_like(wall))
    np.testing.assert_array_equal(target[0],jnp.array([0,0,1,0,0]))
    assert not bool(action_set(jnp.array([[3.5,3.5]]),jnp.full_like(d,INF),legal,danger,wall,jnp.zeros_like(wall))[1].any())
    # Off-centre corner needs a legal centring step, never an uphill grid edge.
    corridor=jnp.full_like(d,INF).at[:,3,3].set(2).at[:,2,3].set(1)
    target,valid=action_set(jnp.array([[3.5,3.85]]),corridor,legal.at[:,0].set(False),danger,wall,jnp.zeros_like(wall))
    assert bool(valid[0]) and float(target[0,2])==1

def test_real_pickup_gate_and_boundaries():
    from scripts.je09_return_expert import gate
    from scripts.je09_bun_bridge import jax_bridge_state,jax_bridge_reward
    ledger=jax_bridge_state(1)
    z=jnp.zeros((1,2),bool);one=jnp.ones((1,2),bool)
    info=dict(drop=one,pickup_loose=z,before_distance=jnp.full((1,2),20),after_distance=jnp.full((1,2),20),carried=z,topology_changed=z,home_half=z,lost=z,settled=z,capture=z,death=z,self=z,trade=z,timeout=z)
    st=SimpleNamespace(scene=jnp.array([9]),alive=one,bun_carried=jnp.array([[1,0]]),trapped=jnp.zeros((1,2),int))
    aux=dict(bridge=ledger,terminal=jnp.array([False]));seat=jnp.array([0])
    assert not bool(gate(st,aux,seat,120).any())
    _,drop,_=jax_bridge_reward(info,ledger,window=120,scale=2)
    assert not bool(gate(st,dict(aux,bridge=drop),seat,120).any())
    _,picked,_=jax_bridge_reward(dict(info,drop=z,pickup_loose=one,carried=one),drop,window=120,scale=2)
    for ticks,expected in [(0,True),(119,True),(120,False),(121,False)]:
        assert bool(gate(st,dict(aux,bridge=dict(picked,ticks=jnp.full((1,2),ticks))),seat,120)[0])==expected
    # Unattributed pickup, direct carry init, combat replay, loss and terminal excluded.
    _,unattributed,_=jax_bridge_reward(dict(info,drop=z,pickup_loose=one,carried=one),ledger)
    assert not bool(gate(st,dict(aux,bridge=unattributed),seat,120).any())
    for field,value in [('scene',jnp.array([0])),('alive',z),('bun_carried',jnp.full((1,2),-1)),('trapped',jnp.ones((1,2),int))]:
        changed=SimpleNamespace(**dict(vars(st),**{field:value}))
        assert not bool(gate(changed,dict(aux,bridge=picked),seat,120).any())
    assert not bool(gate(st,dict(aux,bridge=picked,terminal=jnp.array([True])),seat,120).any())

def test_set_loss_exact_excluded_gradients_and_equal_targets():
    from scripts.je09_return_expert import supervised_loss,gate
    from scripts.je09_bun_bridge import jax_bridge_state
    q=jnp.zeros((5,5));mask=jnp.ones_like(q,bool).at[:,4].set(False)
    targets=jnp.tile(jnp.array([.5,0,.5,0,0]),(5,1))
    b=jax_bridge_state(5);b.update(active=jnp.ones((5,2),bool),paid=jnp.ones((5,2,7),bool),ticks=jnp.zeros((5,2),int).at[1].set(120))
    st=SimpleNamespace(scene=jnp.array([9,9,9,9,0]),alive=jnp.ones((5,2),bool),bun_carried=jnp.tile(jnp.array([1,0]),(5,1)).at[2,0].set(-1),trapped=jnp.zeros((5,2),int))
    active=gate(st,dict(bridge=b,terminal=jnp.array([False,False,False,True,False])),jnp.zeros(5,int),120)
    np.testing.assert_array_equal(active,jnp.array([True,False,False,False,False]))
    grad=jax.grad(lambda p:supervised_loss(p,mask,targets,active,.05,.5)[0])(q)
    assert grad[0,0]==grad[0,2] and grad[0,0]<0
    np.testing.assert_array_equal(grad[1:],0);np.testing.assert_array_equal(grad[:,4],0)
    np.testing.assert_array_equal(jax.grad(lambda p:supervised_loss(p,mask,targets,active,0.,.5)[0])(q),0)
    np.testing.assert_array_equal(jax.grad(lambda p:supervised_loss(p,mask,targets,jnp.zeros(5,bool),.05,.5)[0])(q),0)
