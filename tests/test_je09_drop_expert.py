"""Drop-active supervision: legal geometry, provenance, scope and gradients."""
import jax
import jax.numpy as jnp
import numpy as np


def test_shortest_teacher_detours_brick_and_never_unreachable():
    from scripts.je09_drop_expert import direction
    from scripts.je08_bun_potential import shortest, INF
    wall=jnp.zeros((1,5,7),bool).at[:,0].set(True).at[:,-1].set(True).at[:,:,0].set(True).at[:,:,-1].set(True)
    brick=jnp.zeros_like(wall).at[:,2,3].set(True)
    goals=jnp.zeros_like(wall).at[:,2,4].set(True)
    d=shortest(wall|brick,jnp.zeros_like(brick),goals)
    mm=jnp.ones((1,5),bool)
    move,valid=direction(jnp.array([[2.5,2.5]]),d,mm)
    assert bool(valid[0]);assert int(move[0]) in (0,1)
    move,valid=direction(jnp.array([[2.5,2.5]]),jnp.full_like(d,INF),mm)
    assert not bool(valid[0])
    move,valid=direction(jnp.array([[2.5,2.5]]),d,jnp.zeros_like(mm).at[:,4].set(True))
    assert not bool(valid[0])


def test_loss_scope_zero_and_temperature():
    from scripts.je09_drop_expert import supervised_loss
    logits=jnp.zeros((3,5));mm=jnp.ones((3,5),bool).at[:,3].set(False)
    target=jnp.array([1,0,2]);active=jnp.array([True,False,False])
    f=lambda q:supervised_loss(q,mm,target,active,.2,.5)[0]
    g=jax.grad(f)(logits)
    assert float(g[0,1])<0
    np.testing.assert_array_equal(g[1:],0);np.testing.assert_array_equal(g[:,3],0)
    z=jax.grad(lambda q:supervised_loss(q,mm,target,active,0.,1.)[0])(logits)
    np.testing.assert_array_equal(z,0)
    z=jax.grad(lambda q:supervised_loss(q,mm,target,jnp.zeros(3,bool),1.,1.)[0])(logits)
    np.testing.assert_array_equal(z,0)


def test_native_provenance_reset_carrier_and_snapshot():
    from tests.test_je09_fixture_reward import native_fixture
    from scripts import je09_bun_ops as ops,je09_drop_expert as expert
    from jax_bomb import jax_train as train
    from scripts import je09_bun_scenes as scenes
    scenes.bind_training()
    before,after,info=native_fixture()
    _,aux,_,_=jax.jit(ops.combined_reward)(before,after,info,ops.reward_state(4),jnp.ones(4,bool))
    mm,_=train.both_masks(after)
    seat=jnp.arange(4)%2;idx=jnp.arange(4)+seat*4
    labels,active=jax.jit(expert.labels)(after,aux,mm[idx],seat)
    assert int(active.sum())==4
    for key in ('credit','nonce','cycle_paid'):
        altered=dict(aux);altered['drop']=dict(aux['drop'])
        altered['drop'][key]=jnp.full_like(aux['drop'][key],-1 if key=='credit' else (1 if key=='cycle_paid' else 0))
        assert not bool(expert.labels(after,altered,mm[idx],seat)[1].any())
    altered=dict(aux,potential=dict(aux['potential']))
    altered['potential']['drop']=dict(aux['potential']['drop'],active=jnp.zeros_like(aux['potential']['drop']['active']))
    assert not bool(expert.labels(after,altered,mm[idx],seat)[1].any())
    terminal=dict(aux,terminal=jnp.ones(4,bool))
    assert not bool(expert.labels(after,terminal,mm[idx],seat)[1].any())
    mismatch=dict(aux,routing=dict(aux['routing'],nonce=aux['routing']['nonce']+1))
    assert not bool(expert.labels(after,mismatch,mm[idx],seat)[1].any())
    carrier=after._replace(bun_carried=after.bun_carried.at[jnp.arange(4),seat].set(1-seat))
    dead=after._replace(core=after.core._replace(alive=jnp.zeros_like(after.alive)))
    for st in (carrier,dead):assert not bool(expert.labels(st,aux,mm[idx],seat)[1].any())
    assert not bool(expert.labels(after,ops.reward_state(4),mm[idx],seat)[1].any())
    import pickle
    restored=pickle.loads(pickle.dumps(jax.device_get(aux)))
    for a,b in zip(expert.labels(after,aux,mm[idx],seat),expert.labels(after,restored,mm[idx],seat)):np.testing.assert_array_equal(a,b)


def test_local_safe_gate_allows_remote_bomb_but_rejects_local_danger():
    from tests.test_je09_fixture_reward import native_fixture
    from scripts import je09_bun_ops as ops,je09_drop_expert as expert
    from jax_bomb import jax_train as train,je07_bun_env as env
    from scripts import je09_bun_scenes as scenes
    scenes.bind_training();before,after,info=native_fixture()
    _,aux,_,_=jax.jit(ops.combined_reward)(before,after,info,ops.reward_state(4),jnp.ones(4,bool))
    rows=jnp.arange(4);seat=rows%2;idx=rows+seat*4
    cells=jnp.asarray(np.indices((env.H,env.W)).transpose(1,2,0))
    separation=jnp.min(jnp.sum(jnp.abs(cells[None,None]+.5-after.pos[:,:,None,None]),-1),axis=1)
    remote=jnp.argmax(jnp.where(~after.wall,separation,-1).reshape(4,-1),-1);r,c=remote//env.W,remote%env.W
    changed=after._replace(core=after.core._replace(fuse=after.fuse.at[rows,r,c].set(5),bomb_blast=after.core.bomb_blast.at[rows,r,c].set(1)))
    mm,_=train.both_masks(changed)
    danger=jax.vmap(lambda s:env.base._danger_map(s.fuse,s.wall,s.core.bomb_blast,brick=s.brick))(changed)
    move,active=expert.labels(changed,aux,mm[idx],seat,danger,'local_safe')
    assert int(active.sum())==4
    assert not bool(expert.labels(changed,aux,mm[idx],seat,danger,'global')[1].any())
    # Current-cell danger (including lingering blast) must always suppress.
    cell=changed.pos[rows,seat].astype(jnp.int32)
    neighbor=jnp.clip(cell+jnp.array([[-1,0],[1,0],[0,-1],[0,1],[0,0]])[move],jnp.array([0,0]),jnp.array([env.H-1,env.W-1]))
    next_danger=danger.at[rows,neighbor[:,0],neighbor[:,1]].set(1.)
    assert not bool(expert.labels(changed,aux,mm[idx],seat,next_danger,'local_safe')[1].any())
    danger=danger.at[rows,cell[:,0],cell[:,1]].set(1.)
    assert not bool(expert.labels(changed,aux,mm[idx],seat,danger,'local_safe')[1].any())
