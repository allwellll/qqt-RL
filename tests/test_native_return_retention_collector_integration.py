"""Isolated native acquisition -> collector -> candidate PPO -> restore."""
import os

os.environ.setdefault('JAXBOMB_RULE','bun')


def test_real_transfer_certificate_reaches_candidate_ppo_and_restore(monkeypatch):
    import pickle
    from pathlib import Path
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from jax_bomb import jax_train as train,je07_bun_env as env
    from scripts import je10_ops as ops,je09_bun_scenes as scenes
    from scripts import je09_stage_h5c_ops as base,native_return_retention as retention
    from scripts.je03_bun_stage1_20261011 import split_forward
    from scripts.native_return_retention_recipe import patched

    env.prepare();env.configure_training('full=1',1,retention_replay=0)
    scenes.bind_training();env._DANGER_TRACKING=True;ops.install()
    forward=train.net_forward
    monkeypatch.setattr(train,'net_forward',lambda p,a,o,s=None:split_forward(p,a,o,s,forward))
    n=4;rows=jnp.arange(n);seat=rows%2;enemy=1-seat
    before=scenes.init_batch(jax.random.PRNGKey(901),n,jax.nn.one_hot(9,10))
    assert bool((before.bun_carried[rows,seat]<0).all())
    positions=before.pos.at[rows,seat].set(jnp.array([5.5,4.5]))
    positions=positions.at[rows,enemy].set(jnp.array([5.5,5.3]))
    before=before._replace(core=before.core._replace(pos=positions,
        wall=jnp.zeros_like(before.wall),brick=jnp.zeros_like(before.brick),
        pushable=jnp.zeros_like(before.core.pushable)),
        trapped=before.trapped.at[rows,enemy].set(10))
    actions=jnp.zeros((n,2,2),jnp.int32).at[:,:,0].set(4)
    states,_,info=jax.jit(jax.vmap(lambda st,a,k:env.step(st,a,k,
        auto_reset=False,return_info=True)))(before,actions,jax.random.split(jax.random.PRNGKey(902),n))
    assert bool(info['native_transfer'][rows,seat].all())
    assert bool((states.bun_carried[rows,seat]==enemy).all())
    aux=ops.reward_state(n)
    aux['dagger']=ops.update_custody(before,states,info,aux,aux['dagger'])
    assert bool(aux['dagger']['custody'][rows,seat].all())
    actor=train.init_net(jax.random.PRNGKey(42),'transformer',24,13,15,embed=16,depth=1)
    actor['heads']['wb']=(actor['heads']['wb'][0],jnp.array([1000.,-1000.,-1000.]))
    critic=train.init_net(jax.random.PRNGKey(43),'transformer',24,13,15,embed=16,depth=1)
    params=dict(actor=actor,critic=critic)
    pool=jax.tree.map(lambda x:jnp.stack([x]*11),actor)
    frozen=dict(g0=train.init_net(jax.random.PRNGKey(44),'transformer',24,13,15,embed=16,depth=1),
        stage1_teacher=train.init_net(jax.random.PRNGKey(45),'transformer',24,13,15,embed=16,depth=1),parent=actor)
    course=ops.curriculum_weights(jnp.zeros((10,len(ops.FUNNEL))),.35,.2)
    roll=jax.jit(lambda st,a,k:ops.collect(params,st,a,k,pool,frozen,course,jnp.ones(11)/11,
        2,assist_probability=0.,dagger_round=1,chase_stage=jnp.int32(0),return_config=dict(window=120)))
    states,aux,key,batch,_=roll(states,aux,jax.random.PRNGKey(903));jax.block_until_ready(batch)
    c=batch[-1].reshape(-1,batch[-1].shape[-1]);g=batch[1].reshape(-1,24)
    m=batch[6][0].reshape(-1,5);a=batch[6][1].reshape(-1,3)
    scales=retention.movement_scale(c,g,m,a)
    assert bool((scales==0).any()),'Real acquired native carrier must exercise the candidate'
    assert bool(((c[:,58]>0)&(g[:,14]>.5))[scales==0].all())
    namespace=dict(base.__dict__)
    exec(compile(patched(Path('scripts/je09_stage_h5c_ops.py').read_text()),
        'isolated-native-return-retention','exec'),namespace)
    # The new PPO call site must actually consume the real certificate.
    counts=[];original=retention.movement_scale
    def observed(*args):
        value=original(*args)
        jax.debug.callback(lambda x:counts.append(int(x)),(value==0).sum())
        return value
    monkeypatch.setattr(retention,'movement_scale',observed)
    optimizer=optax.chain(optax.clip_by_global_norm(.5),optax.adam(1e-4))
    update=jax.jit(lambda p,a:namespace['ppo'](p,a,batch,key,p,optimizer,frozen,8,
        axis_name=None,expert_coefficient=.1,return_coefficient=.05))
    trained,adam,ema,loss=update(params,optimizer.init(params));jax.block_until_ready(trained)
    assert counts and sum(counts)>0
    assert np.isfinite(np.asarray(loss)).all()
    for name in ['actor','critic']:
        norm=sum(float(jnp.sum((x-y)**2)) for x,y in zip(jax.tree.leaves(trained[name]),jax.tree.leaves(params[name])))
        assert norm>0
    saved=pickle.loads(pickle.dumps(jax.device_get((trained,adam,ema,states,aux,key))))
    restored=jax.tree.map(jnp.asarray,saved)
    for left,right in zip(jax.tree.leaves((trained,adam,ema,states,aux,key)),jax.tree.leaves(restored)):
        np.testing.assert_array_equal(left,right)
