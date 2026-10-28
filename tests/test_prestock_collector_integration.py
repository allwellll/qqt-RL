"""Isolated CPU wiring check with the real native collector and split network.

Run this file alone: candidate production modules are patched in process only,
then restored. No repository source or running numerical worker is modified.
"""
import os
os.environ.setdefault('JAXBOMB_RULE','bun')
import importlib
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np


def test_real_collector_shared_potential_joint_labels_reset_and_replay():
    from scripts.prestock_recipe import patched_sources
    from scripts import je10_ops as ops
    from scripts import je09_stage_h5c_dagger as dagger
    from jax_bomb import jax_train as train,je07_bun_env as env
    from scripts.je03_bun_stage1_20261011 import split_forward
    from tests.test_prestock_training_state import native
    root=Path(__file__).resolve().parents[1]
    files=['scripts/je09_stage_h5c_ops.py','scripts/je09_stage_h6_ops.py','scripts/je09_route_pbrs.py','scripts/je10_ops.py','scripts/je10_worker.py']
    sources=patched_sources({f:(root/f).read_text() for f in files})
    modules={f:importlib.import_module(f[:-3].replace('/','.')) for f in files}
    backup={f:dict(m.__dict__) for f,m in modules.items()};dagger_backup=dict(dagger.__dict__);forward=train.net_forward
    try:
        for f in files:
            if f!='scripts/je10_worker.py':exec(compile(sources[f],f,'exec'),modules[f].__dict__)
        ops.install();train.net_forward=lambda p,a,o,s=None:split_forward(p,a,o,s,forward)
        from scripts import je09_bun_scenes as scenes
        scenes.bind_training();env._DANGER_TRACKING=True
        # Both native seats, different side/distance to the same real barrier.
        samples=[]
        for seat in [0,1,0,1]:
            st,_=native(seat);position=jnp.array([5.5,10.5 if seat==0 else 4.5])
            st=st._replace(core=st.core._replace(pos=st.pos.at[0,seat].set(position)))
            samples.append(jax.tree.map(lambda x:x[0],st))
        states=jax.tree.map(lambda *x:jnp.stack(x),*samples);n=4
        states=states._replace(core=states.core._replace(t=states.t.at[0].set(scenes.HORIZONS[8]-1)))
        actor=train.init_net(jax.random.PRNGKey(42),'transformer',24,13,15,embed=16,depth=1)
        actor['heads']['wb']=(actor['heads']['wb'][0],jnp.array([1000.,-1000.,-1000.]))
        critic=train.init_net(jax.random.PRNGKey(43),'transformer',24,13,15,embed=16,depth=1)
        params=dict(actor=actor,critic=critic);aux=ops.reward_state(n)
        pool=jax.tree.map(lambda x:jnp.stack([x]*11),actor)
        frozen=dict(g0=actor,stage1_teacher=actor,parent=actor)
        course=ops.curriculum_weights(jnp.zeros((10,len(ops.FUNNEL))),.35,.20)
        roll=jax.jit(lambda st,a,k:ops.collect(params,st,a,k,pool,frozen,course,jnp.ones(11)/11,
            2,assist_probability=0.,dagger_round=1,chase_stage=jnp.int32(0),return_config=dict(window=120)))
        states,aux,key,batch,stats=roll(states,aux,jax.random.PRNGKey(44));jax.block_until_ready(batch)
        ctx=np.asarray(batch[-1]);assert ctx.shape==(2,n,125)
        preparing=ctx[...,110]>0;assert preparing.any()
        assert np.isin(ctx[...,84][preparing],[5,6]).all()
        assert ctx[...,95:110].sum()>0
        assert ctx[0,0,47]>0 and ctx[0,0,83]==0  # terminal reset and PBRS zero endpoint
        assert int(states.t[0])<int(scenes.HORIZONS[8]-1)
        assert np.isfinite(np.asarray(batch[4])).all()
        assert 'prestock' in aux
        from scripts.prestock_joint_replay import buffer,aggregate
        old=buffer(capacity=4,additions=2)
        updated,replay,stats=jax.jit(aggregate)(old,batch,key,jnp.int32(1))
        assert int(stats['noneligible_label'])==0
        import optax
        optimizer=optax.chain(optax.clip_by_global_norm(.5),optax.adam(1e-4))
        step=jax.jit(lambda q,a:ops.ppo(q,a,batch,key,q,optimizer,frozen,8,axis_name=None,
            expert_coefficient=.1,return_coefficient=.05,return_temperature=.5,dagger_replay=replay))
        trained,_,_,loss=step(params,optimizer.init(params));jax.block_until_ready(trained)
        assert np.isfinite(np.asarray(loss)).all()
        for kind in ['actor','critic']:
            norm=sum(float(jnp.sum((x-y)**2)) for x,y in zip(jax.tree.leaves(trained[kind]),jax.tree.leaves(params[kind])))
            assert norm>0
        # Full native auxiliary state and extended FIFO survive host save/restore.
        import pickle
        saved=pickle.loads(pickle.dumps(jax.device_get((states,aux,key,updated))))
        restored=jax.tree.map(jnp.asarray,saved)
        expected=roll(states,aux,key);actual=roll(*restored[:3])
        for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):np.testing.assert_array_equal(a,b)
    finally:
        train.net_forward=forward;dagger.__dict__.clear();dagger.__dict__.update(dagger_backup)
        for f,m in modules.items():m.__dict__.clear();m.__dict__.update(backup[f])
