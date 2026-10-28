"""JE09 CPU production collect/GAE/PPO and bridge reward gradient proof."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import optax
import numpy as np
from scripts import je09_bun_ops as ops,je09_bun_scenes as scenes
from scripts.je09_bun_migration import NAMESPACE, STAGE_B, STAGE_C
from scripts.population_run_state import atomic_json
from scripts.je03_bun_stage1_20261011 import digest,split_forward


def main():
    from jax_bomb import je07_bun_env as env,jax_train as train,jax_net as net
    env.prepare();env.configure_training('full=1',1,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=.1)
    env._DANGER_TRACKING=True;scenes.bind_training()
    original=train.net_forward;train.net_forward=lambda q,a,o,s=None:split_forward(q,a,o,s,original)
    cfg=dict(window=120,scale=1.) if STAGE_B else {}
    key=jax.random.PRNGKey(2035000002 if STAGE_C else 2026101663 if STAGE_B else 2026101519)
    actor=net.init_transformer(key,24,13,15,embed=16,depth=1,heads=4,state_dim=24)
    params=dict(actor=actor,critic=actor);frozen=dict(g0=actor,stage1_teacher=actor,parent=actor)
    pool=jax.tree.map(lambda x:jnp.stack([x]*11),actor)
    course=jax.nn.one_hot(9,10);pfsp=jnp.ones(11)/11
    states=scenes.init_batch(key,4,course);aux=ops.reward_state(4)
    roll=jax.jit(lambda s,a:ops.collect(params,s,a,key,pool,frozen,course,pfsp,4,assist_probability=0.,bridge_config=cfg))
    st,a,k,batch,stats=roll(states,aux);jax.block_until_ready(batch)
    assert digest((st,a,k,batch,stats))==digest(roll(states,aux))
    assert bool(a['bridge']['paid'][...,0].any())
    fixture_kills_suppressed=float(stats[2][ops.COMPONENTS.index('fixture_kill_suppressed')])
    assert fixture_kills_suppressed==4
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5))
    update=jax.jit(lambda b:ops.ppo(params,opt.init(params),b,key,params,opt,frozen,8,axis_name=None)[0])
    new=update(batch);jax.block_until_ready(new)
    norms={name:float(optax.global_norm(jax.tree.map(lambda x,y:x-y,new[name],params[name]))) for name in ['actor','critic']}
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(new)) and all(x>0 for x in norms.values())
    # Native movement in a post-pickup phase. Vary only the new bridge ledger;
    # the existing drop/carry rewards and all policy inputs stay identical.
    from tests.test_je07_native_integration import fixture
    first=fixture(dict(pos=[[5.5,6.02],[5.5,9.5]],carry=[1,-1]))
    before=jax.tree.map(lambda x,y:x.at[0].set(y),states,first)
    wait=jnp.tile(jnp.array([[[4,0],[4,0]]]),(4,1,1))
    step=jax.jit(jax.vmap(lambda s,ac,k:env.step(s,ac,k,False,True)))
    initialized,_,initial_info=step(before,wait,jax.random.split(key,4))
    _,aux,_,_=ops.combined_reward(before,initialized,initial_info,ops.reward_state(4),jnp.ones(4,bool),bridge_config=cfg)
    before=initialized
    actions=jnp.tile(jnp.array([[[2,0],[4,0]]]),(4,1,1))
    after,done,info=step(before,actions,jax.random.split(key,4))
    aux['bridge'].update(active=aux['bridge']['active'].at[0,0].set(True),
        paid=aux['bridge']['paid'].at[0,0,:2].set(True),start_distance=aux['bridge']['start_distance'].at[0,0].set(20.))
    reward,_,_,metrics=ops.combined_reward(before,after,info,aux,jnp.ones(4,bool),bridge_config=cfg)
    bridge_delta=metrics['bridge_reward']+metrics['bridge_clawback']
    assert float(bridge_delta[0,0])>0
    obs=train.both_perspectives(before)[:4];gv=train.both_states(before)[:4]
    masks=tuple(x[:4] for x in train.both_masks(before))
    # Four transitions keep advantage normalization meaningful.
    chosen=actions[:,0];move,ability,value,_=train.net_forward(params,'transformer',obs,gv)
    logp=jax.nn.log_softmax(jnp.where(masks[0],move,-1e9))[jnp.arange(4),chosen[:,0]]+jax.nn.log_softmax(jnp.where(masks[1],ability,-1e9))[jnp.arange(4),chosen[:,1]]
    def make(r):
        rewards=reward[:,0].at[0].set(r)[None]
        adv=train.compute_gae(rewards,value[None],value[None],jnp.zeros((1,4),bool),.995,.95)
        return obs[None],gv[None],chosen[None],logp[None],adv,adv+value[None],tuple(x[None] for x in masks),jnp.zeros((1,4,2))
    paid=update(make(reward[0,0]));zero=update(make(reward[0,0]-bridge_delta[0,0]))
    difference=float(optax.global_norm(jax.tree.map(lambda x,y:x-y,paid['actor'],zero['actor'])))
    assert np.isfinite(difference) and difference>0
    out=Path(NAMESPACE)/'validation/CPU_BRIDGE_PRODUCTION.json'
    atomic_json(out,dict(passed=True,cpu_only=True,collect_gae_ppo=True,deterministic=True,
        bridge_drop_consumed=True,fixture_kills_suppressed=fixture_kills_suppressed,
        finite=True,nonzero_update=norms,bridge_reward=float(bridge_delta[0,0]),
        bridge_actor_counterfactual_l2=difference,shape=[1,4,4,8],tiny_fixture=True,stage_b_reward=cfg))
    print(out,flush=True)


if __name__=='__main__':main()
