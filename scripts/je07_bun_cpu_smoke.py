"""Small CPU production-code diagnostic; no formal GPU resource or gate."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import argparse
import pickle
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scripts.je07_bun_recipe import BASE,atomic_json,sha,save_pickle,digest
from scripts import je07_bun_ops as ops,je07_bun_scenes as scenes,je07_bun_curriculum as curriculum


def snapshot_proof(out,live=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if live is None:
        e=dict(drop=jnp.array([False,True]),drop_origin=jnp.array([-1,1]),drop_credit=jnp.array([-1,0]),
            pickup=jnp.array([True,False]),pickup_origin=jnp.array([1,-1]),carried=jnp.array([1,-1]),
            death=jnp.zeros(2,bool),trade=jnp.array(False),capture=jnp.zeros(2,bool),settled=jnp.array(True))
        _,drop,_=curriculum.drop_reward(e,curriculum.drop_state())
        live=dict(aux=dict(drop=drop,assist=curriculum.assist_state(2)),key=jax.random.PRNGKey(991))
    host=jax.device_get(live);cp=out/'cpu_snapshot.pkl';save_pickle(cp,host)
    atomic_json(out/'CPU_SNAPSHOT.json',dict(state_sha256=sha(cp),digest=digest(host),
        next_rng_digest=digest(jax.random.split(jnp.asarray(host['key']))[0]),
        drop_nonce_digest=digest(host['aux']['drop']['nonce']),fields=sorted(host)))
    return cp


def restore(cp):
    cp=Path(cp);proof=__import__('json').loads((cp.parent/'CPU_SNAPSHOT.json').read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and jax.default_backend()=='cpu'
    assert sha(cp)==proof['state_sha256']
    with cp.open('rb') as f:live=pickle.load(f)
    assert digest(live)==proof['digest']
    assert digest(jax.random.split(jnp.asarray(live['key']))[0])==proof['next_rng_digest']
    assert digest(live['aux']['drop']['nonce'])==proof['drop_nonce_digest']
    atomic_json(cp.parent/'CPU_RESTORE.json',dict(passed=True,cpu_only=True,fresh_process_contract=True,
        drop_nonce_exact=True,next_rng_exact=True,complete_fields=proof['fields'],state_sha256=sha(cp),
        scope='CPU tiny production-code state, not formal eight-GPU validation'))

def drop_gradient_proof(params,frozen,opt,train,env):
    """Offline same-transition +8 counterfactual through production PPO."""
    n=4;key=jax.random.PRNGKey(708)
    before=scenes.init_batch(key,n,jax.nn.one_hot(8,10))
    core=before.core._replace(pos=before.pos.at[0,0].set(jnp.array([5.5,5.5])),brick=jnp.zeros_like(before.brick),
        crate=jnp.zeros_like(before.core.crate),invuln=jnp.zeros_like(before.core.invuln))
    before=before._replace(core=core,bun_loose=before.bun_loose.at[0,5,5,1].set(1),
        bun_stored=before.bun_stored.at[0,1,1].set(0))
    actions=jnp.tile(jnp.array([[[4,0],[4,0]]]),(n,1,1))
    after,done,info=jax.jit(jax.vmap(lambda s,a,k:env.step(s,a,k,False,True)))(before,actions,jax.random.split(key,n))
    assert bool(info['pickup_loose'][0,0]) and int(after.bun_carried[0,0])==1
    aux=ops.reward_state(n)
    # The real drop attribution transition is separately exercised by native
    # integration tests; this frozen counterfactual varies that ledger only.
    aux['drop']['nonce']=aux['drop']['nonce'].at[0,1].set(1)
    aux['drop']['credit']=aux['drop']['credit'].at[0,1].set(0)
    reward=ops.combined_reward(before,after,info,aux,jnp.ones(n,bool))[0]
    uncredited=jax.tree.map(lambda x:x,aux)
    uncredited['drop']['credit']=uncredited['drop']['credit'].at[0,1].set(-1)
    baseline=ops.combined_reward(before,after,info,uncredited,jnp.ones(n,bool))[0]
    assert float(reward[0,0]-baseline[0,0])==8.
    obs=train.both_perspectives(before)[:n];gv=train.both_states(before)[:n]
    masks=tuple(x[:n] for x in train.both_masks(before));chosen=actions[:,0]
    move,ability,value,_=train.net_forward(params,'transformer',obs,gv)
    logp=jax.nn.log_softmax(jnp.where(masks[0],move,-1e9))[jnp.arange(n),chosen[:,0]]+jax.nn.log_softmax(jnp.where(masks[1],ability,-1e9))[jnp.arange(n),chosen[:,1]]
    _,_,tail,_=train.net_forward(params,'transformer',train.both_perspectives(after)[:n],train.both_states(after)[:n])
    def batch(r):
        adv=train.compute_gae(r[:,0][None],value[None],tail[None],done[None],.995,.95)
        return obs[None],gv[None],chosen[None],logp[None],adv,adv+value[None],tuple(x[None] for x in masks),jnp.zeros((1,n,2))
    adam=opt.init(params)
    update=jax.jit(lambda b:ops.ppo(params,adam,b,key,params,opt,frozen,8,axis_name=None)[0]['actor'])
    paid=update(batch(reward));zero=update(batch(baseline));jax.block_until_ready(paid)
    difference=float(jnp.sqrt(sum(jnp.sum((x-y)**2) for x,y in zip(jax.tree.leaves(paid),jax.tree.leaves(zero)))))
    assert np.isfinite(difference) and difference>0
    return dict(passed=True,drop_bonus=8,physical_loose_pickup=True,production_gae_ppo=True,
        actor_counterfactual_l2=difference,scope='offline ledger-attribution counterfactual, no formal rollout reuse')


def main(benchmark=False):
    from jax_bomb import je07_bun_env as env,jax_train as train,jax_net as net
    from scripts.je03_bun_stage1_20261011 import split_forward
    assert jax.default_backend()=='cpu'
    env.prepare();env.configure_training('full=1',1,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=.1)
    env._DANGER_TRACKING=True;scenes.bind_training()
    original=train.net_forward;train.net_forward=lambda q,a,o,s=None:split_forward(q,a,o,s,original)
    key=jax.random.PRNGKey(2026101307)
    actor=net.init_transformer(key,24,13,15,embed=16,depth=1,heads=4,state_dim=24)
    params=dict(actor=actor,critic=actor);frozen=dict(g0=actor,stage1_teacher=actor)
    pool=jax.tree.map(lambda x:jnp.stack([x]*11),actor)
    course=scenes.BASE;pfsp=jnp.ones(11)/11
    states=scenes.init_batch(key,2,course);aux=ops.reward_state(2)
    roll=jax.jit(lambda s,a,k:ops.collect(params,s,a,k,pool,frozen,course,pfsp,2))
    if benchmark:
        from scripts import je05_bun_ops as previous,je05_bun_scenes as oldscenes
        from jax_bomb import je05_bun_env as oldenv
        oldenv.prepare();oldenv.configure_training('full=1',1,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=.1)
        oldenv._DANGER_TRACKING=True;oldscenes.bind_training()
        oldcourse=jax.nn.one_hot(8,9);oldstates=oldscenes.init_batch(key,2,oldcourse)
        oldroll=jax.jit(lambda s,a,k:previous.collect(params,s,a,k,pool,frozen,oldcourse,pfsp,2))
        def sample(fn,s,a):
            jax.block_until_ready(fn(s,a,key));times=[]
            for _ in range(9):
                begin=time.time();jax.block_until_ready(fn(s,a,key));times.append(time.time()-begin)
            return dict(seconds=times,median_seconds=float(np.median(times)),fresh_ticks_per_second=float(4/np.median(times)))
        baseline=sample(oldroll,oldstates,previous.reward_state(2))
        scenes.bind_training();course=jax.nn.one_hot(8,10);states=scenes.init_batch(key,2,course)
        roll=jax.jit(lambda s,a,k:ops.collect(params,s,a,k,pool,frozen,course,pfsp,2))
        current=sample(roll,states,ops.reward_state(2))
        atomic_json(BASE/'validation/CPU_ROLLOUT_BENCHMARK_20261014.json',dict(passed=True,cpu_only=True,je06=baseline,je07=current,
            relative_wall_time=current['median_seconds']/baseline['median_seconds'],shape=[2,2],model='tiny16x1',
            scope='same tiny technical fixture, independent fresh rollouts; not a production GPU estimate'))
        return
    begin=time.time();st,aux,k,batch,stats=roll(states,aux,key);jax.block_until_ready(batch)
    compile_wall=time.time()-begin;begin=time.time()
    repeated=roll(states,ops.reward_state(2),key);jax.block_until_ready(repeated)
    assert digest((st,aux,k,batch,stats))==digest(repeated)
    hot_wall=time.time()-begin
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8));adam=opt.init(params)
    learn=jax.jit(lambda p,a,b,k,e:ops.ppo(p,a,b,k,e,opt,frozen,8,axis_name=None))
    new,adam,ema,loss=learn(params,adam,batch,k,params);jax.block_until_ready(new)
    norms={name:float(jnp.sqrt(sum(jnp.sum((x-y)**2) for x,y in zip(jax.tree.leaves(new[name]),jax.tree.leaves(params[name]))))) for name in ['actor','critic']}
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(new)) and all(v>0 for v in norms.values())
    out=BASE/'validation/cpu_production_smoke_20261014';out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'DROP_GRADIENT_CONSUMED.json',drop_gradient_proof(params,frozen,opt,train,env))
    live=dict(params=new,adam=adam,ema=ema,states=st,aux=aux,key=k,pool=pool,course=course,pfsp=pfsp,
        assist_level=jnp.int32(0),assist_proof_hashes=jnp.zeros((3,32),jnp.uint8),funnel=stats[0],pool_counts=stats[1])
    cp=snapshot_proof(out,live)
    subprocess.run([sys.executable,'scripts/je07_bun_cpu_smoke.py','restore','--checkpoint',str(cp)],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    atomic_json(out/'COMPLETE.json',dict(passed=True,cpu_only=True,actual_production_collect_gae_ppo=True,
        finite=True,nonzero_update=norms,fixed_seed_exact=True,compile_seconds=compile_wall,
        hot_rollout_seconds=hot_wall,fresh_ticks=4,fresh_ticks_per_second=4/hot_wall,
        model='tiny16x1 technical fixture; formal remains392x4',shape=[1,2,2,8],fresh_restore_sha256=sha(out/'CPU_RESTORE.json'),
        not_a_start_gate=True,no_gpu_claim=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['smoke','restore','benchmark']);p.add_argument('--checkpoint');args=p.parse_args()
    if args.command=='restore':restore(args.checkpoint)
    else:main(benchmark=args.command=='benchmark')
