"""Nonblocking CPU production-loop smoke, complete restore and reward benchmark."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORM_NAME='cpu',JAX_PLATFORMS='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1')
import sys,pickle,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
import jax.numpy as jnp
import optax
from scripts.je05_bun_recipe import RUN,sha,atomic_json,digest,save_pickle
from scripts import je05_bun_ops as ops,je05_bun_scenes as scenes
from jax_bomb import je05_bun_env as env,jax_train as train
from scripts.je03_bun_stage1_20261011 import split_forward

OUT=RUN/'validation'

def restore(file):
    with Path(file).open('rb') as f:s=pickle.load(f)
    full=digest(s['live']);assert full==s['full_digest']
    nextkey=np.asarray(jax.random.split(jnp.asarray(s['live']['key']))[0])
    assert digest(nextkey)==s['next_rng_digest']
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(s['live']))
    atomic_json(OUT/'CPU_FRESH_RESTORE_V4.json',dict(passed=True,full_digest=full,next_rng_digest=digest(nextkey),cpu_only=True,
        fields=sorted(s['live']),transport_highwater=True,not_a_start_gate=True))

def smoke():
    env.prepare();env.configure_training('full=1',1,tactical_bomb_placement_reward=.1);env._DANGER_TRACKING=True;scenes.bind_training()
    original=train.net_forward;train.net_forward=lambda p,a,o,s=None:split_forward(p,a,o,s,original)
    with (RUN/'inputs/initial.pkl').open('rb') as f:params=jax.tree.map(jnp.asarray,pickle.load(f))
    with (RUN/'inputs/frozen.pkl').open('rb') as f:fixed=pickle.load(f)
    frozen=jax.tree.map(jnp.asarray,{k:fixed[k] for k in ['g0','stage1_teacher']})
    from scripts.je05_bun_recipe import POOL
    pool=jax.tree.map(lambda *x:jnp.stack(x),*[params['actor'] if name in ['current','ring0','ring1','ring2','ring3'] else fixed[name] for name in POOL])
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8));adam=opt.init(params)
    key=jax.random.PRNGKey(2026101305);course=ops.curriculum_weights(jnp.zeros((9,len(ops.FUNNEL))))
    # Independent native scenes; real production collect and real split PPO.
    states=scenes.init_batch(key,2,course);aux=ops.reward_state(2);pfsp=ops.pfsp_weights(jnp.zeros((11,6)))
    roll=jax.jit(lambda p,s,a,k:ops.collect(p,s,a,k,pool,frozen,course,pfsp,2))
    begin=time.perf_counter();states,aux,key,batch,stats=roll(params,states,aux,key);jax.block_until_ready(batch);rolltime=time.perf_counter()-begin
    times=[]
    for _ in range(3):
        begin=time.perf_counter();states,aux,key,nextbatch,stats=roll(params,states,aux,key);jax.block_until_ready(nextbatch);times.append(time.perf_counter()-begin)
    atomic_json(OUT/'CPU_PRODUCTION_ROLLOUT_V4.json',dict(cpu_only=True,envs=2,steps=2,production_collect=True,independent_fresh_transitions=True,times=times,median_seconds=float(np.median(times)),fresh_ticks_per_second=4/float(np.median(times)),not_gpu_throughput=True))
    learn=jax.jit(lambda p,a,b,k,e:ops.ppo(p,a,b,k,e,opt,frozen,8,axis_name=None))
    begin=time.perf_counter();updated,adam,ema,loss=learn(params,adam,batch,key,params);jax.block_until_ready(updated);updatetime=time.perf_counter()-begin
    norms={name:float(jnp.sqrt(sum(jnp.sum((x-y)**2) for x,y in zip(jax.tree.leaves(params[name]),jax.tree.leaves(updated[name]))))) for name in ['actor','critic']}
    assert all(x>0 and np.isfinite(x) for x in norms.values()) and np.isfinite(loss).all()
    live=jax.device_get(dict(params=updated,adam=adam,ema=ema,states=states,aux=aux,key=key,pool=pool,course=course,pfsp=pfsp,
                            funnel=jnp.zeros((9,len(ops.FUNNEL))),pool_counts=jnp.zeros((11,6))))
    save_pickle(OUT/'cpu_final_v4_state.pkl',dict(live=live,full_digest=digest(live),next_rng_digest=digest(np.asarray(jax.random.split(key)[0]))))
    atomic_json(OUT/'CPU_NONZERO_UPDATE_V4.json',dict(passed=True,backend='cpu',cuda_visible_devices='',shape=[1,2,2,8],
        production_collect=True,production_ppo=True,finite=True,update_l2=norms,rollout_compile_seconds=rolltime,update_compile_seconds=updatetime,
        checkpoint_sha256=sha(OUT/'cpu_final_v4_state.pkl'),not_a_start_gate=True,loss=np.asarray(loss).tolist()))

def baseline_rollout():
    from scripts import je04_bun_ops as old
    from jax_bomb import bun_env as original_env
    original_env.prepare();original_env.configure_training('danger_arena=1',1)
    for name in ['make_obs','global_vec','legal_mask','step','init_batch']:setattr(train,name,getattr(original_env,name))
    forward=train.net_forward;train.net_forward=lambda p,a,o,s=None:split_forward(p,a,o,s,forward)
    with (RUN/'inputs/initial.pkl').open('rb') as f:params=jax.tree.map(jnp.asarray,pickle.load(f))
    with (RUN/'inputs/frozen.pkl').open('rb') as f:fixed=pickle.load(f)
    frozen=jax.tree.map(jnp.asarray,{k:fixed[k] for k in ['g0','stage1_teacher']})
    from scripts.je05_bun_recipe import POOL
    pool=jax.tree.map(lambda *x:jnp.stack(x),*[params['actor'] if name in ['current','ring0','ring1','ring2','ring3'] else fixed[name] for name in POOL])
    course=old.curriculum_weights(jnp.zeros((16,len(old.FUNNEL))));pfsp=old.pfsp_weights(jnp.zeros((11,6)))
    key=jax.random.PRNGKey(2026101305);states=original_env.init_batch(key,2,course);aux=old.reward_state(2)
    roll=jax.jit(lambda p,s,a,k:old.collect(p,s,a,k,pool,frozen,course,pfsp,2))
    states,aux,key,batch,stats=roll(params,states,aux,key);jax.block_until_ready(batch);times=[]
    for _ in range(3):
        t=time.perf_counter();states,aux,key,batch,stats=roll(params,states,aux,key);jax.block_until_ready(batch);times.append(time.perf_counter()-t)
    atomic_json(OUT/'CPU_JE04_PRODUCTION_ROLLOUT.json',dict(cpu_only=True,shape=[1,2,2],times=times,median_seconds=float(np.median(times)),fresh_ticks_per_second=4/float(np.median(times)),scope='real JE04 collect, own curriculum; identical model/shape/CPU cores; no GPU throughput claim'))


def benchmark():
    # Same vectorization as rollout: tables built once; lookup+ledger per tick.
    from scripts import je04_bun_ops as old
    from tests.test_je04_bun_recipe import info
    env.prepare();env.configure_training('full=1',1)
    st=scenes.init_batch(jax.random.PRNGKey(11),1024,scenes.BASE)
    r={k:jnp.repeat(v,1024,axis=0) for k,v in info().items()}
    for name in ['steal_window_trap','steal_window_dead','carrier_death','pickup_loose']:r[name]=jnp.zeros((1024,2),bool)
    settled=jnp.ones(1024,bool)
    def repeat(f,aux):
        def loop(a,_):
            result=f(a);return result[1],result[0].sum()
        return jax.lax.scan(loop,aux,None,length=128)
    base=jax.jit(lambda st,events:repeat(lambda a:old.retention_reward(events,a,settled),old.reward_state(1024)))
    region=jax.jit(lambda st,events:repeat(lambda a:ops.combined_reward(st,st,events,a,settled),ops.reward_state(1024)))
    results={}
    for name,f in [('je04_reward',base),('je05_lookup_ledger',region)]:
        jax.block_until_ready(f(st,r));times=[]
        for _ in range(5):
            t=time.perf_counter();jax.block_until_ready(f(st,r));times.append(time.perf_counter()-t)
        results[name]=dict(median_seconds=float(np.median(times)),samples=times,ticks_per_second=1024*128/float(np.median(times)))
    atomic_json(OUT/'CPU_REGION_BENCHMARK_V2.json',dict(cpu_only=True,shape=[1024,128],scope='reward hot path; not GPU or full rollout throughput',results=results,
        overhead_seconds=results['je05_lookup_ledger']['median_seconds']-results['je04_reward']['median_seconds']))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['smoke','restore','benchmark','baseline-rollout']);ap.add_argument('--state');a=ap.parse_args()
    if a.command=='restore':restore(a.state)
    elif a.command=='benchmark':benchmark()
    elif a.command=='baseline-rollout':baseline_rollout()
    else:smoke()
