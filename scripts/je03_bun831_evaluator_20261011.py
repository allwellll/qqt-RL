"""Durable CPU authority for Bun831 tracks, isolated from GPU dispatch."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import je03_bun831_20261011 as candidate
from scripts import je03_bun_evaluator_20261011 as authority
from scripts.population_run_state import atomic_json,sha,lock
from scripts.bun_checkpoint_state import cpu_command
from scripts.je03_bun_stage1_20261011 import read,verify_checkpoint

RUN=candidate.RUN;OUT=RUN/'authority_bun831_v1'
TASKS=['carry_home','near_steal','route_break','full']


def task_evaluate(policy,target):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax_bomb import bun_env as env,jax_train as train
    assert jax.default_backend()=='cpu' and train.IS_BUN and os.environ['CUDA_VISIBLE_DEVICES']==''
    env.prepare()
    with Path(policy).open('rb') as f:params=jax.tree.map(jnp.asarray,pickle.load(f))
    results={};begin=time.time()
    for ordinal,lesson in enumerate(TASKS):
        env.configure_training(lesson+'=1',1,reward_profile='legacy')
        key=jax.random.PRNGKey(2026101131+ordinal);states=env.init_batch(key,16)
        active=jnp.ones(16,jnp.bool_)
        @jax.jit
        def step(states,key,active):
            key,kact,kstep=jax.random.split(key,3)
            masks=train.both_masks(states)
            actions,_,_=train.sample_actions(params,'transformer',train.both_perspectives(states)[:16],
                tuple(x[:16] for x in masks),kact,state=train.both_states(states)[:16])
            opponent=jnp.tile(jnp.array([[4,0]],jnp.int32),(16,1))
            joint=jnp.stack([actions,opponent],1)
            next_states,done,info=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(states,joint,jax.random.split(kstep,16))
            next_states=jax.tree.map(lambda n,o:jnp.where(active.reshape((-1,)+(1,)*(n.ndim-1)),n,o),next_states,states)
            values=jnp.stack([info[k][:,0].astype(jnp.int32) for k in ['capture','steal','return_bun','death','own_bomb_defeat']],-1)
            used=(actions[:,1]==2).astype(jnp.int32)
            values=jnp.concatenate([values,used[:,None]],-1)*active[:,None]
            return next_states,key,active&~done,values
        totals=np.zeros((16,6),np.int32)
        for _ in range(300):
            states,key,active,values=step(states,key,active);totals+=np.asarray(values)
        results[lesson]=dict(games=16,capture_episodes=int((totals[:,0]>0).sum()),steal_episodes=int((totals[:,1]>0).sum()),
            return_episodes=int((totals[:,2]>0).sum()),death_episodes=int((totals[:,3]>0).sum()),
            self_episodes=int((totals[:,4]>0).sum()),item_use=int(totals[:,5].sum()),episodes=totals.tolist())
    atomic_json(target,dict(passed=True,policy_sha256=sha(policy),seed_base=2026101131,primary_ticks=300,
        opponent='frozen idle; task diagnostic, never Hunter substitute',native_bun=True,cuda_visible_devices='',
        results=results,wall_seconds=time.time()-begin))


def evaluate(label,policy,proof):
    out=OUT/label; doc=authority.evaluate(label,policy,proof)
    final=out/'COMPLETE_WITH_BUN_TASKS.json'
    if final.exists():
        result=read(final)
        assert result['authority_sha256']==sha(out/'COMPLETE.json')
        assert all(sha(out/k)==h for k,h in result['additional_shards'].items());return result
    model=out/'model/model.json'; policy_hash=sha(policy);target=out/'bun_tasks.json';start=time.time()
    def probe(difficulty):
        path=out/f'native_probe_{difficulty}.json'
        if not path.exists():
            subprocess.run(['node',str(ROOT/'scripts/eval_web_native_bun_probe.js'),'--model',str(model),
                '--difficulty',difficulty,'--seed','392010731','--checkpoint-sha',policy_hash,'--out',str(path)],check=True)
        data=read(path)
        assert data['checkpoint_sha256']==policy_hash and data['native_items'] and data['native_trap'] and data['real_web_hunter']
        assert len(data['episodes'])==2 and all(not e['hazards_remaining'] for e in data['episodes'])
        return path.name,sha(path)
    if not target.exists():
        with (out/'bun_tasks.log').open('a') as log:
            child=subprocess.Popen(cpu_command([sys.executable,str(Path(__file__).resolve()),'tasks',
                '--policy',str(policy),'--out',str(target)], 'BUN_TASK_CPUS'),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:shards=dict(pool.map(probe,['hard','normal','easy']))
            assert child.wait()==0,'native task evaluator failed; GPU unaffected'
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:shards=dict(pool.map(probe,['hard','normal','easy']))
    task=read(target);assert task['policy_sha256']==policy_hash and task['native_bun'] and task['cuda_visible_devices']==''
    shards['bun_tasks.json']=sha(target)
    result=dict(passed=True,policy_sha256=policy_hash,authority_sha256=sha(out/'COMPLETE.json'),
        additional_shards=shards,tasks=task['results'],task_wall_seconds=time.time()-start,
        stage1_passed=False,stage2_enabled=False,limitation='native1v1+rescue probes stress safety; Bun24 has unobserved teammate/trap state, no claim complete product certification')
    atomic_json(final,result);return result


def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    authority.OUT=OUT;authority.RUN=RUN;authority.POINTS=candidate.POINTS
    protocol=dict(namespace=OUT.name,paired=True,hunter_games_per_tier_per_arm=64,tactical_games_per_arm=64,
        hunter_primary=300,hunter_residual=64,tactical_primary=300,spawns_sha256=sha(authority.SPAWNS),
        seed=read(authority.SPAWNS)['seed'],checkpoints=candidate.POINTS,native_bun_tasks=TASKS,
        native_task_games=16,native_task_seed_base=2026101131,native_task_ticks=300,
        native_web_safety_seed=392010731,cuda_visible_devices='',stage2_enabled=False,
        sources={f:sha(ROOT/f) for f in authority.FILES+['scripts/je03_bun831_evaluator_20261011.py','scripts/eval_web_native_bun_probe.js']})
    if (OUT/'PROTOCOL.json').exists():assert read(OUT/'PROTOCOL.json')==protocol
    else:atomic_json(OUT/'PROTOCOL.json',protocol)
    if not (OUT/'spawns.json').exists():atomic_json(OUT/'spawns.json',read(authority.SPAWNS))


def owner():
    setup();own=lock(OUT/'owner.lock');atomic_json(OUT/'OWNER.json',dict(pid=os.getpid(),cuda_visible_devices='',unix=time.time()))
    while True:
        jobs=[('original_bun_it4000',candidate.BUN_SOURCE,dict(reference_sha256=sha(candidate.BUN_SOURCE)))]
        for g in candidate.POINTS:
            for arm in candidate.ARMS:
                cp=RUN/'formal'/arm/f'gen{g:06d}'
                if (cp/'COMMITTED.json').exists():
                    m=verify_checkpoint(cp,sha(RUN/'plan.json'))
                    jobs.append((f'{arm}_g{g:06d}',cp/'policy.pt',dict(generation=g,committed_sha256=sha(cp/'COMMITTED.json'))))
        complete=[];errors=[]
        for label,policy,proof in jobs:
            try:evaluate(label,policy,proof);complete.append(label)
            except Exception as e:
                target=OUT/f'TECHNICAL_STOP_{label}_{time.time_ns()}.json'
                atomic_json(target,dict(error=repr(e),unix=time.time()));errors.append(label)
        atomic_json(OUT/'STATE.json',dict(pid=os.getpid(),completed=complete,errors=errors,phase='CPU_ASYNC',unix=time.time()))
        for track in ['teaching','capacity_initialization']:
            arms=read(RUN/'plan.json')['tracks'][track]['arms']
            for g in candidate.POINTS:
                cells=[OUT/f'{a}_g{g:06d}/COMPLETE_WITH_BUN_TASKS.json' for a in arms]
                if all(c.exists() for c in cells):
                    atomic_json(OUT/f'PAIR_{track}_g{g:06d}.json',dict(passed=True,arms=arms,generation=g,
                        complete_hashes=[sha(c) for c in cells],stage2_enabled=False))
        if all((RUN/f'TRAINING_COMPLETE_{a}.json').exists() for a in candidate.ARMS) and len(complete)==1+len(candidate.ARMS)*len(candidate.POINTS):
            atomic_json(OUT/'EVALUATION_COMPLETE.json',dict(passed=True,completed=complete,unix=time.time()))
            atomic_json(RUN/'AB_COMPLETE.json',dict(passed=True,technical_only=True,stage1_passed=False,stage2_enabled=False,unix=time.time()));return
        time.sleep(5)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['owner','tasks']);ap.add_argument('--policy');ap.add_argument('--out');args=ap.parse_args()
    if args.command=='tasks':task_evaluate(args.policy,args.out)
    else:owner()
