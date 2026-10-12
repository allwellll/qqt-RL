"""Native Bun it4000 continuation, shared versus independent Critic.

Four-device PPO uses the existing Bun rollout/reward/action APIs. The opponent
is frozen it4000 plus fixed combat rules; only learner P0 transitions enter PPO.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.population_run_state import atomic_json,sha,lock

RUN=ROOT/'runs/je03_bun_stage1_20261011/bun_it4000_split_v1'
SOURCE=Path(os.environ.get('BUN_IT4000_CHECKPOINT', ROOT/'runs/overnight_tf_v2/phase2_it4000.pt'))
SOURCE_SHA='6e190b009f180f2e449fd19941c930c0d1a86d680071eeb02262aadb6808ebd0'
POINTS=[0,32,64,128,192,256,512,768,1024,1536,2048,3072,4096,5120,6144,7168,8192]


def read(path):return json.loads(Path(path).read_text())


def plan():
    return dict(experiment_id='JE03-BUN-IT4000-SPLIT-20261011-v1',stage=1,stage2_enabled=False,
        initialization=dict(source=str(SOURCE),sha256=SOURCE_SHA,adaptation='none; original Bun192 weights, no battle migration',optimizer='fresh matched Adam'),
        architecture=dict(embed=192,depth=4,heads=4,patch=3,obs=[24,13,15],state=24,move_actions=5,ability_actions=3,value_bins=128),
        variable='shared versus independent Critic only',arms=dict(control='original shared actor-critic',treatment='same Actor; independent identical-initialized Critic'),
        devices=dict(control='0,1,2,3',treatment='4,5,6,7'),seed=2026101111,generations=8192,
        envs_per_device=1024,steps=128,minibatch_per_device=4096,epochs=2,adv_top_frac=1.,
        ticks_per_generation=524288,ticks_per_arm=4294967296,checkpoints=POINTS,recovery_every=128,
        optimizer=dict(name='Adam',lr=.0003,eps=1e-8),gamma=.995,lam=.95,entropy=.01,clip=.2,value_coef=.5,ema_decay=.995,
        environment=dict(rule='bun',map=806,hp=1,action_repeat=1,inputs='native Bun24 spatial +24 global; no masking or battle adapter',
            curriculum='danger_arena=0.35,full_ambush=0.25,full=0.2,carry_home=0.2',reward_profile='combat_evolution',
            historical_recipe='runs/overnight_tf_v2/phase2_it4000.json',horizon='existing lesson-specific limits, full2400'),
        opponents=dict(dynamic=False,refresh=False,frozen_it4000=.7,rule_combat=.3,learner_seat='P0; P1 excluded from both Actor and Critic loss'),
        evaluation=dict(primary='real web Hunter hard/normal/easy,64 paired games each,300+64 residual',
            tactical='frozen Python Tactical v2,64 paired games,300 ticks; separate results',
            reference='same-protocol original Bun it4000/gen0, never non-Bun it831',
            promotion='stable Hunter safe kills above original Bun it4000, per-tier and death/self/trade noninferiority; independent confirmation before any stage2',
            cpu_async=True,cuda_disabled=True),
        no_cutoff=True,no_dynamic_selfplay=True,scope='Stage1 Bun reconstruction; common frozen teaching opponents replace historical dynamic selfplay; no claim exact historical selfplay replay')


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    p=plan();assert sha(SOURCE)==SOURCE_SHA
    if (RUN/'plan.json').exists():
        assert read(RUN/'plan.json')==p
        return
    with SOURCE.open('rb') as f:params=pickle.load(f)
    assert params['tok'][0].shape==(216,192) and params['state_w'].shape==(24,192)
    assert len(params['blocks'])==4 and params['heads']['wb'][0].shape==(192,3)
    (RUN/'inputs').mkdir(exist_ok=True)
    with (RUN/'inputs/gen0.pkl').open('wb') as f:pickle.dump(params,f);f.flush();os.fsync(f.fileno())
    atomic_json(RUN/'plan.json',p)
    paths=['scripts/je03_bun_stage1_20261011.py','jax_bomb/jax_train.py','jax_bomb/jax_net.py','jax_bomb/bun_env.py',
           'scripts/je03_native_vit_export_20261011.py','scripts/export_bun_web_model.py',
           'web/sim.js','web/bun_hunter_bot.js','scripts/eval_web_hunter.js','scripts/eval_web_hunter_residual.js']
    atomic_json(RUN/'SOURCE_SEAL.json',dict(sources={s:sha(ROOT/s) for s in paths},plan_sha256=sha(RUN/'plan.json'),gen0_sha256=sha(RUN/'inputs/gen0.pkl')))
    atomic_json(RUN/'READY_CPU.json',dict(passed=True,bun_native=True,stage2_enabled=False,plan_sha256=sha(RUN/'plan.json'),unix=time.time()))


def split_forward(params,arch,obs,state,forward):
    if not isinstance(params,dict) or 'actor' not in params:return forward(params,arch,obs,state)
    move,bomb,_,_=forward(params['actor'],arch,obs,state)
    _,_,value,logits=forward(params['critic'],arch,obs,state)
    return move,bomb,value,logits


def digest(tree):
    import jax
    h=hashlib.sha256();leaves,structure=jax.tree.flatten(tree);h.update(str(structure).encode())
    for x in leaves:
        a=np.asarray(x);h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


def verify_checkpoint(cp,plan_hash):
    cp=Path(cp);m=read(cp/'COMMITTED.json')
    assert m['passed'] and m['plan_sha256']==plan_hash
    assert sha(cp/'state.pkl')==m['state_sha256'] and sha(cp/'policy.pt')==m['policy_sha256']
    return m


def worker(arm,phase,resume=None):
    os.environ['JAXBOMB_RULE']='bun'
    import jax
    import jax.numpy as jnp
    import optax
    from jax_bomb import jax_train as train,bun_env as env
    p=read(RUN/'plan.json');assert train.IS_BUN
    assert len(jax.local_devices())==4 and all(d.platform=='gpu' for d in jax.local_devices())
    env.prepare();env.configure_training(p['environment']['curriculum'],1,reward_profile=p['environment']['reward_profile'])
    forward=train.net_forward
    train.net_forward=lambda q,a,o,s=None:split_forward(q,a,o,s,forward)
    # The existing asymmetric pool supplies learner P0. Evaluate one frozen
    # network rather than three identical weak/old/recent copies per tick.
    def opponent(states,obs,gv,masks,arch,key,kinds,weak,old,recent):
        n=states.pos.shape[0];mm,bm=masks[0][n:],masks[1][n:];kr,kf=jax.random.split(key)
        rules=train.flee_bot_actions(states.pos[:,1],states.pos[:,0],mm,bm,kr,idle_ratio=0.,roam_ratio=0.,pure_flee_ratio=0.)
        acts,_,_=train.sample_actions(old,arch,obs[n:],(mm,bm),kf,state=gv[n:])
        return jnp.where((kinds==2)[:,None],rules,acts)
    train.bun_opponent_actions=opponent
    with (RUN/'inputs/gen0.pkl').open('rb') as f:frozen=jax.tree.map(jnp.asarray,pickle.load(f))
    weights=jnp.asarray(train.parse_bun_opponent_weights('rule_combat=.3,old=.7'))
    n,steps,mb=p['envs_per_device'],p['steps'],p['minibatch_per_device']
    opt=optax.adam(p['optimizer']['lr'],eps=p['optimizer']['eps'])
    def rollout(params,states,key):
        result=train.collect_rollout(params,'transformer',states,key,steps,flee_bot_ratio=0.,
            bun_opponent_weights=weights,bun_opponent_old_params=frozen)
        states,batch=result[:2];obs,st,acts,lps,vals,rew,done,masks=batch
        select=lambda x:x[:,:n]
        obs,st,acts,lps,vals,rew,done=map(select,(obs,st,acts,lps,vals,rew,done))
        masks=tuple(select(x) for x in masks)
        all_masks=train.both_masks(states)
        _,_,tail=train.sample_actions(params,'transformer',train.both_perspectives(states)[:n],
            tuple(x[:n] for x in all_masks),jax.random.split(key)[0],state=train.both_states(states)[:n])
        adv=train.compute_gae(rew,vals,jnp.concatenate([vals[1:],tail[None]]),done,p['gamma'],p['lam'])
        return states,(obs,st,acts,lps,adv,adv+vals,masks),jnp.array([jnp.mean(rew),jnp.mean(done),jnp.sum(result[3])])
    def update(params,state,batch,key,ema):
        params,state,loss=train.ppo_update(params,opt,state,'transformer',batch,key,mb,p['clip'],p['value_coef'],p['entropy'],p['epochs'],
            axis_name='dev',return_loss=True,adv_top_frac=p['adv_top_frac'])
        return params,state,jax.random.split(key)[0],jax.tree.map(lambda e,x:.995*e+.005*x,ema,params),loss
    roll=jax.pmap(rollout,in_axes=(0,0,0));learn=jax.pmap(update,axis_name='dev',in_axes=(0,0,0,0,0))
    rep=lambda q:jax.tree.map(lambda x:jnp.stack([x]*4),q)
    out=RUN/phase/arm;out.mkdir(parents=True,exist_ok=True);own=lock(out/'worker.lock')
    if resume:
        meta=verify_checkpoint(resume,sha(RUN/'plan.json'))
        with (Path(resume)/'state.pkl').open('rb') as f:saved=pickle.load(f)
        live=jax.tree.map(jnp.asarray,saved['live']);generation=saved['generation']
        assert digest(live)==meta['full_state_digest'],'full params/Adam/env/key/EMA restore mismatch'
    else:
        params=frozen if arm=='control' else dict(actor=frozen,critic=jax.tree.map(lambda x:x.copy(),frozen))
        key=jax.random.split(jax.random.PRNGKey(p['seed']),4)
        states=jax.pmap(lambda k:env.init_batch(k,n))(key)
        live=(rep(params),rep(opt.init(params)),states,key,rep(params));generation=0
    atomic_json(out/'WORKER.json',dict(pid=os.getpid(),arm=arm,phase=phase,restored_from=resume,generation=generation,
        full_state_initial_digest=digest(live),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],unix=time.time()))
    stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    def checkpoint(g):
        cp=out/f'gen{g:06d}';cp.mkdir(exist_ok=True)
        saved=dict(generation=g,live=jax.device_get(live),full_state_digest=digest(live))
        if (cp/'COMMITTED.json').exists():
            assert verify_checkpoint(cp,sha(RUN/'plan.json'))['full_state_digest']==saved['full_state_digest'];return cp
        for name,value in [('state.pkl',saved),('policy.pt',jax.tree.map(lambda x:np.asarray(x[0]),live[0]['actor'] if arm=='treatment' else live[0]))]:
            with (cp/(name+'.tmp')).open('wb') as f:pickle.dump(value,f);f.flush();os.fsync(f.fileno())
            os.replace(cp/(name+'.tmp'),cp/name)
        atomic_json(cp/'COMMITTED.json',dict(passed=True,generation=g,fresh_ticks=g*p['ticks_per_generation'],
            state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'),full_state_digest=saved['full_state_digest'],plan_sha256=sha(RUN/'plan.json')))
        return cp
    if not resume:checkpoint(0)
    finish=2 if phase!='formal' else p['generations'];begin=time.time();compute=0
    for g in range(generation+1,finish+1):
        previous=live[0];start=time.time()
        atomic_json(out/'PHASE.json',dict(phase='ROLLOUT_OR_COMPILE',generation=g,pid=os.getpid(),unix=start))
        states,batch,stats=roll(live[0],live[2],live[3]);jax.block_until_ready(batch);roll_seconds=time.time()-start;t=time.time()
        atomic_json(out/'PHASE.json',dict(phase='UPDATE_OR_COMPILE',generation=g,pid=os.getpid(),unix=t))
        params,opt_state,key,ema,loss=learn(live[0],live[1],batch,live[3],live[4]);jax.block_until_ready(params)
        update_seconds=time.time()-t;live=(params,opt_state,states,key,ema);del batch
        dt=time.time()-start;compute+=dt
        arrays=jax.tree.leaves(live[0]);replicas=all(np.array_equal(np.asarray(x)[0],np.asarray(x)[i]) for x in arrays for i in range(1,4))
        finite=all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(live)) and np.isfinite(np.asarray(loss)).all()
        norm=float(np.sqrt(sum(float(jnp.sum((x-y)**2)) for x,y in zip(arrays,jax.tree.leaves(previous)))))
        assert finite and replicas and norm>0,(g,finite,replicas,norm)
        updates=p['epochs']*n*steps//(mb//2)
        row=dict(generation=g,pid=os.getpid(),fresh_ticks=g*p['ticks_per_generation'],finite=bool(finite),replicas_exact=replicas,
            parameter_update_l2=norm,seconds=dt,rollout_seconds=roll_seconds,update_seconds=update_seconds,
            fresh_ticks_per_second=p['ticks_per_generation']/dt,optimizer_updates=updates,optimizer_updates_per_second=updates/dt,
            loss=np.asarray(loss).tolist(),stats=np.asarray(stats).tolist(),unix=time.time())
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
        if phase!='formal' or g in p['checkpoints'] or g%p['recovery_every']==0 or stopping[0]:checkpoint(g)
        atomic_json(out/'STATE.json',dict(row,wall_seconds=time.time()-begin,compute_seconds=compute));print(json.dumps(row),flush=True)
        if stopping[0]:break
    checkpoint(g)
    atomic_json(out/'TRAINER_EXIT.json',dict(passed=True,generation=g,pid=os.getpid(),final_state_digest=digest(live),
        fresh_ticks=g*p['ticks_per_generation'],wall_seconds=time.time()-begin,compute_seconds=compute,stopped_at_update_boundary=stopping[0]))


def gpu_owners():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    ids={r.split(',')[1].strip():int(r.split(',')[0]) for r in rows.splitlines()}
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    result={i:[] for i in range(8)}
    for r in rows.splitlines():
        pid,uuid=r.split(',');result[ids[uuid.strip()]].append(int(pid))
    return result


def publish_pairs(p):
    for g in p['checkpoints']:
        path=RUN/'blocks/screen'/f'CHECKPOINT_PAIR_g{g:06d}.json'
        if path.exists():continue
        cp={a:RUN/'formal'/a/f'gen{g:06d}' for a in p['devices']}
        if all((c/'COMMITTED.json').exists() for c in cp.values()):
            for c in cp.values():verify_checkpoint(c,sha(RUN/'plan.json'))
            atomic_json(path,dict(passed=True,generation=g,checkpoints={a:str(c) for a,c in cp.items()},hashes={a:sha(c/'COMMITTED.json') for a,c in cp.items()},unix=time.time()))


def owner():
    prepare();own=lock(RUN/'owner.lock');p=read(RUN/'plan.json');seal=read(RUN/'SOURCE_SEAL.json')
    assert all(sha(ROOT/f)==h for f,h in seal['sources'].items())
    atomic_json(RUN/'OWNER.json',dict(pid=os.getpid(),unique_writer=True,no_cutoff=True,unix=time.time()))
    children={};stages={a:'WAITING_CARDS' for a in p['devices']};attempts={a:0 for a in p['devices']};logs=[]
    def launch(a,phase,resume=None):
        attempts[a]+=1
        if phase=='formal' and resume is None:
            cps=sorted((RUN/phase/a).glob('gen*/COMMITTED.json'))
            if cps:resume=str(cps[-1].parent)
        if resume and phase=='formal':
            meta=verify_checkpoint(resume,sha(RUN/'plan.json'));ev=RUN/phase/a/'events.jsonl'
            if ev.exists():
                original=ev.read_text();archive=ev.with_name(f'events_before_resume_{time.time_ns()}.jsonl');archive.write_text(original)
                tmp=ev.with_suffix('.tmp');tmp.write_text(''.join(s+'\n' for s in original.splitlines() if json.loads(s)['generation']<=meta['generation']));tmp.replace(ev)
        cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--arm',a,'--phase',phase]
        if resume:cmd+=['--resume',resume]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=p['devices'][a],JAXBOMB_RULE='bun',JAX_PLATFORMS='cuda,cpu',JAX_PLATFORM_NAME='cuda',
            XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        log=(RUN/f'{phase}_{a}_attempt{attempts[a]:03d}.log').open('a');logs.append(log)
        children[a]=subprocess.Popen(['taskset','-c','32-63' if a=='control' else '64-95']+cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        stages[a]=phase.upper()
        atomic_json(RUN/f'{a}_LAUNCH.json',dict(pid=children[a].pid,phase=phase,resume=resume,devices=p['devices'][a],unix=time.time()))
    while True:
        owners=gpu_owners()
        for a in stages:
            stage=stages[a]
            if stage=='COMPLETE':continue
            if stage=='WAITING_CARDS':
                if all(not owners[int(i)] for i in p['devices'][a].split(',')):
                    atomic_json(RUN/f'GPU_GATE_RELEASED_{a}.json',dict(passed=True,owners=owners,assigned_devices=p['devices'][a],reason='user-authorized sequential per-arm handoff after previous atomic retirement',unix=time.time()))
                    if (RUN/f'READY_GPU_{a}.json').exists():launch(a,'formal')
                    else:launch(a,'smoke')
                continue
            child=children[a]
            if child.poll() is None:continue
            atomic_json(RUN/f'{a}_{stage}_EXIT_{child.pid}.json',dict(pid=child.pid,returncode=child.returncode,unix=time.time()))
            if child.returncode!=0:
                atomic_json(RUN/f'TECHNICAL_STOP_{a}_{child.pid}.json',dict(passed=False,phase=stage,returncode=child.returncode,unix=time.time()))
                if stage=='FORMAL' and attempts[a]<8:launch(a,'formal');continue
                raise RuntimeError(f'{a} {stage} failed: inspect preserved log')
            if stage=='SMOKE':launch(a,'replay',str(RUN/'smoke'/a/'gen000001'))
            elif stage=='REPLAY':
                smoke=read(RUN/'smoke'/a/'TRAINER_EXIT.json');replay=read(RUN/'replay'/a/'TRAINER_EXIT.json')
                assert smoke['passed'] and replay['passed'] and smoke['final_state_digest']==replay['final_state_digest']
                rows=[json.loads(s) for s in (RUN/'smoke'/a/'events.jsonl').read_text().splitlines()]
                assert len(rows)==2 and all(r['finite'] and r['replicas_exact'] and r['parameter_update_l2']>0 for r in rows)
                atomic_json(RUN/f'READY_GPU_{a}.json',dict(passed=True,fresh_process_full_state_bitwise=True,smoke=smoke,replay=replay,
                    plan_sha256=sha(RUN/'plan.json'),unix=time.time()))
                launch(a,'formal')
            elif stage=='FORMAL':
                result=read(RUN/'formal'/a/'TRAINER_EXIT.json');assert result['passed'] and result['generation']==p['generations']
                stages[a]='COMPLETE'
        publish_pairs(p)
        atomic_json(RUN/'SUPERVISOR_STATE.json',dict(pid=os.getpid(),stages=stages,children={a:dict(pid=c.pid,returncode=c.poll()) for a,c in children.items()},unix=time.time()))
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
        with (RUN/'GPU_SAMPLING.jsonl').open('a') as f:f.write(json.dumps(dict(stages=stages,unix=time.time(),gpu=gpu))+'\n')
        if all(s=='COMPLETE' for s in stages.values()):
            atomic_json(RUN/'TRAINING_COMPLETE.json',dict(passed=True,arms={a:read(RUN/'formal'/a/'TRAINER_EXIT.json') for a in stages},unix=time.time()))
            atomic_json(RUN/'NO_MORE_CHECKPOINTS.json',dict(passed=True,unix=time.time()));return
        time.sleep(3)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','owner','worker']);parser.add_argument('--arm',choices=['control','treatment']);parser.add_argument('--phase',choices=['smoke','replay','formal']);parser.add_argument('--resume');args=parser.parse_args()
    try:
        if args.command=='worker':worker(args.arm,args.phase,args.resume)
        else:globals()[args.command]()
    except BaseException as e:
        atomic_json(RUN/f'TECHNICAL_STOP_{args.command}_{os.getpid()}.json',dict(passed=False,error=repr(e),traceback=traceback.format_exc(),unix=time.time()))
        raise
