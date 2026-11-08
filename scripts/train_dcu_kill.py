"""Pure-kill training on the live v4 JS engine, with synchronous JAX PPO on DCUs.

Modes: plan (no device initialization), smoke (bounded diagnostic), train, eval.
The training budget is wall time from process entry, including compilation and
evaluation. Each DCU has an independent persistent Node simulation worker.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import select
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RULE_FILES=('web/sim-bun-native-v4-facing.js','web/kill_opening_items.js',
    'web/self_respawn_preview.js','web/r3_model_inputs.js','web/assets/maps/levels.json',
    'scripts/kill_mode_bridge.js','qqt_rl/training/kill_ppo.py','scripts/train_dcu_kill.py',
    'scripts/r4_k4_observation.py','scripts/r3_items_observation.py',
    'scripts/retention_phase_conditioning.py','jax_bomb/jax_net.py','qqt_rl/training/dcu.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def arguments(argv=None):
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=('plan','smoke','train','eval'))
    p.add_argument('--devices',default='0,1,2,3,4,5,6,7')
    p.add_argument('--envs',type=int,default=8,help='environments per card / Node worker')
    p.add_argument('--ticks',type=int,default=64)
    p.add_argument('--minibatch',type=int,default=64,help='samples per card per optimizer step')
    p.add_argument('--hours',type=float,default=5)
    p.add_argument('--max-updates',type=int,default=100000)
    p.add_argument('--critic-warmup',type=int,default=10)
    p.add_argument('--save-every',type=int,default=20)
    p.add_argument('--eval-every',type=int,default=1800,help='seconds between evaluations; 0 disables')
    p.add_argument('--eval-ticks',type=int,default=1800)
    p.add_argument('--seed',type=int,default=2026110801)
    p.add_argument('--out',type=Path)
    p.add_argument('--checkpoint',type=Path,help='verified actor+critic checkpoint directory from this trainer')
    p.add_argument('--initial',choices=('DUEL_latest','SP_latest'),default='DUEL_latest')
    p.add_argument('--node',default=str(ROOT/'.tools/node/bin/node'))
    p.add_argument('--acknowledge-training',action='store_true',help='explicit launch of this new bounded round')
    a=p.parse_args(argv)
    try:ids=[int(x) for x in a.devices.split(',')]
    except ValueError:p.error('--devices must be comma-separated integers')
    if not ids or min(ids)<0 or len(set(ids))!=len(ids):p.error('devices must be distinct nonnegative indices')
    a.device_ids=ids
    if a.envs<2 or a.envs%2:p.error('--envs must be positive and even (balanced learner seats)')
    if a.ticks<1 or a.minibatch<1 or a.envs*a.ticks%a.minibatch:p.error('envs*ticks must be divisible by a positive minibatch')
    if not (0<a.hours<=5):p.error('--hours must be in (0,5]')
    if min(a.max_updates,a.save_every,a.eval_ticks)<1 or min(a.critic_warmup,a.eval_every)<0:p.error('invalid update/evaluation budget')
    if a.eval_ticks>1800:p.error('--eval-ticks cannot exceed a full 1800-tick game')
    if not 0<=a.seed<=0xffffffff:p.error('--seed must be uint32')
    if a.mode=='smoke' and (len(ids)!=1 or a.envs>4 or a.ticks>8 or a.max_updates>2):
        p.error('smoke requires one device, <=4 envs, <=8 ticks and --max-updates <=2')
    if a.mode=='train' and not a.acknowledge_training:p.error('train requires --acknowledge-training')
    if a.mode!='plan' and a.out is None:p.error('--out is required')
    if a.out and a.out.exists():p.error('output exists; use a new directory')
    return a


def plan(args):
    return dict(schema='qqt.dcu-kill-plan/v1',devices=args.device_ids,
        envs_per_device=args.envs,global_envs=len(args.device_ids)*args.envs,
        ticks=args.ticks,global_samples=len(args.device_ids)*args.envs*args.ticks,
        minibatch_per_device=args.minibatch,global_minibatch=len(args.device_ids)*args.minibatch,
        wall_seconds=int(args.hours*3600),includes_compilation_and_evaluation=True,
        game_ticks=1800,critic_warmup_updates=args.critic_warmup,
        simulator='live bun-native-v4-facing + opening items + safe respawn',
        gradient_aggregation='pmean before clipping/Adam; globally weighted live policy samples',
        initial_actor=args.initial,checkpoint=str(args.checkpoint) if args.checkpoint else None,
        precise_resume=False,source_sha256={f:sha(ROOT/f) for f in RULE_FILES})


class Bridge:
    def __init__(self,node,seeds,log):
        self.log=open(log,'w')
        self.process=subprocess.Popen([node,str(ROOT/'scripts/kill_mode_bridge.js')],cwd=ROOT,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0)
        self.buffer=b''
        try:self.initial=self.call(dict(command='init',seeds=seeds))
        except BaseException:self.close();raise

    def call(self,request):
        self.process.stdin.write((json.dumps(request)+'\n').encode());self.process.stdin.flush()
        deadline=time.monotonic()+60
        while b'\n' not in self.buffer:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Node simulation response exceeded 60 seconds')
            ready,_,_=select.select([self.process.stdout],[],[],remaining)
            if not ready:raise TimeoutError('Node simulation response exceeded 60 seconds')
            chunk=os.read(self.process.stdout.fileno(),1024*1024)
            if not chunk:raise RuntimeError('Node simulation exited; inspect worker log')
            self.buffer+=chunk
        line,self.buffer=self.buffer.split(b'\n',1)
        row=json.loads(line)
        if 'error' in row:raise RuntimeError(row['error'])
        import numpy as np
        n=row['n']
        for name,shape in [('obs',(n,2,25,13,15)),('state',(n,2,38))]:
            row[name]=np.frombuffer(base64.b64decode(row[name],validate=True),dtype='<f4').reshape(shape).copy()
            if not np.isfinite(row[name]).all():raise RuntimeError('Nonfinite JS observation')
        for name in ('mm','bm','live','rewards','done','metrics'):
            if name in row:row[name]=np.asarray(row[name],np.float32)
        return row

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.log.close()


def require_local_launch(args):
    # New-round acknowledgement is scoped to this command; it does not unlock
    # historical entry points or rewrite the tracked stop policy.
    if args.mode=='train' and not args.acknowledge_training:
        raise RuntimeError('Explicit new-round acknowledgement required')
    private=ROOT/'.hermes/TRAINING_STOP_STATE.json'
    if private.exists() and not json.loads(private.read_text()).get('automatic_resume_allowed',False):
        raise RuntimeError('Persistent private stop active')


def run(args,started):
    import numpy as np
    import jax
    import jax.numpy as j
    from scripts import portable_training as portable
    from scripts.r4_k4_observation import actor_critic_forward
    from qqt_rl.training.kill_ppo import build
    require_local_launch(args)
    devices=jax.local_devices()
    if len(devices)!=len(args.device_ids) or any(d.platform!='gpu' for d in devices):
        raise RuntimeError(f'Expected {len(args.device_ids)} ROCm DCUs, got {devices}')
    node=shutil.which(args.node)
    if node is None:raise RuntimeError('Node executable missing; see docs/dcu_kill_training.md')
    portable.verify_bundle()
    protocol=portable.read(portable.BUNDLE/'protocols/DUEL.json')
    actor=portable.load_actor(protocol,args.initial)
    params=portable.fresh_parameters(actor,args.seed)
    if args.checkpoint:
        meta=json.loads((args.checkpoint/'CHECKPOINT.json').read_text())
        if meta['source_sha256']!=plan(args)['source_sha256']:
            raise RuntimeError('Checkpoint source differs; explicit migration required')
        for branch in ('actor','critic'):
            file=args.checkpoint/(branch+'.pt')
            if sha(file)!=meta[branch+'_sha256']:raise RuntimeError('Checkpoint hash mismatch: '+branch)
            loaded=pickle.loads(file.read_bytes())  # trusted local trainer output only
            if jax.tree.structure(loaded)!=jax.tree.structure(params[branch]):
                raise RuntimeError('Checkpoint tree mismatch: '+branch)
            for actual,expected in zip(jax.tree.leaves(loaded),jax.tree.leaves(params[branch])):
                if actual.shape!=expected.shape or actual.dtype!=expected.dtype or not np.isfinite(actual).all():
                    raise RuntimeError('Checkpoint shape/dtype/finite mismatch: '+branch)
            params[branch]=jax.tree.map(j.asarray,loaded)
    fixed={name:jax.tree.map(j.asarray,portable.load_actor(protocol,name))
        for name in ('DUEL_latest','SP_latest','G_frozen')}
    initial=params['actor']
    count=len(devices)
    init,infer,update=build(actor_critic_forward,devices,args.minibatch)
    finite_check=jax.pmap(lambda q,a:j.all(j.stack([j.all(j.isfinite(x))
        for x in jax.tree.leaves((q,a))])),devices=devices)
    # Every card starts from the exact same model and optimizer state.
    replicate=lambda tree:jax.device_put_replicated(tree,devices)
    adam=replicate(init(params));p=replicate(params);reference=replicate(initial)
    keys=jax.random.split(jax.random.PRNGKey(args.seed),count)
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    config=plan(args);config.update(jax=jax.__version__,device_kinds=[d.device_kind for d in devices],
        packages={name:__import__('importlib.metadata',fromlist=['version']).version(name)
            for name in ('jax','jaxlib','optax','numpy')})
    atomic_json(out/'PLAN.json',config)
    deadline=started+args.hours*3600
    training_deadline=deadline
    stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def stopped():
        if stop[0] or time.monotonic()>=deadline or (out/'USER_STOP').exists():return True
        if args.mode=='train':require_local_launch(args)
        return False
    workers=[]
    last_saved=None;completed=0;total_ticks=0
    history=[initial]
    timings=[]
    pool=ThreadPoolExecutor(max_workers=count)
    def open_worlds(seed):
        result=[]
        try:
            for d in range(count):
                seeds=[(seed+d*100003+e*997)&0xffffffff for e in range(args.envs)]
                result.append(Bridge(node,seeds,out/f'worker-{seed}-{d}.log'))
        except BaseException:
            for worker in result:worker.close()
            raise
        return result
    def unpack(rows):
        # The same balanced seats persist across resets; both physical seats
        # contribute equally. Opponent observations occupy the second half.
        seats=np.arange(args.envs)%2;ids=np.arange(args.envs)
        result={}
        for field in ('obs','state','mm','bm'):
            result[field]=np.stack([np.concatenate([r[field][ids,seats],r[field][ids,1-seats]]) for r in rows])
        result['live']=np.stack([r['live'][ids,seats] for r in rows])
        return result
    def step_worlds(worlds,actions):
        seats=np.arange(args.envs)%2;ids=np.arange(args.envs)
        pair=np.zeros((count,args.envs,2,2),np.int32)
        for d in range(count):
            pair[d,ids,seats]=actions[d,:args.envs];pair[d,ids,1-seats]=actions[d,args.envs:]
        return list(pool.map(lambda item:item[0].call(dict(command='step',actions=item[1].tolist())),zip(worlds,pair)))
    def opponent_bank(episode):
        # 4/8 current-policy selfplay, 2/8 historical own policies, 1/8 SP,
        # 1/8 G. Rotate the schedule between whole 1800-tick games only.
        latest=jax.tree.map(lambda x:x[0],p['actor'])
        choices=[latest]*4+[history[-1-(episode%len(history))]]*2+[fixed['SP_latest'],fixed['G_frozen']]
        actors=[choices[(d+episode)%8] for d in range(count)]
        return jax.device_put_sharded(actors,devices)
    def checkpoint(index):
        nonlocal last_saved
        if last_saved==index:return
        target=out/f'u{index:06d}';target.mkdir(exist_ok=False)
        host=jax.tree.map(lambda x:np.asarray(x[0]),p)
        for branch in ('actor','critic'):
            tmp=target/(branch+'.pt.tmp');tmp.write_bytes(pickle.dumps(host[branch],protocol=5));tmp.replace(target/(branch+'.pt'))
        meta=dict(schema='qqt.kill-actor-critic/v1',update=index,source_sha256=config['source_sha256'],
            actor_sha256=sha(target/'actor.pt'),critic_sha256=sha(target/'critic.pt'),
            global_environment_steps=total_ticks*count*args.envs,
            optimizer_saved=False,precise_resume=False,source_actor=args.initial,
            rule='bun-native-v4-facing-kill-live',mode_row=0)
        atomic_json(target/'CHECKPOINT.json',meta)
        atomic_json(out/'LATEST.json',dict(checkpoint=target.name,**meta));last_saved=index
    def evaluate(label):
        began=time.monotonic();anchors=[]
        # All anchors on every card; same disjoint seeds and sampling keys for
        # baseline/candidates. Evaluation does not consume training RNG state.
        for anchor_id,anchor in enumerate(('DUEL_latest','SP_latest','G_frozen')):
            if stopped():break
            seed=(args.seed^0xE7A10000^anchor_id*0x1000000)&0xffffffff
            eval_keys=jax.random.split(jax.random.PRNGKey(seed),count)
            worlds=open_worlds(seed)
            metrics=np.zeros((count,args.envs,2,6),np.float64)
            rows=[w.initial for w in worlds];steps=0
            opponents=replicate(fixed[anchor])
            try:
                for _ in range(args.eval_ticks):
                    if stopped():break
                    x=unpack(rows)
                    acts,_,_,eval_keys=infer(p,opponents,x['obs'],x['state'],x['mm'],x['bm'],eval_keys,np.full(count,.3,np.float32))
                    rows=step_worlds(worlds,np.asarray(acts));metrics+=np.stack([r['metrics'] for r in rows]);steps+=1
            finally:
                for w in worlds:w.close()
            ids=np.arange(args.envs);seats=ids%2
            own=metrics[:,ids,seats].sum((0,1));enemy=metrics[:,ids,1-seats].sum((0,1))
            anchors.append(dict(opponent=anchor,ticks=steps,full_game=steps==1800,
                completed_games=count*args.envs if steps==1800 else 0,
                learner_metrics=own.tolist(),opponent_metrics=enemy.tolist()))
        row=dict(label=label,anchors=anchors,full_game=len(anchors)==3 and all(a['full_game'] for a in anchors),
            metric_names=['kills','deaths','self_defeats','placed','trapped_ticks','game_score'],
            seeds_namespace='eval',temperature=.3,seconds=time.monotonic()-began)
        atomic_json(out/('EVAL-'+label+'.json'),row)
        print(json.dumps(dict(event='evaluation',**row)),flush=True)
        return row
    try:
        print(json.dumps(dict(event='ready',**config)),flush=True)
        if args.mode=='eval':evaluate('only');return
        final_reserve=0
        if args.mode=='train' and args.eval_every:
            baseline=evaluate('baseline')
            final_reserve=baseline['seconds']*1.5+30
            training_deadline=deadline-final_reserve
        if stopped():return
        workers=open_worlds(args.seed);rows=[w.initial for w in workers]
        opponents=opponent_bank(0)
        next_eval=time.monotonic()+args.eval_every if args.eval_every else float('inf')
        initial_host=jax.device_get(params)
        for index in range(1,args.max_updates+1):
            if stopped() or time.monotonic()>=training_deadline:break
            begin=time.monotonic();samples=[];values=[];rewards=[];dones=[]
            metric=np.zeros((count,args.envs,2,6),np.float64)
            for tick in range(args.ticks):
                if stopped() or time.monotonic()>=training_deadline:break
                if total_ticks and total_ticks%1800==0:
                    history.append(jax.tree.map(lambda x:x[0],p['actor']));history=history[-8:]
                    opponents=opponent_bank(total_ticks//1800)
                x=unpack(rows)
                actions,logp,value,keys=infer(p,opponents,x['obs'],x['state'],x['mm'],x['bm'],keys,np.ones(count,np.float32))
                actions,logp,value=map(np.asarray,(actions,logp,value))
                n=args.envs
                samples.append(dict(obs=x['obs'][:,:n],state=x['state'][:,:n],mm=x['mm'][:,:n],bm=x['bm'][:,:n],
                    live=x['live'],action=actions[:,:n],logp=logp))
                rows=step_worlds(workers,actions)
                seats=np.arange(n)%2;ids=np.arange(n)
                values.append(value);rewards.append(np.stack([r['rewards'][ids,seats] for r in rows]));
                dones.append(np.stack([r['done'] for r in rows]));metric+=np.stack([r['metrics'] for r in rows])
                total_ticks+=1
            if len(samples)!=args.ticks:break
            x=unpack(rows)
            _,_,bootstrap,keys=infer(p,opponents,x['obs'],x['state'],x['mm'],x['bm'],keys,np.ones(count,np.float32))
            # Host GAE is tiny; scan runs on CPU rather than transferring the
            # complete native simulation graph to an accelerator compiler.
            reward=np.stack(rewards);value=np.stack(values);done=np.stack(dones)
            carry=np.zeros_like(np.asarray(bootstrap));adv=np.empty_like(value);nv=np.asarray(bootstrap)
            for t in range(args.ticks-1,-1,-1):
                carry=reward[t]+.997*(1-done[t])*nv-value[t]+.997*.95*(1-done[t])*carry
                adv[t]=carry;nv=value[t]
            batch={k:np.stack([s[k] for s in samples],axis=1) for k in samples[0]}
            batch.update(adv=adv.transpose(1,0,2),returns=(adv+value).transpose(1,0,2))
            batch={k:v.reshape((count,-1)+v.shape[3:]) for k,v in batch.items()}
            keys=jax.vmap(lambda k:jax.random.split(k)[0])(keys)
            warm=index<=args.critic_warmup
            (p,adam),loss=update(p,adam,reference,batch,keys,np.full(count,warm))
            jax.block_until_ready(loss)
            # Check on device to avoid copying large optimizer trees per round.
            finite=finite_check(p,adam)
            if not np.asarray(finite).all() or not np.isfinite(np.asarray(loss)).all():raise RuntimeError('Nonfinite PPO state')
            completed=index;elapsed=time.monotonic()-begin;timings.append(elapsed)
            row=dict(update=index,critic_only=warm,seconds=elapsed,elapsed_seconds=time.monotonic()-started,
                environment_steps=total_ticks*count*args.envs,metrics=metric.sum((0,1)).tolist(),
                loss=np.asarray(loss).mean((0,1)).tolist(),remaining_seconds=max(0,deadline-time.monotonic()))
            with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(event='update',**row)),flush=True)
            if index%args.save_every==0:checkpoint(index)
            if time.monotonic()>=next_eval and time.monotonic()+final_reserve<training_deadline:
                checkpoint(index);evaluate(f'u{index:06d}');next_eval=time.monotonic()+args.eval_every
        checkpoint(completed)
        if args.mode=='train' and args.eval_every and not stopped():evaluate('final')
        final=jax.tree.map(lambda x:np.asarray(x[0]),p)
        delta={b:float(np.sqrt(sum(np.square(x-y).sum() for x,y in zip(jax.tree.leaves(final[b]),jax.tree.leaves(initial_host[b]))))) for b in ('actor','critic')}
        receipt=dict(status='PASS_DCU_KILL_UPDATE' if completed else 'NO_UPDATE',updates=completed,
            parameter_delta_l2=delta,cards=count,environment_steps=total_ticks*count*args.envs,
            elapsed_seconds=time.monotonic()-started,automatic_successor=False,
            median_update_seconds=float(np.median(timings[1:] or timings)) if timings else None)
        atomic_json(out/'RESULT.json',receipt);print(json.dumps(receipt),flush=True)
    finally:
        for w in workers:w.close()
        pool.shutdown(wait=True,cancel_futures=True)
        atomic_json(out/'STOPPED.json',dict(updates=completed,automatic_successor=False,elapsed_seconds=time.monotonic()-started))


def main(argv=None):
    started=time.monotonic();args=arguments(argv);os.chdir(ROOT)
    if args.mode=='plan':print(json.dumps(plan(args),indent=2));return
    require_local_launch(args)
    from qqt_rl.training.dcu import configure
    configure(args.device_ids[0])
    visible=','.join(map(str,args.device_ids))
    os.environ.update(HIP_VISIBLE_DEVICES=visible,CUDA_VISIBLE_DEVICES=visible)
    # Fresh compilation is small (~30s on the validated card). Avoid sharing
    # persistent Hygon compiler executables across device counts/backends.
    os.environ['JAX_ENABLE_COMPILATION_CACHE']='false'
    run(args,started)


if __name__=='__main__':main()
