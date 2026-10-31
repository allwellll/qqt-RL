"""Persistent GPU2, physical first-terminal JAX Hunter full-game evaluation.

No reward adapter, training optimizer, Bank, or JS gate is called here.
The historical eight-game native reward-alignment artifacts remain separate.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from scripts.population_run_state import atomic_json, lock, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/bomb_threat_v1'
VERSION = 'jax-hunter-fullgame256-v1'
EVENTS = ('placed', 'requests', 'causal_kills', 'deaths', 'self_deaths',
          'capture', 'stock_steals', 'loose_pickups', 'transfers')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def make_plan(request, spawns):
    modes = [dict(name='greedy', tau=0.), dict(name='categorical', tau=1.)]
    if request.get('iter2_arm'):
        modes.append(dict(name='categorical_tau0.5', tau=.5))
    return dict(version=VERSION, games_per_mode=256, seed_start=spawns['seed'],
                horizon=2400, modes=modes, split=request.get('split', 'dev'),
                spawn_mapping='index modulo frozen JS spawn_cells length, same as bomb_threat_eval.js',
                seat_mapping='episode_index modulo2', first_terminal_only=True,
                teacher=False, gate_input=False, opponent='jax_hunter tier3')


def initial_spec(plan, spawns):
    return [dict(seed=plan['seed_start']+i, episode_index=i, mode=mode['name'],
                 tau=mode['tau'], seat=i % 2,
                 spawn_cells=spawns['spawn_cells'][i % len(spawns['spawn_cells'])])
            for mode in plan['modes'] for i in range(plan['games_per_mode'])]


def summarize(rows, plan):
    result = {}
    assert len(rows) == plan['games_per_mode'] * len(plan['modes'])
    for mode in plan['modes']:
        rr = [r for r in rows if r['mode'] == mode['name']]
        assert len(rr) == plan['games_per_mode']
        assert {r['seed'] for r in rr} == set(range(plan['seed_start'], plan['seed_start']+plan['games_per_mode']))
        assert all(r['seat'] == (r['seed']-plan['seed_start']) % 2 and r['done'] and
                   0 < r['ticks'] <= plan['horizon'] and r['winner'] in (-1,0,1) and
                   all(a <= b for a,b in zip(r['placed'],r['requests'])) for r in rr)
        n = len(rr)
        value = dict(games=n,win=sum(r['winner']==r['seat'] for r in rr),
                     draw=sum(r['winner']==-1 for r in rr),loss=sum(r['winner']==1-r['seat'] for r in rr),
                     artificial_truncations=0)
        value['win_minus_loss'] = value['win']-value['loss']
        value['draw_fraction'] = value['draw']/n
        for field in EVENTS:
            value[field] = sum(r[field][r['seat']] for r in rr)
        value.update(actual_bombs=value['placed'],bomb_requests=value['requests'],
                     bombs_per_game=value['placed']/n,
                     zero_bomb_games=sum(r['placed'][r['seat']]==0 for r in rr),
                     capture_episodes=sum(r['capture'][r['seat']]>0 for r in rr),
                     death_episodes=sum(r['deaths'][r['seat']]>0 for r in rr),
                     delivery=value['capture'],mean_ticks=sum(r['ticks'] for r in rr)/n)
        result[mode['name']] = value
    return result


def latency_report(root=ROOT, base=BASE, now=None):
    """First immutable full-JS result; regenerated gates aren't completion time."""
    now = time.time() if now is None else now
    finished, pending, evidence = [], [], []
    for path in (base/'eval_requests').glob('*.json'):
        req = read(path)
        if not req.get('checkpoint'):
            continue
        cp = root/req['checkpoint']; cm = read(cp/'COMMITTED.json')
        if not cm:
            continue
        started = cm.get('metrics', {}).get('unix') or (cp/'COMMITTED.json').stat().st_mtime
        marker = base/'eval'/path.stem/'NATURAL_COMPLETE.json'
        if marker.exists():
            m = read(marker); elapsed = max(0.,m.get('unix',marker.stat().st_mtime)-started)
            finished.append(elapsed)
            evidence.append(dict(request=path.stem,split=req.get('split','dev'),
                checkpoint=req['checkpoint'],seconds=elapsed,marker=str(marker.relative_to(root)),
                marker_sha256=sha(marker)))
        else:
            pending.append(max(0.,now-started))
    def stats(values):
        return dict(count=len(values),P50_seconds=float(np.median(values)) if values else None,
                    max_seconds=max(values) if values else None)
    return dict(completed=stats(finished),pending_age=stats(pending),evidence=evidence,
                definition='checkpoint native commit unix to immutable NATURAL_COMPLETE unix; historical included, not only current queue wait',
                unix=now)


def build_scan(E, chunk=128):
    """Return a fixed-shape physics scan, reusable across models and jobs."""
    import jax
    import jax.numpy as j

    def scan(params, states, keys, ended, seats, tau):
        n = seats.shape[0]; ids = j.arange(n)
        def body(carry, _):
            st, key, stopped = carry
            split = jax.vmap(lambda k:jax.random.split(k,5))(key)
            nk,ka,kb,ko,ke = j.swapaxes(split,0,1)
            obs = E.train.both_perspectives(st); gv = E.train.both_states(st)
            mv,ab,*_ = E.train.net_forward(params,'transformer',obs[ids+seats*n],gv[ids+seats*n])
            mm,bm = jax.vmap(E.env.legal_mask)(st)
            ml = j.where(mm[ids,seats],mv,-1e9); bl = j.where(bm[ids,seats],ab,-1e9)
            safe_tau = j.where(tau>0,tau,1.)[:,None]
            moves = j.where(tau>0,jax.vmap(jax.random.categorical)(ka,ml/safe_tau),j.argmax(ml,-1))
            abilities = j.where(tau>0,jax.vmap(jax.random.categorical)(kb,bl/safe_tau),j.argmax(bl,-1))
            own = j.stack((moves,abilities),-1)
            def hunter(s,seat,m,b,k):
                return E.bun_jax_bots.rule_bot_actions(jax.tree.map(lambda x:x[None],s),seat[None],m[None],b[None],k,j.array([3]))[0]
            enemy = jax.vmap(hunter)(st,1-seats,mm[ids,1-seats],bm[ids,1-seats],ko)
            actions = j.zeros((n,2,2),j.int32).at[ids,seats].set(own).at[ids,1-seats].set(enemy)
            after,done,info = jax.vmap(lambda s,a,k:E.env.step(s,a,k,auto_reset=False,return_info=True))(st,actions,ke)
            live = ~stopped
            events = j.stack((info['bomb_placed'],actions[:,:,1]==1,info['causal_kill'],
                info['death'],info['own_bomb_defeat'],info['capture'],info['steal'],
                info['pickup_loose'],info['native_transfer']),-1).astype(j.int8)
            data = dict(events=j.where(live[:,None,None],events,0),live=live,
                terminal=done&live,winner=j.where(done&live,info['winner'],-2).astype(j.int8),
                actions=j.where(live[:,None,None],actions,-1).astype(j.int8))
            return (E.select_tree(stopped,st,after),j.where(stopped[:,None],key,nk),stopped|done),data
        return jax.lax.scan(body,(states,keys,ended),None,length=chunk)
    return jax.jit(scan)


class Engine:
    def __init__(self):
        # Called before importing anything that initializes JAX.
        os.environ.update(CUDA_VISIBLE_DEVICES='2',JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',
            JAXBOMB_RULE='bun',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
            JAX_COMPILATION_CACHE_DIR=str(BASE/'native_fullgame_v2/cache'),
            JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0')
        import jax
        import jax.numpy as j
        from scripts import carrier_mechanism_engine as E
        from scripts.bomb_threat_runtime import install
        E.configure();install()
        assert jax.default_backend()=='gpu' and jax.local_device_count()==1
        assert E.bun_jax_bots.TIER_NAMES[3]=='hunter'
        self.jax,self.j,self.E = jax,j,E
        self.scan = build_scan(E)
        def initial(seeds,spawns):
            envkeys=jax.vmap(jax.random.PRNGKey)(seeds)
            states,_=E.initial(envkeys,j.array(0,j.int32))
            positions=spawns+.5
            states=states._replace(core=states.core._replace(pos=positions),bun_spawn_pos=positions)
            keys=jax.vmap(lambda k:jax.random.fold_in(k,0x515154))(envkeys)
            return states,keys
        self.initial=jax.jit(initial)

    def evaluate(self, request, protocol):
        from scripts.checkpoint_policy_artifact import load_actor,verify_policy
        jax,j,E=self.jax,self.j,self.E
        out=BASE/'eval'/request['name']/'native_hunter256_v1';out.mkdir(parents=True,exist_ok=True)
        guard=lock(out/'worker.lock')
        if (out/'NATIVE_HUNTER_COMPLETE.json').exists():return
        cp=ROOT/request['checkpoint'];cm=verify_policy(cp)
        spawn_path=BASE/f"{request.get('split','dev')}_spawns.json";spawns=read(spawn_path)
        plan=make_plan(request,spawns);cases=initial_spec(plan,spawns);n=len(cases)
        params=jax.device_put(load_actor(cp),jax.devices()[0])
        seeds=j.array([r['seed'] for r in cases],j.uint32)
        pairs=j.array([r['spawn_cells'] for r in cases],j.float32)
        seats=j.array([r['seat'] for r in cases],j.int32);tau=j.array([r['tau'] for r in cases],j.float32)
        states,keys=self.initial(seeds,pairs);jax.block_until_ready(states)
        assert all(x.platform=='gpu' for x in states.t.devices())
        assert np.all(np.asarray(states.t)==0) and np.all(np.asarray(states.bun_carried)<0)
        assert np.all(np.asarray(states.lesson)==E.env.LESSON_FULL)
        assert np.all(np.asarray(states.brick).any((1,2)))
        initial_brick=np.asarray(states.brick);initial_pos=np.asarray(states.pos)
        manifest=dict(**plan,checkpoint=request['checkpoint'],policy_sha256=cm['policy_sha256'],
            protocol_sha256=sha(BASE/'native_fullgame_v2/PROTOCOL.json'),spawn_sha256=sha(spawn_path),
            actor_device_platform='gpu',state_device_platform='gpu',batch_games=n,
            iter2_arm=request.get('iter2_arm'),counter_names=list(EVENTS),
            initial_empty_hand=True,initial_bricks_intact=True,
            limitations=['JAX/JS PRNG and Hunter implementations differ; not an engine-only contrast',
                'frozen64 birth pairs reused cyclically for256 seed labels, matching JS modulo mapping',
                'no reward/ledger tail; no critic/bootstrap estimate',
                'auxiliary only; never production JS gate/BEST/PBT input'],unix=time.time())
        atomic_json(out/'MANIFEST.json',manifest)
        carry=(states,keys,j.zeros(n,bool));traces=[];timings=[];started=time.time()
        for offset in range(0,2432,128):
            began=time.monotonic();carry,data=self.scan(params,*carry,seats,tau)
            jax.block_until_ready(carry);elapsed=time.monotonic()-began
            # Keep numeric traces on device until the full match finishes.
            traces.append(data);timings.append(elapsed)
            atomic_json(out/'PROGRESS.json',dict(offset=offset+128,batch_games=n,
                last_scan_seconds=elapsed,completed_games=int(np.asarray(carry[2]).sum()),
                compute_device='gpu',elapsed_seconds=time.time()-started,unix=time.time()))
            if np.all(np.asarray(carry[2])):break
        assert np.all(np.asarray(carry[2])),'2400tick physical terminal missing; never invent a truncation result'
        assert np.isfinite(np.asarray(carry[0].pos)).all()
        host_start=time.monotonic()
        arrays={key:np.concatenate([np.asarray(t[key]) for t in traces],axis=0) for key in traces[0]}
        assert np.all(arrays['terminal'].sum(0)==1)
        rows=[];stored=np.asarray(carry[0].bun_stored)
        for index,case in enumerate(cases):
            end=int(np.flatnonzero(arrays['terminal'][:,index])[0]);live=arrays['live'][:,index]
            counts=arrays['events'][:,index].astype(np.int32).sum(0)
            rows.append(dict(**{k:case[k] for k in ('seed','episode_index','seat','mode','tau')},
                done=True,ticks=int(live.sum()),winner=int(arrays['winner'][end,index]),
                **{name:counts[:,i].tolist() for i,name in enumerate(EVENTS)},stored=stored[index].tolist()))
        summary=summarize(rows,plan)
        raw=out/'TRACE.npz';tmp=out/'TRACE.npz.tmp'
        with tmp.open('wb') as stream:
            # Uncompressed numeric arrays avoid serial JSON/compression stalls.
            np.savez(stream,**arrays,initial_brick=initial_brick,initial_pos=initial_pos,
                seeds=np.asarray(seeds),seats=np.asarray(seats),tau=np.asarray(tau),
                counter_names=np.asarray(EVENTS));stream.flush();os.fsync(stream.fileno())
        tmp.replace(raw)
        atomic_json(out/'EPISODES.json',dict(plan=plan,episodes=rows,policy_sha256=cm['policy_sha256']))
        atomic_json(out/'NATIVE_HUNTER_SUMMARY.json',dict(**manifest,natural_jax_hunter=summary,
            scan_seconds=timings,host_output_seconds=time.monotonic()-host_start,
            environment_steps=int(arrays['live'].sum()),elapsed_seconds=time.time()-started))
        files={p.name:sha(p) for p in (raw,out/'EPISODES.json',out/'NATIVE_HUNTER_SUMMARY.json',out/'MANIFEST.json')}
        atomic_json(out/'NATIVE_HUNTER_COMPLETE.json',dict(status='COMPLETE',version=VERSION,
            files=files,policy_sha256=cm['policy_sha256'],games_per_mode=256,gate_input=False,unix=time.time()))


def run_owner():
    """Same native owner lock; one persistent GPU graph, no training launch."""
    from scripts.bomb_threat_eval_owner import catchup_order,request_kind
    from scripts.bomb_threat_owner import invalid_ancestry
    from scripts.iter_pbt_owner import notify
    guard=lock(BASE/'NATIVE_OWNER.lock');gpu_guard=lock(BASE/'GPU2_NATIVE.lock')
    protocol=read(BASE/'native_fullgame_v2/PROTOCOL.json')
    for file,h in protocol['source_sha256'].items():assert sha(ROOT/file)==h,file
    engine=None
    while not any((BASE/name).exists() for name in ('USER_STOP_NATIVE.json','USER_STOP.json','USER_PAUSE.json')):
        pending=[]
        for path in catchup_order(list((BASE/'eval_requests').glob('*.json'))):
            req=read(path)
            if not req.get('checkpoint') or request_kind(req)=='technical_boundary':continue
            cp=ROOT/req['checkpoint'];run=cp.parent
            if run.name.startswith('bt_') and (invalid_ancestry(run.name) or (run/'EXCLUDED_EVALUATION.json').exists()):continue
            out=BASE/'eval'/path.stem/'native_hunter256_v1'
            if (out/'NATIVE_HUNTER_COMPLETE.json').exists() or (out/'FAILURE.json').exists():continue
            pending.append((path,req,out))
        atomic_json(BASE/'NATIVE_STATUS.json',dict(status='RUNNING' if pending else 'IDLE_NO_PENDING',
            version=VERSION,pending=len(pending),dedicated_gpu=2,
            persistent_engine=True,training_launches=0,old_queues_consumed=False,unix=time.time()))
        if not pending:
            time.sleep(5);continue
        path,req,out=pending[0];out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/'LAUNCH.json',dict(request=str(path.relative_to(ROOT)),gpu=2,version=VERSION,unix=time.time()))
        try:
            if engine is None:engine=Engine()
            engine.evaluate(req,protocol)
            atomic_json(BASE/'native_fullgame_v2/JS_LATENCY.json',latency_report())
            atomic_json(BASE/'METRICS_REFRESH.json',dict(status='RETRY_PENDING',unix=0))
            notify('GPU2_'+path.stem+'_HUNTER256_COMPLETE',f'GPU2 JAX Hunter整局完成 {path.stem}；每模式256局、辅助指标，不替代JS闸门。')
        except Exception as error:
            import traceback
            failure=dict(status='BLOCKED',error_type=type(error).__name__,
                reason=str(error).replace(str(ROOT),'<WORKSPACE>'),
                traceback=traceback.format_exc().replace(str(ROOT),'<WORKSPACE>'),training_untouched=True,unix=time.time())
            atomic_json(out/'FAILURE.json',failure)
            notify('GPU2_'+path.stem+'_HUNTER256_FAILED',f'GPU2 JAX整局评估阻塞 {path.stem}：{type(error).__name__}；训练未停，证据已保存。')
            # A global technical fault needs repair, not a loop failing every job.
            atomic_json(BASE/'native_fullgame_v2/BLOCKED.json',failure)
            raise


if __name__=='__main__':
    run_owner()
