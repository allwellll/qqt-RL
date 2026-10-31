"""12:30 passive JS lineage watch; never launches/signals a training worker."""
import json,os,sys,time,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scripts.population_run_state import atomic_json,sha,lock
from scripts.bomb_threat_round15 import read,pin
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'
OUT=BASE/'iter2/a3sr/lineage_watch_20261030'
REPORT=ROOT/'reports/bomb_threat_v1_a3sr_lineage_watch_20261030.md'
INSTRUCTION='.hermes/STEER_post_publish_20261030_1230.md'
MODULE='scripts.bomb_threat_a3sr_lineage_watch'


def confirmation_gate(points):
    missing=[u for u in ('2280','2320') if points.get(u,{}).get('games')!=128 or
        any(points.get(u,{}).get('by_seat',{}).get(s,{}).get('games')!=64 for s in ('P0','P1'))]
    if missing:return dict(status='MISSING',missing_updates=missing)
    scores=[points[u]['by_seat']['P1']['win_minus_loss'] for u in ('2280','2320')]
    mean=sum(scores)/2
    return dict(status='BLOCK' if mean<=2 else 'PASS',P1_scores=scores,P1_mean=mean,threshold=2,missing_updates=[])


def boundary_block(slot):
    if slot!=5:return None
    r=read(OUT/'STOP_AT_CHUNK_BOUNDARY.json',{})
    override=read(BASE/'reboot_resume_20261030/SLOT05_EXPLICIT_RESUME.json',{})
    if override.get('status')=='AUTHORIZED' and r.get('reasons')==['rolling_holdout_BLOCK']:
        points=r.get('rolling',{}).get('points',[])
        updates=[int(p.split('_u')[-1].split('_')[0]) for p in points]
        if updates and max(updates)<=override['historical_update_max']:
            return None
    return r if r.get('status')=='BLOCK' and r.get('slot')==5 else None


def latch_stop(rolling,confirmation):
    reasons=[]
    if rolling.get('status')=='BLOCK':reasons.append('rolling_holdout_BLOCK')
    if confirmation.get('status')=='BLOCK':reasons.append('fresh_confirmation_P1_mean_le_2')
    if reasons and not boundary_block(5):
        atomic_json(OUT/'STOP_AT_CHUNK_BOUNDARY.json',dict(status='BLOCK',slot=5,reasons=reasons,
            rolling=rolling,confirmation=confirmation,action='complete current chunk atomically, stop only slot05; no rollback',unix=time.time()))


def freeze():
    path=OUT/'PROTOCOL.json'
    if path.exists():return read(path)
    oldpath=BASE/'fresh256_a3sr_vs_u800_20261030/PROTOCOL.json';old=read(oldpath)
    spawn=read(BASE/'holdout_spawns.json');spawn.update(seed=2026173000,games=128,
        spawn_cells=(spawn['spawn_cells']*2)[:128])
    OUT.mkdir(parents=True,exist_ok=True);atomic_json(OUT/'confirmation_spawns.json',spawn)
    proto=dict(id='A3SR-LINEAGE-WATCH-20261030-1230',status='FROZEN',instruction=INSTRUCTION,
        instruction_sha256=sha(ROOT/INSTRUCTION),reference_protocol=str(oldpath.relative_to(ROOT)),reference_protocol_sha256=sha(oldpath),
        confirmation=dict(slot=5,updates=[2280,2320],games_per_point=128,seats={'P0':64,'P1':64},
            seed_start=2026173000,seed_end=2026173127,same_paired_seeds_both_points=True,tier='hard',mode='greedy',
            spawn_manifest=str((OUT/'confirmation_spawns.json').relative_to(ROOT)),spawn_sha256=sha(OUT/'confirmation_spawns.json')),
        fresh=dict(slots=[0,3,4,7],update=2400,tiers=old['tiers'],games_per_tier=256,seats={'P0':128,'P1':128},
            baseline=old['models']['a3sr_u1280'],baseline_results=str((BASE/'fresh256_a3sr_vs_u800_20261030/RESULTS.json').relative_to(ROOT)),
            mode='greedy',new_games=3072,baseline_games_reused=768),
        engine='production_JS',opponent='bun.hunter',rule_version='stock-window-v1',horizon=2400,teacher=False,
        workers=4,CPU_affinity='116-119',GPU_training_cards_used=[],training_changes=[],no_publication=True,no_BEST_change=True,
        stop='only slot05 at complete chunk boundary if rolling BLOCK or mean P1 W-L of both full confirmation points <=2; no automatic rollback',
        source_sha256={f:h for f,h in old['source_sha256'].items() if f.endswith('.js')},
        limitations=['confirmation repeats frozen64 holdout geometries twice; seeds and seats paired, not128 unique maps',
            'fresh repeats original64 geometries four times; seed parity binds seat; not256 unique maps',
            'production JS runs on CPU; GPU2 native JAX remains auxiliary only and never substitutes for JS judgment'],unix=time.time())
    atomic_json(OUT/'AMENDMENT01_EXECUTION.json',dict(status='FROZEN',instruction=INSTRUCTION,
        clarification='same new128 seeds for two confirmation points; original production-JS seed/birth manifests reused for all four u2400 points; existing u1280 baseline reused without rerun',
        compute='CPU-only JS; existing GPU2 native queue may evaluate normal cadence requests, not training cards',
        retention='pin existing and future exact checkpoints before writes; already absent full states recorded unavailable, never reconstructed from policy',
        budget='256 confirmation plus3072 fresh complete new games, fixed points not score selection',unix=time.time()))
    atomic_json(path,proto);return proto


def protect_future():
    slots=read(BASE/'SLOTS.json',{});want=[]
    for g in (0,3,4,5,7):
        e=slots.get(str(g))
        if not e:continue
        for chunk in range(e['chunk'],e['chunk']+3):
            us=[2280,2320] if g==5 else [2400]
            want.extend(f'runs/bomb_threat_v1/formal/bt_slot{g:02d}_c{chunk:06d}/u{u:06d}' for u in us)
    pin(want,'12:30 fixed checkpoint observation; pin before atomic write, no policy-only recovery')


def find_checkpoint(slot,update):
    from scripts.bomb_threat_owner import causal_lineage_limits
    entry=read(BASE/'SLOTS.json',{})[str(slot)];limits=causal_lineage_limits(entry['run'])
    paths=[]
    for run,limit in limits.items():
        if not run.startswith(f'bt_slot{slot:02d}_') or update>limit:continue
        p=BASE/'formal'/run/f'u{update:06d}/COMMITTED.json'
        if not p.exists():continue
        cm=read(p)
        if cm.get('metrics',{}).get('fresh_parent'):continue
        proto=read(BASE/'protocols'/f'{run}.json',{})
        if proto.get('namespace_id')!='A3SR-ATTACK-R1-20261030' and proto.get('arms',{}).get(run,{}).get('iter2_arm')=='A3SR':paths.append(cm)
    # Boundary snapshots can repeat the same policy. Prefer actual training commit.
    return paths[0] if paths else None


def resolve_model(slot,update):
    path=OUT/'models'/f'slot{slot:02d}_u{update:06d}.json'
    if path.exists():
        d=read(path);assert sha(ROOT/d['model_json'])==d['model_sha256'];return d
    cm=find_checkpoint(slot,update)
    if not cm:return None
    cp=ROOT/cm['checkpoint'];pin([cm['checkpoint']],'12:30 committed fixed checkpoint; retain full state')
    assert sha(cp/'policy.pt')==cm['policy_sha256']
    # Reuse the already verified normal cadence export when available. Otherwise
    # export once under a separate CPU-only subprocess/lock, never a GPU process.
    model=BASE/'eval'/f'{cp.parent.name}_{cp.name}'/'model/model.json'
    if not model.exists():
        out=OUT/'exports'/f'slot{slot:02d}_u{update:06d}'
        code='from pathlib import Path;from scripts.je08_bun_authority import export;import sys;export(Path(sys.argv[1]),Path(sys.argv[2]))'
        with (OUT/'EXPORT.log').open('ab') as log:subprocess.run([sys.executable,'-c',code,str(cp/'policy.pt'),str(out)],env=cpu_env(),stdout=log,stderr=subprocess.STDOUT,check=True)
        model=out/'model.json'
    assert read(model)['meta']['checkpoint_sha256']==cm['policy_sha256']
    d=dict(slot=slot,update=update,checkpoint=cm['checkpoint'],policy_sha256=cm['policy_sha256'],
        model_json=str(model.relative_to(ROOT)),model_sha256=sha(model),state_retained=(cp/'state.pkl').exists())
    atomic_json(path,d);return d


def cpu_env():
    return dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',QQT_STOCK_RULE='stock-window-v1')


def specs(proto):
    c=proto['confirmation'];jobs=[]
    for u in c['updates']:
        for start in range(0,128,8):jobs.append(dict(kind='confirmation',slot=5,update=u,tier='hard',start=start,
            games=128,seed=c['seed_start'],spawn_manifest=c['spawn_manifest']))
    for slot in proto['fresh']['slots']:
        for tier,t in proto['fresh']['tiers'].items():
            for start in range(0,256,8):jobs.append(dict(kind='fresh',slot=slot,update=2400,tier=tier,start=start,
                games=256,seed=t['seed'],spawn_manifest=t['spawn_manifest']))
    return jobs


def shard_path(job):
    return OUT/job['kind']/f"slot{job['slot']:02d}_u{job['update']:06d}"/job['tier']/f"greedy_{job['start']:03d}.json"


def validate_shard(path,job,model):
    a=read(path);b=read(Path(str(path)+'.outcome.json'));rows=b['episodes'];eps=a['results']['natural_full2400']['episodes']
    assert a['model_json_sha256']==model['model_sha256'] and not a['assist']
    assert b['mode']=='greedy' and b['engine']=='production_JS' and not b['teacher'] and b['rule_version']=='stock-window-v1'
    assert b['simulator_sha256']==sha(ROOT/'web/sim-stock-window-v1.js')
    expected=list(range(job['seed']+job['start'],job['seed']+job['start']+8))
    assert a['seeds']==expected and [r['seed'] for r in rows]==expected and len(eps)==8
    for i,(r,e) in enumerate(zip(rows,eps,strict=True)):
        assert r['done'] and 0<r['ticks']<=2400 and r['seat']==(job['start']+i)%2
        assert r['winner'] in (None,-1,0,1)
        assert (r['seed'],r['seat'],r['ticks'])==(e['seed'],e['team'],e['ticks'])
        assert r['causal_kills'][r['seat']]==e['causal_kill']
    return rows,eps


def collect(job,model,proto):
    p=shard_path(job);p.parent.mkdir(parents=True,exist_ok=True);guard=lock(Path(str(p)+'.lock'))
    accepted=Path(str(p)+'.ACCEPTED.json')
    if accepted.exists():
        for f,h in read(accepted)['files'].items():assert sha(ROOT/f)==h
        validate_shard(p,job,model);return
    # A watcher-only restart adopts a surviving Node shard by its exact output
    # argument; it must not launch that same eight-game shard a second time.
    while shard_is_running(p):time.sleep(3)
    if not(p.exists() and Path(str(p)+'.outcome.json').exists()):
        env=cpu_env();env.update(BT_FRESH_TIER=job['tier'],JE10_ROUTE_TASK='natural_full2400',
            JE10_ROUTE_GAMES=str(job['games']),JE10_ROUTE_START=str(job['start']),JE10_ROUTE_END=str(job['start']+8),JE10_ROUTE_SEED_START=str(job['seed']))
        with p.with_suffix('.log').open('ab') as log:subprocess.run(['node','scripts/bomb_threat_fresh256_eval.js','greedy',model['model_json'],
            str(p.relative_to(ROOT)),'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',job['spawn_manifest']],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    validate_shard(p,job,model)
    atomic_json(accepted,dict(job=job,model=model,protocol_sha256=sha(OUT/'PROTOCOL.json'),files={str(f.relative_to(ROOT)):sha(f) for f in (p,Path(str(p)+'.outcome.json'))},unix=time.time()))


def shard_is_running(path):
    relative=str(path.relative_to(ROOT)).encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            argv=(proc/'cmdline').read_bytes().split(b'\0')
            if b'scripts/bomb_threat_fresh256_eval.js' in argv and relative in argv and (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return True
        except OSError:pass
    return False


def summarize_group(jobs,model):
    from scripts.bomb_threat_eval import summarize
    rows=[];episodes=[];files={}
    for j in jobs:
        p=shard_path(j);proof=Path(str(p)+'.ACCEPTED.json')
        if not proof.exists():continue
        for f,h in read(proof)['files'].items():assert sha(ROOT/f)==h;files[f]=h
        rr,ee=validate_shard(p,j,model);rows+=rr;episodes+=ee
    if not rows:return dict(games=0,files={})
    assert len(rows)==len({r['seed'] for r in rows})
    d=summarize(rows);d['final_enemy_buns_stored']=d['delivery'];d['delivery']=sum(e['delivery'] for e in episodes)
    for seat in (0,1):
        g=d['by_seat'][f'P{seat}']
        if g:
            g['final_enemy_buns_stored']=g['delivery'];g['delivery']=sum(e['delivery'] for e in episodes if e['team']==seat)
    d.update(files=files,model=model);return d


def publish(proto):
    from scripts.bomb_threat_round16_report import argmax
    conf={};fresh={};jobs=specs(proto)
    for u in (2280,2320):
        model=read(OUT/'models'/f'slot05_u{u:06d}.json',{})
        if model:conf[str(u)]=summarize_group([j for j in jobs if j['kind']=='confirmation' and j['update']==u],model)
    for slot in (0,3,4,7):
        model=read(OUT/'models'/f'slot{slot:02d}_u002400.json',{})
        if model:fresh[str(slot)]={tier:summarize_group([j for j in jobs if j['kind']=='fresh' and j['slot']==slot and j['tier']==tier],model) for tier in ('hard','normal','easy')}
    rolling=read(BASE/'iter2/a3sr/promotion_20261030/rolling_gates/slot05.json',{})
    confirmation=confirmation_gate(conf);latch_stop(rolling,confirmation)
    historical=[]
    for u in range(1960,2321,40):
        cm=find_checkpoint(5,u)
        if not cm:continue
        cp=Path(cm['checkpoint']);name=f'{cp.parent.name}_{cp.name}_holdout'
        d=argmax(name)
        if d:historical.append(dict(update=u,checkpoint=cm['checkpoint'],metrics=d))
    baseline=read(ROOT/proto['fresh']['baseline_results'])['metrics']
    total=sum(v.get('games',0) for v in conf.values())+sum(v.get('games',0) for x in fresh.values() for v in x.values())
    status='COMPLETE' if total==3328 else 'RUNNING'
    result=dict(status=status,completed_new_games=total,total_new_games=3328,confirmation=conf,confirmation_gate=confirmation,
        rolling=rolling,stop_at_chunk_boundary=boundary_block(5),fresh=fresh,historical=historical,
        baseline={t:baseline[t]['a3sr_u1280'] for t in ('hard','normal','easy')},protocol_sha256=sha(OUT/'PROTOCOL.json'),unix=time.time())
    atomic_json(OUT/'RESULTS.json',result)
    from scripts.iter_pbt_owner import notify
    for u,value in conf.items():
        done=OUT/'point_completions'/f'slot05_u{int(u):06d}_confirmation.json'
        if value['games']==128 and not done.exists():
            atomic_json(done,dict(status='COMPLETE',update=int(u),games=128,metrics=value,unix=time.time()))
            notify('A3SR_LINEAGE_CONFIRM_'+u,
                f"A3SR slot05 u{u}新增128局JS greedy确认完成：P0 W−L{value['by_seat']['P0']['win_minus_loss']:+d}、P1 {value['by_seat']['P1']['win_minus_loss']:+d}；两点均值判定{confirmation['status']}，不自动回退。")
    for slot,tiers in fresh.items():
        done=OUT/'point_completions'/f'slot{int(slot):02d}_u002400_fresh.json'
        if all(tiers[t]['games']==256 for t in ('hard','normal','easy')) and not done.exists():
            scores={t:tiers[t]['by_seat']['P1']['win_minus_loss'] for t in ('hard','normal','easy')}
            atomic_json(done,dict(status='COMPLETE',slot=int(slot),update=2400,games=768,P1_scores=scores,unix=time.time()))
            notify('A3SR_LINEAGE_FRESH_SLOT'+slot,f'A3SR slot{slot}固定u2400三档fresh768局完成，P1 W−L {scores}；与已发布u1280同seed对照已回填，不晋升不发布。')
    text=['# A3SR JS holdout 连续下滑与固定u2400复核 — 20261030','',
        f"状态 **{status}**；新增真实生产JS完整局 {total}/3328。slot05配方不变、不自动回退；四条新线只评固定u2400，不按分数选点。训练不停，不改网页/BEST。",'',
        f"冻结协议 `{(OUT/'PROTOCOL.json').relative_to(ROOT)}`；JS只用CPU116–119四路，GPU2原native辅助队列独立，绝不占训练卡或用JAX饱和成绩替代JS判断。",'',
        '## slot05 原holdout逐点','',
        '| update | seat | n | W/D/L | W−L | 因果击杀 | 泡/局 | 单方零泡 | 结构双零 | 自伤 | 笼内击杀/因果击杀 |',
        '|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
    def row(label,seat,g):
        if not g:return
        text.append(f"|{label}|{seat}|{g['games']}|{g['win']}/{g['draw']}/{g['loss']}|{g['win_minus_loss']:+d}|{g['causal_kills']}|{g['bombs_per_game']:.3f}|{g['zero_bomb_games']}|{g['structural_zero_bomb_games']}|{g['self_deaths']}|{g['cage_kills']}/{g['causal_kills']}|")
    for h in historical:
        for seat,g in h['metrics']['by_seat'].items():row(h['update'],seat,g)
    text+=['',f"当前rolling **{rolling.get('status','MISSING')}**，失败项{rolling.get('failures',[])}；新seed确认 **{confirmation['status']}**。达到rolling BLOCK或两个完整确认点P1 W−L均值≤+2时锁存停止请求，只在当前完整chunk边界阻止slot05下一chunk；不会中断在途更新，不自动选u2080回退。",'',
        '## 额外128局确认（固定新seed）','',
        'u2280与u2320共用seed2026173000–2026173127，每点P0/P1各64；hard、greedy、2400tick、teacher off。出生几何复用64池两次，不是128独立地图。与原64局holdout的seat分母32不同，不混算。','',
        '| update | seat | n | W/D/L | W−L | 因果击杀 | 泡/局 | 单方零泡 | 结构双零 | 自伤 | 笼内击杀/因果击杀 |',
        '|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for u,g in conf.items():
        for seat,s in g.get('by_seat',{}).items():row(u,seat,s)
    text+=['','## 四新线固定u2400 fresh256 与已发布u1280','',
        'hard/normal/easy原seed2026142900/2026143900/2026144900，每档256、P0/P1各128，旧u1280的768局直接复用；4×768新增3072局。全部生产JS Hunter bun.hunter、stock-window-v1、带砖、greedy、2400tick、老师关闭；64几何循环4次且seat绑定奇偶，不夸称256地图泛化。', '',
        '| line/tier | seat | n | W/D/L | W−L | 因果击杀 | 泡/局 | 单方零泡 | 结构双零 | 自伤 | 笼内击杀/因果击杀 |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for tier in ('hard','normal','easy'):
        for seat,g in baseline[tier]['a3sr_u1280']['by_seat'].items():row('u1280/'+tier,seat,g)
        for slot,tiers in fresh.items():
            for seat,g in tiers[tier].get('by_seat',{}).items():row('slot'+slot+'/u2400/'+tier,seat,g)
    audit=read(OUT/'RETENTION_INITIAL_AUDIT.json',{})
    text+=['','## 完整state与证据边界','',
        '已在收到指令后立即pin全部u2080–u2240及未来u2280/u2320/u2400。首次现场审计中以下state.pkl已经不存在：'+str(audit.get('missing_states',[]))+'。仅policy存在不能恢复原Critic/Adam/env/RNG，不伪造完整state、不重训补历史。u2240完整state保留，其他同SHA副本搜索记录在RETENTION_REPLICA_AUDIT.json。','',
        '原始文件路径及SHA逐点列于RESULTS.json中historical[*].metrics.files、confirmation及fresh各条件的files；每片8局，逐局seed/seat/终局/因果击杀与主日志校验。原holdout位于`runs/bomb_threat_v1/eval/<run>_u<update>_holdout/natural_greedy_*.json{,.outcome.json}`；本次新增位于`runs/bomb_threat_v1/iter2/a3sr/lineage_watch_20261030/{confirmation,fresh}/`。零泡严格分单方、结构双零；自伤用实际ownBombDefeat；笼内击杀分母为因果击杀，零分母不推断。','',
        f"复现 `<PYTHON> -m {MODULE}`，只消费缺失分片，独立锁防重复；训练owner的边界闸门直接读取STOP_AT_CHUNK_BOUNDARY.json。",'']
    REPORT.write_text('\n'.join(text))
    atomic_json(OUT/'EXECUTION_STATUS.json',dict(status=status,completed_new_games=total,total_new_games=3328,
        confirmation_gate=confirmation,stop_requested=bool(boundary_block(5)),report=str(REPORT.relative_to(ROOT)),report_sha256=sha(REPORT),unix=time.time()))
    if status=='COMPLETE' and not (OUT/'COMPLETE.json').exists():
        atomic_json(OUT/'COMPLETE.json',dict(status='COMPLETE',games=total,results_sha256=sha(OUT/'RESULTS.json'),unix=time.time()))
        notify('A3SR_LINEAGE_WATCH_COMPLETE','A3SR slot05新seed确认与四新线固定u2400三档fresh全部3328局完成；报告已落盘，未改配方/网页/BEST。',report=str(REPORT.relative_to(ROOT)))
    return result


def ensure_running():
    if not (OUT/'PROTOCOL.json').exists() or (OUT/'COMPLETE.json').exists():return
    if time.time()-read(OUT/'BLOCKED.json',{}).get('unix',0)<60:return
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if MODULE.encode() in (p/'cmdline').read_bytes().split(b'\0') and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return
        except OSError:pass
    stream=(OUT/'OWNER.log').open('ab',buffering=0)
    subprocess.Popen(['taskset','-c','116-119',sys.executable,'-m',MODULE],cwd=ROOT,env=cpu_env(),
        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    stream.close()


def main():
    os.environ.update(cpu_env());OUT.mkdir(parents=True,exist_ok=True);guard=lock(OUT/'OWNER.lock');proto=freeze()
    for f,h in proto['source_sha256'].items():assert sha(ROOT/f)==h,f
    protect_future();futures={};last=0;pool=ThreadPoolExecutor(max_workers=4)
    try:
        while not (OUT/'COMPLETE.json').exists():
            protect_future()
            for future,job in list(futures.items()):
                if not future.done():continue
                future.result();del futures[future];last=0
            if time.time()-last>30:
                r=publish(proto);last=time.time()
                atomic_json(OUT/'TASK_CONSUMED.json',dict(status='COMPLETE' if r['status']=='COMPLETE' else 'RUNNING',
                    instruction=INSTRUCTION,protocol_sha256=sha(OUT/'PROTOCOL.json'),active_shards=list(futures.values()),
                    completed_new_games=r['completed_new_games'],training_worker_signals=0,unix=time.time()))
                if boundary_block(5):
                    from scripts.iter_pbt_owner import notify
                    notify('A3SR_LINEAGE_SLOT05_BOUNDARY_REQUEST','A3SR slot05 JS监控触发批准停止条件；请求在当前完整chunk原子保存后停，其他线继续、不自动回退。')
            ready_models={}
            for job in specs(proto):
                if len(futures)>=4:break
                if job in futures.values() or Path(str(shard_path(job))+'.ACCEPTED.json').exists():continue
                key=(job['slot'],job['update'])
                if key not in ready_models:ready_models[key]=resolve_model(*key)
                model=ready_models[key]
                if model:futures[pool.submit(collect,job,model,proto)]=job
            time.sleep(3)
    except Exception as error:
        atomic_json(OUT/'BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,reason=str(error).replace(str(ROOT),'<WORKSPACE>'),training_untouched=True,unix=time.time()))
        from scripts.iter_pbt_owner import notify
        notify('A3SR_LINEAGE_WATCH_BLOCKED',f'A3SR被动JS观察技术阻塞：{type(error).__name__}，训练不受影响，已保留完整分片与日志。')
        raise
    finally:pool.shutdown(wait=True)


if __name__=='__main__':main()
