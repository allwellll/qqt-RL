"""Durable, CPU-only ITER2 temperature/rank diagnostics; never starts training."""
import json,os,subprocess,sys,time
from threading import Lock
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scripts.population_run_state import atomic_json,sha,lock
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/bomb_threat_v1/iter2'
PROGRESS_LOCK=Lock()


def progress(value):
    # population_run_state.atomic_json uses a fixed .tmp name. Independent
    # shard threads must serialize writes to their shared progress marker.
    with PROGRESS_LOCK:atomic_json(BASE/'D12_PROGRESS.json',value)


def read(p):return json.loads(Path(p).read_text())


def accept_shard(path,model_hash,start,seed,tau,rank=False):
    document=read(path);outcome=read(str(path)+'.outcome.json');actions=read(str(path)+'.actions.json')
    rows=outcome['episodes'];episodes=document['results']['natural_full2400']['episodes']
    assert document['model_json_sha256']==model_hash
    assert outcome['engine']=='production_JS' and outcome['rule_version']=='stock-window-v1' and not outcome['teacher']
    assert actions['tau']==tau and actions['record']==rank
    assert len(rows)==len(episodes)==8
    by_seed={r['seed']:r for r in rows}
    assert set(by_seed)==set(range(seed+start,seed+start+8))
    for e in episodes:
        r=by_seed[e['seed']];assert e['team']==r['seat']==e['game']%2
        assert r['done'] and 0<r['ticks']<=2400 and e['ticks']==r['ticks']
    if rank:
        assert sum(r['ticks'] for r in actions['rows'])==sum(r['ticks'] for r in rows)
        for r in actions['rows']:assert sum(r['rank_counts'])==r['bomb_legal_ticks']
    files=[path,Path(str(path)+'.outcome.json'),Path(str(path)+'.actions.json')]
    return rows,dict(files={str(p.relative_to(ROOT)):sha(p) for p in files},
        games=8,seed_start=seed+start,tau=tau,rank_recorded=rank)


def source_rows(spec,mode,manifest):
    """Reuse only exact frozen native-brick production JS cells."""
    marker=ROOT/spec['reuse']/ 'NATURAL_COMPLETE.json'
    if not marker.exists():return None
    doc=read(marker)
    assert doc['policy_sha256']==spec['policy_sha256']
    assert doc['rule_version']=='stock-window-v1' and doc['seed']==manifest['seed']
    assert doc['simulator_sha256']==manifest['simulator_sha256'] and doc['spawn_sha256']==manifest['spawn_sha256']
    paths=[marker.parent/f'natural_{mode}_{s:03d}.json.outcome.json' for s in range(0,64,8)]
    if not all(p.exists() for p in paths):return None
    rows=[]
    for p in paths:
        assert sha(p)==doc['files'][p.name]
        side=read(p);assert side['engine']=='production_JS' and side['mode']==mode and not side['teacher']
        rows+=side['episodes']
    assert len(rows)==64 and {r['seed'] for r in rows}==set(range(manifest['seed'],manifest['seed']+64))
    assert all(r['done'] and r['seat']==(r['seed']-manifest['seed'])%2 and 0<r['ticks']<=2400 for r in rows)
    return rows,dict(reused=True,marker=str(marker.relative_to(ROOT)),marker_sha256=sha(marker),
        files={str(p.relative_to(ROOT)):sha(p) for p in paths})


def summarize_rank(paths):
    rows=[];examples=[]
    for p in paths:
        a=read(str(p)+'.actions.json');rows+=a['rows']
        for e in a['examples']:
            if len(examples)<3 and not any(x['seed']==e['seed'] and x['phase']==e['phase'] for x in examples):examples.append(e)
    out={}
    for r in rows:
        z=out.setdefault(r['phase'],dict(ticks=0,bomb_legal_ticks=0,bomb_top1_count=0,bomb_top1_gap_sum=0.,
            bomb_probability_sum=0.,rank_counts=[0,0,0],top_ability_counts=[0,0,0],movement_counts=[0]*5))
        for k in z:
            if isinstance(z[k],list):z[k]=[a+b for a,b in zip(z[k],r[k])]
            else:z[k]+=r[k]
    for z in out.values():
        z['bomb_top1_mean_gap']=z['bomb_top1_gap_sum']/z['bomb_top1_count'] if z['bomb_top1_count'] else None
        z['mean_bomb_probability_when_legal']=z['bomb_probability_sum']/z['bomb_legal_ticks'] if z['bomb_legal_ticks'] else None
    return dict(by_phase=out,examples=examples,heads_independent=True,
        ability_names=['none','bomb','held_item'],movement_names=['up','down','left','right','stay'])


def main():
    guard=lock(BASE/'DIAGNOSTIC_OWNER.lock');manifest=read(BASE/'D123_MANIFEST.json')
    runtime=read(BASE/'D123_EXECUTION_AMENDMENT.json') if (BASE/'D123_EXECUTION_AMENDMENT.json').exists() else {}
    for f,h in manifest['source_sha256'].items():assert sha(ROOT/f)==runtime.get('source_sha256',{}).get(f,h),f
    if (BASE/'D3/COMPLETE.json').exists() and (BASE/'D12_COMPLETE.json').exists():return
    started=time.time();results=[]
    # Dependencies stay local to each condition. Every shard has immutable SHA
    # acceptance and is resumed independently; failures never rerun accepted work.
    from scripts.bomb_threat_eval import summarize
    from scripts.je08_bun_authority import export
    for spec in manifest['models']:
        assert sha(ROOT/spec['checkpoint']/'policy.pt')==spec['policy_sha256']
        model=export(ROOT/spec['checkpoint']/'policy.pt',BASE/'models'/spec['name'])
        for tau in manifest['temperatures']:
            rank=tau==0 and spec['name'] in ('parent','slot00_u1600')
            target=BASE/'D1'/spec['name']/f'tau{tau:g}';target.mkdir(parents=True,exist_ok=True)
            done=target/'COMPLETE.json'
            if done.exists():
                old=read(done)
                for f,h in old['files'].items():assert sha(ROOT/f)==h
                results.append(old);continue
            mode='greedy' if tau==0 else 'categorical'
            reuse=source_rows(spec,mode,manifest) if tau in (0,1) and not rank else None
            if reuse:
                rows,proof=reuse;files=proof['files']
            else:
                def shard(start):
                    path=target/f'games{start:03d}.json'
                    lane=manifest['cpu_lanes'][(start//8)%len(manifest['cpu_lanes'])]
                    # Lane locks prevent two tasks sharing the same reserved pair.
                    with (BASE/f'lane{lane[0]}.lock').open('a') as lane_lock:
                        import fcntl
                        fcntl.flock(lane_lock,fcntl.LOCK_EX)
                        if not Path(str(path)+'.ACCEPTED.json').exists():
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                                QQT_STOCK_RULE='stock-window-v1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                ITER2_TAU=str(tau),ITER2_RANK=str(int(rank)),JE10_ROUTE_TASK='natural_full2400',
                                JE10_ROUTE_GAMES='64',JE10_ROUTE_SEED_START=str(manifest['seed']),
                                JE10_ROUTE_START=str(start),JE10_ROUTE_END=str(start+8))
                            cmd=['taskset','-c',','.join(map(str,lane)),'node','scripts/bomb_threat_iter2_actions.js',mode,
                                str(model.relative_to(ROOT)),str(path.relative_to(ROOT)),manifest['regions'],manifest['spawn_manifest']]
                            tick=time.time();retained=all(p.exists() for p in (path,Path(str(path)+'.outcome.json'),Path(str(path)+'.actions.json')))
                            if not retained:
                                with path.with_suffix('.log').open('ab') as stream:
                                    subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                            rows,proof=accept_shard(path,sha(model),start,manifest['seed'],tau,rank)
                            atomic_json(str(path)+'.ACCEPTED.json',dict(**proof,elapsed_seconds=None if retained else time.time()-tick,
                                resumed_complete_output=retained,unix=time.time()))
                        else:
                            proof=read(str(path)+'.ACCEPTED.json')
                            for f,h in proof['files'].items():assert sha(ROOT/f)==h
                            rows,_=accept_shard(path,sha(model),start,manifest['seed'],tau,rank)
                    progress(dict(model=spec['name'],tau=tau,last_shard=start,
                        completed_conditions=len(results),cpu_only=True,training_unaffected=True,unix=time.time()))
                    return path,rows,proof
                with ThreadPoolExecutor(max_workers=len(manifest['cpu_lanes'])) as pool:parts=list(pool.map(shard,range(0,64,8)))
                rows=[r for _,rr,_ in parts for r in rr];files={f:h for _,_,p in parts for f,h in p['files'].items()}
                proof=dict(reused=False,passive_rank_instrumentation=rank)
                if rank:
                    rank_value=summarize_rank([p for p,_,_ in parts]);atomic_json(target/'RANK.json',rank_value)
                    # Repeat greedy trajectories only to obtain missing passive
                    # logits; their existing full outcomes must remain identical.
                    old=source_rows(spec,'greedy',manifest)
                    if old:
                        keys=('seed','seat','ticks','winner','requests','placed','deaths','causal_kills','cage_kills','self_deaths','stored')
                        def canonical(rr):return [{k:r[k] for k in keys} for r in sorted(rr,key=lambda r:r['seed'])]
                        assert canonical(rows)==canonical(old[0]),'passive logging changed frozen outcomes'
                        proof['existing_outcomes_exact']=True
            assert len(rows)==64 and len({r['seed'] for r in rows})==64
            value=dict(model=spec['name'],checkpoint=spec['checkpoint'],policy_sha256=spec['policy_sha256'],tau=tau,
                metrics=summarize(rows),files=files,provenance=proof,manifest_sha256=sha(BASE/'D123_MANIFEST.json'),unix=time.time())
            atomic_json(done,value);results.append(value);write_report(results,started)
    atomic_json(BASE/'D12_COMPLETE.json',dict(status='COMPLETE',results=results,elapsed_seconds=time.time()-started,
        new_training_steps=0,production_decode_changed=False,unix=time.time()))
    write_report(results,started)
    from scripts.iter_pbt_owner import notify
    notify('BOMB_THREAT_ITER2_D12_COMPLETE','ITER2温度扫描与放泡排名诊断已完成；10:00起点决定已消费，三arm训练独立继续，接续CPU D3。',
        receipt='runs/bomb_threat_v1/iter2/D12_COMPLETE.json')
    with (BASE/'D3.log').open('ab') as stream:
        subprocess.run(['taskset','-c','68-71',sys.executable,'-m','scripts.bomb_threat_iter2_native'],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1'),
            stdout=stream,stderr=subprocess.STDOUT,check=True)
    native=read(BASE/'D3/COMPLETE.json');assert native['status']=='COMPLETE'
    write_report(results,started)
    execution=read(BASE/'EXECUTION.json');execution.update(status='D123_COMPLETE_TRAINING_CONTINUES',
        source_state_block_resolved_by_1000_decision=True,training_status='runs/bomb_threat_v1/iter2/TRAINING_STATUS.json',
        D3_complete_sha256=sha(BASE/'D3/COMPLETE.json'),unix=time.time());atomic_json(BASE/'EXECUTION.json',execution)
    notify('BOMB_THREAT_ITER2_D123_COMPLETE','ITER2 D1/D2/D3诊断完成；u960 Actor加新Critic预热的三arm训练继续，未改网页或BEST。',
        receipt='runs/bomb_threat_v1/iter2/EXECUTION.json')


def write_report(results,started):
    report=ROOT/'reports/bomb_threat_v1_iter2_diagnostics_20261029.md'
    text=['# ITER2 限制因素诊断（20261029）','',
        'amend2整局审计已完成；起点缺完整state的历史阻塞已由10:00决定解决，使用u960 Actor、新Critic/Adam/env/RNG和20次Critic预热。完整副本查验为历史收据，不再作为阻塞；实际arm状态见 `runs/bomb_threat_v1/iter2/TRAINING_STATUS.json`。此诊断不控制训练、不修改网页。',
        '', '冻结诊断：`runs/bomb_threat_v1/iter2/D123_MANIFEST.json`。dev seed2026122900起64局，带砖、自然出生空手、席位交替、生产JS normal Hunter、teacher关闭；独立actor与opponent RNG。温度同时作用于移动/ability两个logit头，τ0=原argmax、τ1=原categorical；网页未改。τ0/1已有同协议结果按逐局SHA复用；D2需要未记录logits而额外被动重放两点argmax，逐局终局/事件和已有结果严格核对。',
        '', 'D2口径amendment：ability的none不是移动/停止，移动头独立。排名只在合法ability集合内；全决策tick与合法泡tick分母分列；固定选前三个不同seed/phase的rank2状态，不按输赢选。',
        '', '|模型|τ|局数|胜/平/负|胜−负|因果击杀|死亡|泡/局|零泡局|', '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in results:
        m=r['metrics'];text.append(f"|{r['model']}|{r['tau']}|{m['games']}|{m['win']}/{m['draw']}/{m['loss']}|{m['win_minus_loss']}|{m['causal_kills']}|{m['deaths']}|{m['bombs_per_game']:.3f}|{m['zero_bomb_games']}|")
    text+=['','D2实际排名（合法ability分母；不是Q或动作优势）：',
        '', '|模型/阶段|全决策tick|合法放泡tick|放泡rank1/2/3|合法时平均放泡概率|放泡top1平均概率差|',
        '|---|---:|---:|---|---:|---:|']
    for model in ('parent','slot00_u1600'):
        path=BASE/'D1'/model/'tau0/RANK.json'
        if not path.exists():
            text.append(f'|{model}/待完成|—|—|—|—|—|');continue
        rank=read(path)
        for phase,r in rank['by_phase'].items():
            text.append(f'|{model}/{phase}|{r["ticks"]}|{r["bomb_legal_ticks"]}|{r["rank_counts"]}|{r["mean_bomb_probability_when_legal"]}|{r["bomb_top1_mean_gap"]}|')
        examples=[{k:e[k] for k in ('seed','seat','tick','phase','top_ability_name','movement_name','ability_logits','probabilities','state_sha256')} for e in rank['examples']]
        text+=['',f'{model} 固定首个seed/phase的rank2实例（完整状态及哈希见`{path.relative_to(ROOT)}`）：`'+json.dumps(examples,ensure_ascii=False)+'`。','']
    text+=['',f'已完成条件{len(results)}/20；当前诊断进程墙钟{time.time()-started:.1f}秒，非训练量。逐局原始数据、policy SHA、接纳收据在上述namespace。D3单独使用JAX hunter，不和JS合并；环境、随机流/边界差异均须披露，不能单因素声称“对手更弱”。',
        '', 'arm配置按任务及10:00初始化决定：slot04=A1（cur .10、weights [.05,.35,.30,.15,.10,.05]），slot07=A2（A1+320更新entropy .005→.0015），slot01=A3（A1+三项×5、cap .60）。slot00/03、05/06保持。A3完整局审计仅32局rule3且无撤销/拒绝暴露，不能用它证明所有场景无异常。',
        '', '200更新桶、640更新比较与真实首更新由独立训练收据/轮次报告提供，诊断结果不伪填缺失训练量。无BEST晋升、无发布。']
    d3=BASE/'D3/COMPLETE.json'
    if d3.exists():
        v=read(d3);text+=['','D3完整局结果：`'+str(d3.relative_to(ROOT))+'`，'+json.dumps(v['metrics'],ensure_ascii=False)+
            '。JAX hunter即tier3，旧脚本部分标为rule3；同seed/出生对/交替席位但不同引擎和PRNG实现，不能把总差异单归因给对手。']
    report.write_text('\n'.join(text)+'\n')
    atomic_json(BASE/'EXECUTION.json',dict(status='CPU_DIAGNOSTICS_RUNNING_INIT_DECISION_CONSUMED',
        completed_D1_conditions=len(results),expected_D1_conditions=20,report=str(report.relative_to(ROOT)),report_sha256=sha(report),
        sole_existing_training_owner=True,training_worker_signals=0,unix=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(BASE/'DIAGNOSTIC_FAILURE.json',dict(status='BLOCKED',error_type=type(error).__name__,
            reason=str(error)[:400] if isinstance(error,AssertionError) else 'see local diagnostic log',training_unaffected=True,unix=time.time()))
        raise
