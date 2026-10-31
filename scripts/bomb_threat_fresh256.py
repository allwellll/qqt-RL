"""Bounded CPU-only, resumable production-JS fresh B recheck (never promotes)."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',QQT_STOCK_RULE='stock-window-v1')
import json,time,subprocess,sys,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from scripts.population_run_state import atomic_json,sha,lock
ROOT=Path(__file__).resolve().parents[1];BASE=Path('runs/bomb_threat_v1');OUT=BASE/'fresh256_slot01_u800'
TIERS=('hard','normal','easy');MODES=('greedy','categorical');ROLES=('candidate','parent','transport1536','selfplay3072')


def read(path):return json.loads(Path(path).read_text())


def verdict_status(verdicts):
    """An established failure survives unrelated missing evaluation data."""
    states=[v['status'] for v in verdicts]
    return 'FAIL' if 'FAIL' in states else 'INCOMPLETE' if 'INCOMPLETE' in states else 'PASS'


def gate(candidate,parent,references,shaping):
    """Frozen256 thresholds; each condition uses only its complete dependencies."""
    missing=[]
    required=('win_minus_loss','zero_bomb_games','bombs_per_game','draw_fraction','causal_kills','deaths','self_deaths','cage_kill_fraction')
    def finite(v):return isinstance(v,(int,float)) and math.isfinite(v)
    for role,m in [('candidate',candidate),('parent',parent)]:
        if m.get('games')!=256:missing.append(role+'_256_COMPLETE_REQUIRED')
        missing.extend(role+'_'+k+'_MISSING' for k in required if not finite(m.get(k)))
    for role in ('transport1536','selfplay3072'):
        m=references.get(role,{})
        if m.get('games')!=256 or not finite(m.get('win_minus_loss')):missing.append(role+'_MATCHED_256_REQUIRED')
    if not finite(shaping):missing.append('FROZEN_NATIVE_SHAPING_SHARE_MISSING')
    def known(m,k):return m.get('games')==256 and finite(m.get(k))
    checks={}
    def own(name,key,test):
        checks[name]=bool(test(candidate[key])) if known(candidate,key) else None
    def paired(name,key,test):
        checks[name]=bool(test(candidate[key],parent[key])) if known(candidate,key) and known(parent,key) else None
    paired('win_minus_loss_parent_plus32','win_minus_loss',lambda c,p:c>=p+32)
    own('zero_bomb_at_most28','zero_bomb_games',lambda c:c<=28)
    own('bombs_per_game_at_least20','bombs_per_game',lambda c:c>=20)
    paired('draw_fraction_at_most_parent','draw_fraction',lambda c,p:c<=p)
    paired('causal_kills_at_least_parent','causal_kills',lambda c,p:c>=p)
    paired('deaths_at_most_parent_plus20','deaths',lambda c,p:c<=p+20)
    paired('self_deaths_at_most_parent_plus8','self_deaths',lambda c,p:c<=p+8)
    paired('cage_kill_fraction_at_most_parent_plus10pp','cage_kill_fraction',lambda c,p:c<=p+.1+1e-9)
    checks['native_shaping_share_at_most30percent']=0<=shaping<=.3 if finite(shaping) else None
    for role in ('transport1536','selfplay3072'):
        m=references.get(role,{})
        checks[f'win_minus_loss_above_{role}']=candidate['win_minus_loss']>m['win_minus_loss'] if known(candidate,'win_minus_loss') and known(m,'win_minus_loss') else None
    failures=[k for k,v in checks.items() if v is False]
    return dict(status='FAIL' if failures else 'INCOMPLETE' if missing else 'PASS',
        data_status='INCOMPLETE' if missing else 'COMPLETE',conditions=checks,failures=failures,missing=missing)


def verify_sources(protocol):
    # Only the authorized Python decision/report revision may differ from the
    # original frozen source. Evaluation JS, models, seeds and shards stay bound.
    amendment_path=OUT/'gatefix_1945/SOURCE_AMENDMENT.json'
    for file,h in protocol['source_sha256'].items():
        actual=sha(file)
        if actual==h:continue
        amendment=read(amendment_path)
        assert file=='scripts/bomb_threat_fresh256.py' and amendment['status']=='AUTHORIZED',file
        assert amendment['old_sha256']==h and amendment['new_sha256']==actual,file
        assert amendment['protocol_sha256']==sha(OUT/'PROTOCOL.json'),file
        assert amendment['instruction']=='.hermes/STEER_gatefix_20261029_1945.md',file
        assert sha(amendment['instruction'])==amendment['instruction_sha256'],file


def validate_shard(path,job,protocol):
    primary=read(path);side=read(str(path)+'.outcome.json')
    model=protocol['models'][job['role']];tier=protocol['tiers'][job['tier']]
    assert primary['model_json_sha256']==model['model_sha256'] and not primary['assist']
    assert side['rule_version']=='stock-window-v1' and side['engine']=='production_JS' and side['teacher'] is False
    assert side['mode']==job['mode'] and side['simulator_sha256']==protocol['source_sha256']['web/sim-stock-window-v1.js']
    rows=side['episodes'];episodes=primary['results']['natural_full2400']['episodes']
    expected=list(range(tier['seed']+job['start'],tier['seed']+job['start']+8))
    assert [r['seed'] for r in rows]==expected and primary['seeds']==expected and len(episodes)==8
    for i,(r,e) in enumerate(zip(rows,episodes,strict=True)):
        assert r['done'] and 0<r['ticks']<=2400 and r['winner'] in (None,-1,0,1)
        assert r['seed']==e['seed'] and r['seat']==e['team']==(job['start']+i)%2 and r['ticks']==e['ticks']
        assert r['causal_kills'][r['seat']]==e['causal_kill']
    return rows


def jobs():
    # All three greedy tiers first; categorical candidate never gates the verdict.
    return [dict(tier=t,role=r,mode=m,start=s) for m in MODES for t in TIERS
        for r in (ROLES if m=='greedy' else ('candidate',)) for s in range(0,256,8)]


def job_path(job):return OUT/job['tier']/job['role']/f"{job['mode']}_{job['start']:03d}.json"


def publish(protocol):
    from scripts.bomb_threat_eval import summarize
    summaries={};files={};done=0
    for tier in TIERS:
        summaries[tier]={}
        for role in ROLES:
            summaries[tier][role]={}
            for mode in (MODES if role=='candidate' else ('greedy',)):
                rows=[];complete=True
                for start in range(0,256,8):
                    job=dict(tier=tier,role=role,mode=mode,start=start);p=job_path(job);proof=Path(str(p)+'.ACCEPTED.json')
                    if not proof.exists():complete=False;continue
                    accepted=read(proof)
                    for file,h in accepted['files'].items():assert sha(file)==h;files[file]=h
                    rows+=validate_shard(p,job,protocol);done+=8
                if complete:
                    assert len(rows)==256 and len({r['seed'] for r in rows})==256
                    summaries[tier][role][mode]=summarize(rows)
    verdicts={t:gate(summaries[t].get('candidate',{}).get('greedy',{}),summaries[t].get('parent',{}).get('greedy',{}),
        {r:summaries[t].get(r,{}).get('greedy',{}) for r in ROLES[2:]},protocol['native_shaping_share']) for t in TIERS}
    complete=all(len(summaries[t][r])==(2 if r=='candidate' else 1) for t in TIERS for r in ROLES)
    status=verdict_status(verdicts.values())
    value=dict(status=status,collection_status='COMPLETE' if complete else 'INCOMPLETE',completed_games=done,total_games=3840,per_tier=verdicts,metrics=summaries,
        protocol_sha256=sha(OUT/'PROTOCOL.json'),files=files,teacher=False,best_changed=False,training_untouched=True,unix=time.time())
    atomic_json(OUT/'RESULTS.json',value)
    hard=verdicts['hard'];c=summaries['hard']['candidate'].get('greedy',{});p=summaries['hard']['parent'].get('greedy',{})
    reasons=[]
    if 'win_minus_loss_parent_plus32' in hard['failures']:reasons.append(f"W-L差{c['win_minus_loss']-p['win_minus_loss']}<32")
    if 'zero_bomb_at_most28' in hard['failures']:reasons.append(f"单方零泡{c['zero_bomb_games']}>28")
    text=f'B 最终判定：{status}'+('（hard: '+ '，'.join(reasons)+'）' if reasons else '')+'\n\n'
    text+='# B fresh256 复核（20261029）\n\n'
    text+=f'**B最终判定：{status}。** '+('采集尚未完成；已确定失败立即列FAIL，缺失单列INCOMPLETE。normal/easy继续CPU采集，仅作信息参考。' if not complete else '三个难度分别判定，argmax为主；categorical仅报告。')+' 不晋升BEST，不发布。\n\n'
    text+='固定候选`bt_slot01_c000015/u000800`；生产JS带砖自然完整局，老师关闭，2400 tick原生终局；各档P0/P1各128。每档四模型argmax各256，候选categorical256，共3840局。父点与两网页参照同seed同出生点匹配；保留历史64局，不混作fresh。\n\n'
    text+='|档/角色/模式|局数|胜/平/负|胜-负|击杀|死亡|泡/局|单方零泡|双方零泡|平局击杀差|\n|---|---:|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for tier,rs in summaries.items():
        for role,ms in rs.items():
            for mode,m in ms.items():text+=f"|{tier}/{role}/{mode}|256|{m['win']}/{m['draw']}/{m['loss']}|{m['win_minus_loss']}|{m['causal_kills']}|{m['deaths']}|{m['bombs_per_game']:.3f}|{m['zero_bomb_games']}|{m['structural_zero_bomb_games']}|{m['draw_kill_difference']}|\n"
    text+='\n'+''.join(f'- {t}：{v["status"]}；失败={v["failures"]}；缺失={v["missing"]}。\n' for t,v in verdicts.items())
    text+='\n冻结细则：`runs/bomb_threat_v1/fresh256_slot01_u800/amendment01.json`；累计计数按64→256同比例扩展，比例阈值保持。fresh仅新seed，出生点复用冻结holdout64池并循环，非新地图泛化；训练塑形占比仅沿用候选真实训练前缀，不把JS回算当训练奖励。逐局结果及SHA见`RESULTS.json`。\n'
    p=Path('reports/bomb_threat_v1_fresh256_20261029.md');tmp=p.with_suffix('.md.tmp');tmp.write_text(text);tmp.replace(p)
    if complete:atomic_json(OUT/'COMPLETE.json',dict(status=status,result_sha256=sha(OUT/'RESULTS.json'),report=str(p),report_sha256=sha(p),unix=time.time()))
    return value


def main():
    OUT.mkdir(parents=True,exist_ok=True);guard=lock(OUT/'OWNER.lock');protocol=read(OUT/'PROTOCOL.json')
    verify_sources(protocol)
    for model in protocol['models'].values():assert sha(model['model_json'])==model['model_sha256']
    if (OUT/'COMPLETE.json').exists():return
    atomic_json(OUT/'EXECUTION.json',dict(status='RUNNING',cpu_only=True,workers=4,cpu_affinity=sorted(os.sched_getaffinity(0)),training_signals=0,unix=time.time()))
    def run(job):
        p=job_path(job);p.parent.mkdir(parents=True,exist_ok=True);proof=Path(str(p)+'.ACCEPTED.json')
        if proof.exists():
            for f,h in read(proof)['files'].items():assert sha(f)==h
            validate_shard(p,job,protocol);return
        if not (p.exists() and Path(str(p)+'.outcome.json').exists()):
            env=dict(os.environ,BT_FRESH_TIER=job['tier'],JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='256',
                JE10_ROUTE_START=str(job['start']),JE10_ROUTE_END=str(job['start']+8),JE10_ROUTE_SEED_START=str(protocol['tiers'][job['tier']]['seed']))
            with p.with_suffix('.log').open('ab') as f:
                subprocess.run(['node','scripts/bomb_threat_fresh256_eval.js',job['mode'],protocol['models'][job['role']]['model_json'],str(p),
                    'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',protocol['tiers'][job['tier']]['spawn_manifest']],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        validate_shard(p,job,protocol)
        atomic_json(proof,dict(job=job,protocol_sha256=sha(OUT/'PROTOCOL.json'),files={str(f):sha(f) for f in (p,Path(str(p)+'.outcome.json'))},unix=time.time()))
    try:
        publish(protocol)
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending={pool.submit(run,j):j for j in jobs()}
            for i,f in enumerate(as_completed(pending)):
                f.result();atomic_json(OUT/'PROGRESS.json',dict(status='RUNNING',last_job=pending[f],completed_shards=i+1,total_shards=480,unix=time.time()))
                if (i+1)%16==0:publish(protocol)
        result=publish(protocol)
        atomic_json(OUT/'EXECUTION.json',dict(status='COMPLETED',B=result['status'],completed_games=result['completed_games'],training_untouched=True,unix=time.time()))
        from scripts.iter_pbt_owner import notify
        notify('B_FRESH256_COMPLETE',f'B fresh256每档argmax复核完成：{result["status"]}；3840完整局含匹配父/网页参照与候选采样，报告已保存，未晋升BEST。')
    except Exception as error:
        atomic_json(OUT/'EXECUTION.json',dict(status='BLOCKED',error_type=type(error).__name__,reason=str(error).replace(str(ROOT),'<WORKSPACE>'),training_untouched=True,unix=time.time()))
        raise

if __name__=='__main__':main()
