"""Resumable CPU-only pressure matrix; full games never gate healthy GPU."""
import json
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from scripts.population_run_state import atomic_json, lock, sha

BASE=Path('runs/r4_combat_20261031');OUT=BASE/'interceptor'


def read(p):return json.loads(Path(p).read_text())


def freeze():
    p=OUT/'PRESSURE_PROTOCOL.json'
    if p.exists():return read(p)
    from scripts.bun_spawn_side import manifest
    spawn=OUT/'fresh_spawns.json';atomic_json(spawn,manifest(2026336000,256))
    old=read('runs/r3_selfplay_ramp_20261031/evaluation/parent/model/IDENTITY.json')
    parent=read(BASE/'EFFECTIVE_PROTOCOL.json')['parent']
    models=dict(u3040={k:old[k] for k in ('model_json','model_sha256')},
        u4002=dict(model_json=parent['model_json'],model_sha256=parent['model_sha256']))
    models['u3040']['policy_sha256']=old['source']['policy_sha256']
    models['u4002']['policy_sha256']=parent['policy_sha256']
    value=dict(id='R4-CARRIER-PRESSURE-V1',status='FROZEN',models=models,seed=2026336000,
        spawn_manifest=str(spawn),spawn_sha256=sha(spawn),games=256,seats='128 per physical seat, alternating',
        conditions=['interceptor','coop_hard'],smoke=dict(model='u3040',games=64,conditions=['interceptor','hunter_hard'],
            comparison='first64 fixed games; no outcome selection; <70% is expectation, not assumed result'),
        mode='greedy',engine='production JS Sim.step bun-native-v2 nativeItems/nativeTrap',teacher=False,
        candidate_cadence='dev80; full pressure at400 and endpoint, no peak selection',
        candidate_gate='interceptor P1 win-minus-loss >=parent-5 in matched256; no promotion from64 smoke',
        requested_arms=['K1','K2','K3','K4'],input_scope='public physical bot contract only',
        metrics='r4_pressure v1 plus existing causal ControlFunnel and full R3 item chains',
        unavailable=['matched no-item causal net benefit','full-chain exact glue counterfactual delay'],
        training_pool='no retroactive change; JAX parity required before separate forward pool version',
        resource=dict(cpu_only=True,workers=8,cores=list(range(176,184))),
        sources={f:sha(Path(f)) for f in ['web/bun_interceptor_bot.js','scripts/r4_pressure_metrics.js',
            'scripts/r4_interceptor_eval.js','web/bun_hunter_bot.js','web/bun_coop_hunter_bot.js','web/sim-bun-native-v2.js']})
    atomic_json(p,value);return value


def collect(model,opponent,start,p,core):
    target=OUT/'evaluation'/model/opponent/f'shard_{start:03d}.json';target.parent.mkdir(parents=True,exist_ok=True)
    guard=lock(Path(str(target)+'.lock'));accepted=Path(str(target)+'.ACCEPTED.json');side=Path(str(target)+'.outcome.json')
    if accepted.exists():
        assert all(sha(Path(f))==h for f,h in read(accepted)['files'].items());return
    for f,h in p['sources'].items():assert sha(Path(f))==h,f
    identity=p['models'][model];assert sha(Path(identity['model_json']))==identity['model_sha256']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        QQT_STOCK_RULE='stock-window-v1',BUN_V2_RULE='bun-native-v2',BUN_V2_TIER='hard',
        BUN_V2_OPPONENT='bun.coop_hunter' if opponent=='coop_hard' else 'bun.hunter',R4_PRESSURE_OPPONENT=opponent,
        JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='256',JE10_ROUTE_START=str(start),JE10_ROUTE_END=str(start+8),JE10_ROUTE_SEED_START=str(p['seed']))
    if not (target.exists() and side.exists()):
        with target.with_suffix('.log').open('ab') as log:
            subprocess.run(['taskset','-c',str(core),'node','scripts/r4_interceptor_eval.js',p['mode'],identity['model_json'],
                str(target),'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',p['spawn_manifest']],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    es=read(side)['episodes'];assert len(es)==8 and all(e['done'] and 'r4_pressure' in e and 'r3' in e for e in es)
    assert [e['seed'] for e in es]==list(range(p['seed']+start,p['seed']+start+8))
    assert [e['seat'] for e in es]==[(start+i)%2 for i in range(8)]
    assert read(target)['model_json_sha256']==identity['model_sha256']
    atomic_json(accepted,dict(status='PASS',games=8,files={str(f):sha(f) for f in [target,side]},unix=time.time()))


def summarize(es):
    results={}
    for seat in (0,1):
        games=[e for e in es if e['seat']==seat];carry=[a for e in games for a in e['r4_pressure']['possessions'][seat] if a['origin']==1-seat]
        value=dict(games=len(games),wins=sum(e['winner']==seat for e in games),losses=sum(e['winner']==1-seat for e in games),
            draws=sum(e['winner'] is None for e in games),self_deaths=sum(e['self_deaths'][seat] for e in games),
            native_acquisitions=len(carry),same_possession_delivered=sum(a['delivered'] for a in carry),
            carrier_causal_deaths=sum(a['causal_carrier_death'] for a in carry),
            reasons={k:sum(a['reason']==k for a in carry) for k in sorted({a['reason'] for a in carry})},
            mean_delivered_carry_ticks=sum(a['carry_ticks'] for a in carry if a['delivered'])/max(1,sum(a['delivered'] for a in carry)),
            glue_releases=sum(sum(r['type']==2 for r in a['releases']) for a in carry),
            actual_enemy_glue_contacts=sum(len(a['glue_enemy_contacts']) for a in carry),
            self_banana_contacts=sum(len(a['self_banana_contacts']) for a in carry),
            complete_self_slides=sum(len(a['complete_self_slides']) for a in carry),
            same_polyline_saved_ticks_estimate=sum(sum(s.get('saved_ticks_estimate') or 0 for s in a['complete_self_slides']) for a in carry),
            opponent_carrier_kills=sum(k['attacker']==seat and k['victim_carried']==seat for e in games for k in e['kills']),
            used_item={},no_item={})
        value['win_minus_loss']=value['wins']-value['losses']
        value['carry_success']=value['same_possession_delivered']/len(carry) if carry else None
        for used,key in [(True,'used_item'),(False,'no_item')]:
            group=[a for a in carry if a['used_item']==used];n=len(group);delivered=sum(a['delivered'] for a in group)
            value[key]=dict(acquisitions=n,delivered=delivered,success=delivered/n if n else None)
        results['P'+str(seat)]=value
    return results


def finish(model,opponent,count=256):
    folder=OUT/'evaluation'/model/opponent;records=[];files={}
    for start in range(0,count,8):
        f=folder/f'shard_{start:03d}.json';receipt=Path(str(f)+'.ACCEPTED.json')
        if not receipt.exists():return None
        r=read(receipt);assert all(sha(Path(n))==h for n,h in r['files'].items());files.update(r['files'])
        records+=read(str(f)+'.outcome.json')['episodes']
    assert len(records)==len({(e['seed'],e['seat']) for e in records})==count
    d=dict(status='COMPLETE',model=model,opponent=opponent,games=count,results=summarize(records),files=files,
        limitations=records[0]['r4_pressure']['limitations'],missing_data=records[0]['r4_pressure']['missing_data'],unix=time.time())
    atomic_json(folder/('SMOKE64.json' if count==64 else 'COMPLETE.json'),d);return d


def main():
    OUT.mkdir(parents=True,exist_ok=True);guard=lock(OUT/'EVAL_OWNER.lock');p=freeze()
    # First mandatory matched smoke, then full pressure baselines. Accepted
    # shards are reused exactly when extending the same condition to256.
    jobs=[('u3040',c,i) for c in ('interceptor','hunter_hard') for i in range(0,64,8)]
    jobs += [(m,c,i) for m in ('u3040','u4002') for c in p['conditions'] for i in range(0,256,8)
             if not(m=='u3040' and c=='interceptor' and i<64)]
    for offset in range(0,len(jobs),8):
        with ThreadPoolExecutor(max_workers=8) as pool:
            tasks=[pool.submit(collect,m,c,i,p,176+k) for k,(m,c,i) in enumerate(jobs[offset:offset+8])]
            for f in tasks:f.result()
        atomic_json(OUT/'EVAL_STATUS.json',dict(status='RUNNING',shards_accepted_through=offset+len(tasks),total_shards=len(jobs),unix=time.time()))
        if offset>=8 and not (OUT/'SMOKE_COMPARISON.json').exists():
            a=finish('u3040','interceptor',64);b=finish('u3040','hunter_hard',64)
            if a and b:
                atomic_json(OUT/'SMOKE_COMPARISON.json',dict(status='COMPLETE',interceptor=a['results'],hard=b['results'],
                    matched_seeds=True,performance_target_not_assumed=True,no_training_or_promotion_gate=True,unix=time.time()))
                subprocess.run(['jdme-notify','send','新拦截猎手与hard已完成u3040固定同seed各64完整局，真实分母与自伤见R4 interceptor/SMOKE_COMPARISON.json；继续256局运包压力集，不预设更强。'],capture_output=True)
    for m in p['models']:
        for c in p['conditions']:assert finish(m,c)
    atomic_json(OUT/'COMPLETE.json',dict(status='COMPLETE',baseline_games=1088,smoke_and_full_shards_reused=True,protocol_sha256=sha(OUT/'PRESSURE_PROTOCOL.json'),unix=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUT/'TECHNICAL_BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,reason=str(error),unix=time.time()));raise
