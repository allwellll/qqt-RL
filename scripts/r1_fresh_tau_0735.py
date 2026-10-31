"""Frozen paired full-game CPU evaluation; never trains or publishes."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',QQT_STOCK_RULE='stock-window-v1')
import json,time,subprocess,statistics,sys,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from scripts.population_run_state import atomic_json,sha,lock
from scripts.bomb_threat_eval import summarize

BASE=Path('runs/bomb_threat_v1')
OUT=BASE/'fresh256_slot05_tau_20261031'
REPORT=Path('reports/r1_fresh256_slot05_20261031.md')
INSTRUCTION='.hermes/STEER_r1_fresh256_tau_20261031_0735.md'

def read(p):return json.loads(Path(p).read_text())

def freeze():
    p=OUT/'PROTOCOL.json'
    if p.exists():return read(p)
    OUT.mkdir(parents=True,exist_ok=True)
    slot=read(BASE/'SLOTS.json')['5'];folder=BASE/'formal'/slot['run']
    commits=sorted((x for x in folder.glob('u*/COMMITTED.json') if int(x.parent.name[1:])%40==0),key=lambda x:int(x.parent.name[1:]))
    # Consumption was observed before implementation; keep that exact point.
    cp=BASE/'formal/bt_slot05_c000045/u006640'
    assert cp/'COMMITTED.json' in commits and max(commits).parent.name>='u006640'
    cm=read(cp/'COMMITTED.json');assert cm['passed'] and sha(cp/'policy.pt')==cm['policy_sha256']
    old=read(BASE/'fresh256_a3sr_vs_u800_20261030/PROTOCOL.json')
    candidate=BASE/'eval/bt_slot05_c000045_u006640_holdout/model/model.json'
    assert read(candidate)['meta']['checkpoint_sha256']==cm['policy_sha256']
    models={'candidate':dict(checkpoint=str(cp),policy_sha256=cm['policy_sha256'],model_json=str(candidate),model_sha256=sha(candidate)),
            'baseline':old['models']['a3sr_u1280']}
    for m in models.values():assert sha(Path(m['model_json']))==m['model_sha256']
    tiers={}
    for tier,seed in zip(('hard','normal','easy'),(2026311000,2026312000,2026313000)):
        doc=read(old['tiers'][tier]['spawn_manifest']);doc['seed']=seed
        path=OUT/f'{tier}_spawns.json';atomic_json(path,doc)
        tiers[tier]=dict(seed=seed,spawn_manifest=str(path),spawn_sha256=sha(path),games=256)
    files=['scripts/r1_fresh_tau_0735.js','scripts/r1_fresh_tau_0735.py','scripts/bomb_threat_iter2_actions.js']+[f for f in old['source_sha256'] if f.endswith('.js')]
    proto=dict(status='FROZEN',instruction=INSTRUCTION,instruction_sha256=sha(Path(INSTRUCTION)),
        candidate_selection='latest complete multiple40 at first instruction consumption: c45/u6640, not a score-selected peak',
        models=models,tiers=tiers,conditions=[dict(role='candidate',mode='greedy',tau=0),dict(role='baseline',mode='greedy',tau=0),dict(role='candidate',mode='categorical',tau=.5)],
        source_sha256={f:sha(Path(f)) for f in files},games=2304,shard_size=8,engine='production_JS',opponent='production_JS_Hunter',rule='stock-window-v1',teacher=False,horizon=2400,bricks=True,
        seats=dict(P0=128,P1=128),workers=6,cpu_affinity='120-125',throughput_guard=.85,
        no_pbt=True,no_publish=True,no_decoder_change=True,
        limitations=['new seeds, reused64 birth geometries; not256 independent maps','baseline explicitly specified u1280, live website identity not inferred','tau half is diagnostic only, gate remains argmax'],unix=time.time())
    atomic_json(p,proto);atomic_json(OUT/'TASK_CONSUMED.json',dict(status='FROZEN_READY',protocol_sha256=sha(p),unix=time.time()))
    return proto

def training_snapshot():
    roots=[(BASE,'SLOTS.json'),(Path('runs/bun_native_v2_20261030'),'R2_SLOTS.json')];result={}
    for root,filename in roots:
        for slot,entry in read(root/filename).items():
            file=root/'formal'/entry['run']/'updates.jsonl'
            if not file.exists():continue
            rows=[json.loads(x) for x in file.read_text().splitlines()][-8:]
            if rows and time.time()-rows[-1]['unix']<180:
                result[f'{root.name}:{slot}']=dict(run=entry['run'],points=[r['new_update'] for r in rows],env_per_second=statistics.median(r['actual_environment_steps_this_update']/r['cycle_seconds'] for r in rows))
    return result

def jobs(p):
    return [dict(**c,tier=t,start=i) for t in p['tiers'] for i in range(0,256,8) for c in p['conditions']]

def job_path(j):return OUT/j['tier']/j['role']/('greedy' if j['tau']==0 else 'tau0.5')/f"shard_{j['start']:03d}.json"

def validate(j,p):
    f=job_path(j);a=read(f);b=read(str(f)+'.outcome.json');m=p['models'][j['role']];t=p['tiers'][j['tier']]
    assert a['model_json_sha256']==m['model_sha256'] and not a['assist']
    assert b['engine']=='production_JS' and not b['teacher'] and b['mode']==j['mode'] and b['rule_version']=='stock-window-v1'
    assert b['simulator_sha256']==p['source_sha256']['web/sim-stock-window-v1.js']
    rows=b['episodes'];expected=list(range(t['seed']+j['start'],t['seed']+j['start']+8))
    assert [r['seed'] for r in rows]==expected==a['seeds']
    assert len(rows)==8 and all(r['done'] and 0<r['ticks']<=2400 and r['seat']==(j['start']+i)%2 for i,r in enumerate(rows))
    assert read(str(f)+'.sampling.json')['tau']==j['tau']
    return rows

def collect(j,p):
    f=job_path(j);f.parent.mkdir(parents=True,exist_ok=True);proof=Path(str(f)+'.ACCEPTED.json')
    guard=lock(Path(str(f)+'.lock'))
    if proof.exists():
        assert all(sha(Path(k))==v for k,v in read(proof)['files'].items());return validate(j,p)
    if not all(Path(str(f)+s).exists() for s in ('','.outcome.json','.sampling.json')):
        env=dict(os.environ,FRESH_TAU=str(j['tau']),BT_FRESH_TIER=j['tier'],JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='256',JE10_ROUTE_START=str(j['start']),JE10_ROUTE_END=str(j['start']+8),JE10_ROUTE_SEED_START=str(p['tiers'][j['tier']]['seed']))
        with f.with_suffix('.log').open('ab') as log:
            subprocess.run(['node','scripts/r1_fresh_tau_0735.js',j['mode'],p['models'][j['role']]['model_json'],str(f),'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',p['tiers'][j['tier']]['spawn_manifest']],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=validate(j,p)
    atomic_json(proof,dict(job=j,protocol_sha256=sha(OUT/'PROTOCOL.json'),files={str(Path(str(f)+s)):sha(Path(str(f)+s)) for s in ('','.outcome.json','.sampling.json')},unix=time.time()))
    return rows

def paired(a,b):
    key=lambda r:(r['seed'],r['seat'])
    aa={key(r):r for r in a};bb={key(r):r for r in b};assert aa.keys()==bb.keys()
    score=lambda r:1 if r['winner']==r['seat'] else 0 if r['winner'] in (None,-1) else -1
    result=[]
    for k in sorted(aa):
        x,y=aa[k],bb[k];s=k[1]
        result.append(dict(seed=k[0],seat=s,candidate_score=score(x),baseline_score=score(y),score_difference=score(x)-score(y),
            causal_kill_difference=x['causal_kills'][s]-y['causal_kills'][s],bomb_difference=x['placed'][s]-y['placed'][s],death_difference=x['deaths'][s]-y['deaths'][s],self_death_difference=x['self_deaths'][s]-y['self_deaths'][s]))
    return result

def summarize_results(p):
    groups={};files={};done=0;pairings={}
    for t in p['tiers']:
        groups[t]={};raw={}
        for c in p['conditions']:
            label=c['role']+('_greedy' if c['tau']==0 else '_tau0.5');rows=[]
            for start in range(0,256,8):
                j=dict(**c,tier=t,start=start);f=job_path(j);proof=Path(str(f)+'.ACCEPTED.json')
                if not proof.exists():continue
                assert all(sha(Path(k))==v for k,v in read(proof)['files'].items())
                rows+=validate(j,p);files.update(read(proof)['files'])
            done+=len(rows)
            if rows:groups[t][label]=summarize(rows)
            if len(rows)==256:
                assert len({r['seed'] for r in rows})==256;raw[label]=rows
        if 'baseline_greedy' in raw:
            pairings[t]={label:paired(rows,raw['baseline_greedy']) for label,rows in raw.items() if label!='baseline_greedy'}
    result=dict(status='COMPLETE' if done==2304 else 'RUNNING',games=done,budget=2304,metrics=groups,paired_per_game=pairings,files=files,protocol_sha256=sha(OUT/'PROTOCOL.json'),unix=time.time())
    atomic_json(OUT/'RESULTS.json',result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    p=freeze()
    if args.freeze:return
    guard=lock(OUT/'OWNER.lock')
    assert all(sha(Path(f))==h for f,h in p['source_sha256'].items())
    if (OUT/'COMPLETE.json').exists():return
    before=read(OUT/'TRAINING_BEFORE.json') if (OUT/'TRAINING_BEFORE.json').exists() else training_snapshot();atomic_json(OUT/'TRAINING_BEFORE.json',before)
    try:
        workers=p['workers'];pending=jobs(p)
        while pending:
            group=pending[:workers];pending=pending[workers:]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for f in as_completed([pool.submit(collect,j,p) for j in group]):f.result()
            result=summarize_results(p);now=training_snapshot();ratios={k:v['env_per_second']/before[k]['env_per_second'] for k,v in now.items() if k in before and v['run']==before[k]['run']}
            if ratios and min(ratios.values())<.85:workers=max(1,workers//2)
            atomic_json(OUT/'EXECUTION_STATUS.json',dict(status='RUNNING',completed_games=result['games'],workers=workers,cpu_only=True,training_ratios=ratios,training_current=now,unix=time.time()))
        result=summarize_results(p);assert result['status']=='COMPLETE'
        atomic_json(OUT/'COMPLETE.json',dict(status='COMPLETE',games=2304,results_sha256=sha(OUT/'RESULTS.json'),unix=time.time()))
    except Exception as e:
        atomic_json(OUT/'BLOCKED.json',dict(status='BLOCKED',reason=type(e).__name__+': '+str(e),training_untouched=True,unix=time.time()));raise

if __name__=='__main__':main()
