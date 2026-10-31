"""Bounded R4 CPU evaluations, adopt orphan shards and never block training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from scripts.population_run_state import atomic_json,lock,sha

BASE=Path('runs/r4_combat_20261031');OUT=BASE/'candidate_evaluation'
CORES=(160,161,162,163)


def read(p,default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default


def freeze():
    path=OUT/'PROTOCOL.json'
    if path.exists():return read(path)
    old=read('runs/r3_items_route_20261030/OWNSIDE_EVALUATION_PROTOCOL.json')
    d=dict(status='FROZEN',id='R4-K1-K3-CPU-EVAL-V1',teacher=False,
        source='same committed checkpoint exports, native full games and clear-brick bun combat separately',
        dev=old['dev'],holdout=old['holdout'],fresh=old['fresh'],
        cadence=dict(dev='absolute80 saved boundary >=first branch update+40',
            full='absolute400 boundary and exact640-new-update endpoint'),
        dev_games=64,full_holdout_games=256,clear_per_tier_games=64,fresh_per_tier_games=256,
        modes=['greedy','categorical'],full_fresh_modes=['greedy'],
        parent=read(BASE/'EFFECTIVE_PROTOCOL.json')['parent'],
        pressure_protocol=str(BASE/'interceptor/PRESSURE_PROTOCOL.json'),
        pressure='same256 seeds per interceptor/coop at full endpoints; independent owner queue',
        kill_arena='BLOCKED until full deterministic and fixed-item parity; do not substitute bun-clear results',
        resource=dict(cpu_only=True,cores=CORES,workers=4),no_publication=True,
        guard='two consecutive comparable full endpoints: cage kills, draw kill difference, bombs without kills, self harm, zero bomb; observation flags only, no short-score early termination',
        sources={n:sha(n) for n in ['scripts/r4_combat_eval.js','scripts/r4_combat_metrics.js',
            'scripts/r3_items_candidate_eval.js','scripts/r3_items_inference.js','web/sim-bun-native-v2.js']},unix=time.time())
    atomic_json(path,d);return d


def requests():
    p=freeze();rr=[dict(key='parent_dev',source=p['parent'],kind='dev',priority=2),
        dict(key='parent_full',source=p['parent'],kind='full',priority=4)]
    for arm in ('k1','k3'):
        out=BASE/'formal'/('r4_'+arm+'_c000000')
        for f in sorted((out/'eval_queue').glob('*.json'),reverse=True):
            s=read(f);u=s['update']
            if u>=4042 and u%80==0:
                rr.append(dict(key=out.name+f'_u{u:06d}_dev',source=s,kind='dev',priority=1))
            if u>=4042 and u%400==0:
                rr.append(dict(key=out.name+f'_u{u:06d}_full',source=s,kind='full',priority=3))
        cp=read(out/'COMPLETE.json')
        if cp:rr.append(dict(key=out.name+f"_u{cp['update']:06d}_full",source=cp,kind='full',priority=0))
    return sorted(rr,key=lambda r:(r['priority'],-r['source']['update'],r['key']))


def specs(req,p):
    if req['kind']=='dev':conditions=[dict(p['dev'],name='natural_hard',tier='hard',scene='natural',games=64)]
    else:
        conditions=[dict(p['holdout'],name='holdout_hard',tier='hard',scene='natural',games=256)]
        for tier,v in p['fresh'].items():
            conditions += [dict(v,name='fresh_'+tier,tier=tier,scene='natural',games=256),
                dict(v,name='clear_'+tier,tier=tier,scene='clear',games=64)]
    return [dict(c,mode=m,start=i) for c in conditions
        for m in (['greedy'] if c['name'].startswith('fresh_') else ['greedy','categorical'])
        for i in range(0,c['games'],8)]


def model(req):
    dest=OUT/req['key']/'model';dest.mkdir(parents=True,exist_ok=True);proof=dest/'IDENTITY.json'
    if proof.exists():
        d=read(proof);assert sha(d['model_json'])==d['model_sha256'];return d
    cp=Path(req['source']['checkpoint']);assert sha(cp/'policy.pt')==req['source']['policy_sha256']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (dest/'EXPORT.log').open('ab') as log:
        subprocess.run(['taskset','-c','164-167',sys.executable,'-m','scripts.r3_items_export',str(cp),str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    f=dest/'model.json';assert read(f)['meta']['checkpoint_sha256']==req['source']['policy_sha256']
    d=dict(model_json=str(f),model_sha256=sha(f),source=req['source']);atomic_json(proof,d);return d


def path(req,job):return OUT/req['key']/job['name']/job['mode']/f"shard_{job['start']:03d}.json"


def collect(req,job,identity,p,core):
    f=path(req,job);f.parent.mkdir(parents=True,exist_ok=True);guard=lock(Path(str(f)+'.lock'))
    receipt=Path(str(f)+'.ACCEPTED.json')
    if receipt.exists():
        assert all(sha(n)==h for n,h in read(receipt)['files'].items());return
    from scripts.r3_items_eval_owner import running_shard
    # Entry is distinct: detect orphan native node invocations explicitly.
    def inflight():
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                a=(proc/'cmdline').read_bytes().split(b'\0')
                if b'scripts/r4_combat_eval.js' in a and str(f).encode() in a:return True
            except OSError:pass
        return False
    while inflight():time.sleep(3)
    assert all(sha(n)==h for n,h in p['sources'].items())
    assert sha(job['spawn_manifest'])==job['spawn_sha256']
    side=Path(str(f)+'.outcome.json')
    if not (f.exists() and side.exists()):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            QQT_STOCK_RULE='stock-window-v1',BUN_V2_RULE='bun-native-v2',BUN_V2_TIER=job['tier'],BUN_V2_OPPONENT='bun.hunter',
            R4_COMBAT_SCENE=job['scene'],JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES=str(job['games']),
            JE10_ROUTE_START=str(job['start']),JE10_ROUTE_END=str(job['start']+8),JE10_ROUTE_SEED_START=str(job['seed']))
        with f.with_suffix('.log').open('ab') as log:
            subprocess.run(['taskset','-c',str(core),'node','scripts/r4_combat_eval.js',job['mode'],identity['model_json'],str(f),
                'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',job['spawn_manifest']],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    d=read(side);es=d['episodes']
    assert len(es)==8 and all(e['done'] and 'r4' in e for e in es)
    assert d['mode']==job['mode'] and not d['teacher'] and d['rule_version']=='bun-native-v2'
    assert [e['seed'] for e in es]==list(range(job['seed']+job['start'],job['seed']+job['start']+8))
    assert [e['seat'] for e in es]==[(job['start']+i)%2 for i in range(8)]
    assert read(f)['model_json_sha256']==identity['model_sha256']
    atomic_json(receipt,dict(status='PASS',games=8,files={str(n):sha(n) for n in (f,side)},job=job,unix=time.time()))


def finish(req,jobs):
    proof=OUT/req['key']/'COMPLETE.json'
    if proof.exists():return True
    groups={};files={};seen=set()
    for job in jobs:
        f=path(req,job);a=read(str(f)+'.ACCEPTED.json')
        if not a:return False
        for n,h in a['files'].items():assert sha(n)==h;files[n]=h
        for e in read(str(f)+'.outcome.json')['episodes']:
            key=(job['name'],job['mode'],e['seed'],e['seat']);assert key not in seen;seen.add(key)
            groups.setdefault(job['name'],{}).setdefault(job['mode'],[]).append(e)
    from scripts.r3_items_eval_owner import summarize
    atomic_json(proof,dict(status='COMPLETE',source=req['source'],kind=req['kind'],games=len(seen),files=files,
        metrics={c:{m:{f'P{s}':summarize(es,s) for s in (0,1)} for m,es in mm.items()} for c,mm in groups.items()},
        limitation='bun clear-brick is not no-cage kill arena; kill arena remains gated',unix=time.time()))
    subprocess.run(['jdme-notify','send','R4异步评估完成 '+req['key']+'；同父分母、greedy/categorical及原始SHA已保存，不发布。'],capture_output=True)
    return True


def main():
    OUT.mkdir(parents=True,exist_ok=True);guard=lock(OUT/'OWNER.lock');p=freeze();models={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        active={}
        while True:
            for fut in list(active):
                if fut.done():fut.result();del active[fut]
            rr=requests()
            for req in rr:
                if (OUT/req['key']/'COMPLETE.json').exists():continue
                jobs=specs(req,p)
                for job in jobs:
                    f=path(req,job)
                    if Path(str(f)+'.ACCEPTED.json').exists() or str(f) in [v[0] for v in active.values()]:continue
                    if len(active)>=4:break
                    if req['key'] not in models:models[req['key']]=model(req)
                    core=next(c for c in CORES if c not in [v[1] for v in active.values()])
                    active[pool.submit(collect,req,job,models[req['key']],p,core)]=(str(f),core)
                finish(req,jobs)
                if len(active)>=4:break
            atomic_json(OUT/'STATUS.json',dict(status='CPU_ASYNC',workers=len(active),requests=len(rr),unix=time.time()))
            if all((BASE/'formal'/('r4_'+a+'_c000000')/'COMPLETE.json').exists() for a in ('k1','k3')) and all((OUT/r['key']/'COMPLETE.json').exists() for r in rr):
                atomic_json(OUT/'COMPLETE.json',dict(status='COMPLETE',requests=len(rr),unix=time.time()));return
            time.sleep(5)


if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUT/'TECHNICAL_BLOCKED.json',dict(status='BLOCKED',error=type(error).__name__,reason=str(error),unix=time.time()));raise
