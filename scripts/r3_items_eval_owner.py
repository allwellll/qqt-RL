"""CPU-only resumable R3 full-game evaluation, never a GPU admission gate."""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha
from scripts.r3_items_owner import OUT, read, notify
OWN_SIDE=Path('runs/v2_ownside_selection_20261031/EVALUATION_SWITCH.json').exists()
EVAL_DIR='evaluation_ownside_v1' if OWN_SIDE else 'evaluation'
PROTOCOL_FILE='OWNSIDE_EVALUATION_PROTOCOL.json' if OWN_SIDE else 'EVALUATION_PROTOCOL.json'


def freeze():
    path=OUT/PROTOCOL_FILE
    if path.exists():return read(path)
    old=read('runs/bun_native_v2_20261030/R2_EVALUATION_PROTOCOL.json')
    value=dict(status='FROZEN',id='R3-ITEMS-ROUTE-EVALUATION-V1',
        dev=old['dev'],holdout=old['holdout'],fresh=old['fresh'],
        cadence='dev64 both modes every40; holdout128 both modes every80 and chunk endpoint; fresh3x256 greedy each640 endpoint',
        baseline='fixed revised R2-b first atomic u2722; identical JS engine, seeds, seats, metrics and modes',
        primary_mode='greedy; categorical separate',teacher=False,rule='bun-native-v2',
        critical_path='geodesic movement proxy, not intention or causal treatment benefit',
        banana='own carrier self acceleration; completed trajectory same-path native speed estimate; not enemy kill chain',
        resource=dict(cpu_cores=[184,185,186,187],export_cores='188-191',workers=4,cpu_only=True),
        guard=dict(self_death_increase_per_seat=3,empty_fraction_absolute_increase=.20,
                   self_glue_per_game=1,item_road_positive_shaping_share=.60),
        boundary_action='R3 safety failure blocks next chunk; health and missing CPU results never stop GPU',
        limitations=['production Sim.step, not human frameStep','single map806','preterminal JS metrics; native ledger tail reported separately'],
        source_sha256={f:sha(Path(f)) for f in ['scripts/r3_items_candidate_eval.js','scripts/r3_items_candidate_metrics.js',
            'scripts/r3_items_metrics.js','scripts/r3_items_inference.js','web/sim-bun-native-v2.js']},unix=time.time())
    atomic_json(path,value);return value


def requests():
    reg=read(OUT/'R3_REGISTRATION.json')
    if not reg:return []
    out=[dict(key='parent_'+k,source=reg['common_parent'],kind=k,priority=0) for k in ('dev','holdout','fresh')]
    for folder in (OUT/'formal').glob('r3_*'):
        if OWN_SIDE and read(OUT/'protocols'/f'{folder.name}.json',{}).get('spawn_side_version')!='own-side-native-v1':continue
        queue=sorted((folder/'eval_queue').glob('*.json'),reverse=True)
        for i,q in enumerate(queue):
            r=read(q);u=r['update'];key=folder.name+f'_u{u:06d}'
            out.append(dict(key=key+'_dev',source=r,kind='dev',priority=2 if i==0 else 6))
            if u%80==0:
                reclaim=read(Path('runs/r3_selfplay_ramp_20261031/RECLAIM_1820_AUTHORIZATION.json'),{})
                priority=-1 if u==4000 and reclaim.get('frozen_controls')=={'a':4002,'b':4002} else 1 if i==0 else 5
                out.append(dict(key=key+'_holdout',source=r,kind='holdout',priority=priority))
        done=read(folder/'COMPLETE.json')
        if done:
            for k in ('holdout','fresh'):out.append(dict(key=folder.name+'_'+k,source=done,kind=k,priority=1))
    return sorted(out,key=lambda r:(r['priority'],-r['source']['update'],r['key']))


def specifications(req,p):
    k=req['kind']
    conditions={t:dict(v,tier=t,games=256) for t,v in p['fresh'].items()} if k=='fresh' else {
        k:dict(p[k],tier='hard',games=64 if k=='dev' else 128)}
    return [dict(condition=c,mode=m,start=i,**v) for c,v in conditions.items()
        for m in (['greedy'] if k=='fresh' else ['greedy','categorical']) for i in range(0,v['games'],8)]


def model_for(req):
    dest=OUT/EVAL_DIR/req['key']/'model';dest.mkdir(parents=True,exist_ok=True)
    receipt=dest/'IDENTITY.json'
    if receipt.exists():
        d=read(receipt);assert sha(Path(d['model_json']))==d['model_sha256'];return d
    cp=Path(req['source']['checkpoint']);assert sha(cp/'policy.pt')==req['source']['policy_sha256']
    code=('from pathlib import Path;from scripts.je08_bun_authority import export;import sys;export(Path(sys.argv[1])/"policy.pt",Path(sys.argv[2]).resolve())'
        if req['key'].startswith('parent_') else 'from scripts.r3_items_export import export;import sys;export(*sys.argv[1:])')
    with (dest/'EXPORT.log').open('ab') as log:subprocess.run(['taskset','-c','188-191',sys.executable,'-c',code,str(cp),str(dest)],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
        stdout=log,stderr=subprocess.STDOUT,check=True)
    f=dest/'model.json';assert read(f)['meta']['checkpoint_sha256']==req['source']['policy_sha256']
    d=dict(model_json=str(f),model_sha256=sha(f),source=req['source']);atomic_json(receipt,d);return d


def job_path(req,job):
    return OUT/EVAL_DIR/req['key']/job['condition']/job['mode']/f"shard_{job['start']:03d}.json"


def running_shard(target):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            a=(p/'cmdline').read_bytes().split(b'\0')
            if b'scripts/r3_items_candidate_eval.js' in a and str(target).encode() in a and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return True
        except OSError:pass
    return False


def collect(req,job,model,core):
    target=job_path(req,job);target.parent.mkdir(parents=True,exist_ok=True)
    guard=lock(Path(str(target)+'.lock'));accepted=Path(str(target)+'.ACCEPTED.json')
    if accepted.exists():
        assert all(sha(Path(f))==h for f,h in read(accepted)['files'].items());return
    side=Path(str(target)+'.outcome.json');assert sha(Path(job['spawn_manifest']))==job['spawn_sha256']
    # An orphaned shard survives evaluator-owner restart; adopt before retry.
    while running_shard(target):
        time.sleep(3)
    if not (target.exists() and side.exists()):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            QQT_STOCK_RULE='stock-window-v1',BUN_V2_RULE='bun-native-v2',BUN_V2_TIER=job['tier'],BUN_V2_OPPONENT='bun.hunter',
            JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES=str(job['games']),JE10_ROUTE_START=str(job['start']),
            JE10_ROUTE_END=str(job['start']+8),JE10_ROUTE_SEED_START=str(job['seed']))
        with target.with_suffix('.log').open('ab') as log:subprocess.run(['taskset','-c',str(core),'node','scripts/r3_items_candidate_eval.js',
            job['mode'],model['model_json'],str(target),'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',job['spawn_manifest']],
            env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    doc=read(side);es=doc['episodes'];assert doc['rule_version']=='bun-native-v2' and not doc['teacher'] and doc['mode']==job['mode']
    assert read(target)['model_json_sha256']==model['model_sha256']
    assert len(es)==8 and all(e['done'] and 'resolved_bombs' in e['r3'] for e in es)
    assert [e['seed'] for e in es]==list(range(job['seed']+job['start'],job['seed']+job['start']+8))
    assert [e['seat'] for e in es]==[(job['start']+i)%2 for i in range(8)]
    atomic_json(accepted,dict(status='PASS',job=job,games=8,files={str(f):sha(f) for f in (target,side)},unix=time.time()))


def summarize(es,seat):
    from scripts.r3_items_report import summarize as detailed
    from scripts.bun_native_v2_eval_owner import summary as inherited
    out=dict(detailed(es,seat),**inherited(es)[f'P{seat}'])
    resolved=[b for e in es if e['seat']==seat for b in e['r3']['resolved_bombs'][seat]]
    out.update(empty_resolved_bombs=sum(b['empty'] for b in resolved),resolved_bombs=len(resolved),
               empty_fraction=sum(b['empty'] for b in resolved)/len(resolved) if resolved else None)
    return out


def gates(candidate,parent):
    comparisons={};blocked=False
    for mode in ('greedy','categorical'):
        cs=candidate['metrics']['holdout'][mode];bs=parent['metrics']['holdout'][mode]
        for seat in ('P0','P1'):
            c,b=cs[seat],bs[seat];assert c['games']==b['games']
            selfharm=c['self_deaths']-b['self_deaths']>=3
            empty=(c['empty_fraction'] is not None and b['empty_fraction'] is not None and
                c['empty_fraction']-b['empty_fraction']>=.20 and c['causal_kills']<=b['causal_kills'])
            glue=c['glue_self_contacts']/c['games']>1
            comparisons[mode+'_'+seat]=dict(selfharm=selfharm,empty_without_kill_gain=empty,self_glue=glue,
                empty_denominator=c['resolved_bombs'],games=c['games'],status='BLOCK' if selfharm or empty or glue else 'PASS')
            blocked|=selfharm or empty or glue
    from scripts.bun_native_v2_eval_owner import gate as inherited
    r2=inherited(candidate,parent)
    return dict(status='BLOCK' if blocked or r2['status']!='PASS' else 'PASS',r3=comparisons,r2=r2,
        action='block next chunk only; no publication',not_superiority_proof=True,unix=time.time())


def finish(req,jobs,p):
    dest=OUT/EVAL_DIR/req['key']/'COMPLETE.json'
    if dest.exists():return read(dest)
    groups={};files={};seen=set()
    for job in jobs:
        f=job_path(req,job);a=read(str(f)+'.ACCEPTED.json')
        if not a:return
        for name,h in a['files'].items():assert sha(Path(name))==h;files[name]=h
        for e in read(str(f)+'.outcome.json')['episodes']:
            key=(job['condition'],job['mode'],e['seed'],e['seat']);assert key not in seen;seen.add(key)
            groups.setdefault(job['condition'],{}).setdefault(job['mode'],[]).append(e)
    result=dict(status='COMPLETE',kind=req['kind'],source=req['source'],games=len(seen),files=files,
        metrics={c:{m:{f'P{s}':summarize(es,s) for s in (0,1)} for m,es in ms.items()} for c,ms in groups.items()},
        protocol_sha256=sha(OUT/PROTOCOL_FILE),unix=time.time())
    atomic_json(dest,result)
    if req['kind']=='holdout' and not req['key'].startswith('parent_'):
        parent=read(OUT/'evaluation/parent_holdout/COMPLETE.json')
        if parent:
            gate=gates(result,parent);atomic_json(dest.parent/'GATE.json',gate)
            if gate['status']=='BLOCK':
                run=Path(req['source']['checkpoint']).parent.name
                atomic_json(OUT/'boundary_blocks'/f'{run}.json',dict(gate=str(dest.parent/'GATE.json'),status='BLOCK',unix=time.time()))
                atomic_json(OUT/'boundary_blocks'/f'{run.split("_c")[0]}.json',dict(gate=str(dest.parent/'GATE.json'),status='BLOCK',unix=time.time()))
            native_guard(req,result,parent)
    if req['kind']=='fresh' and not req['key'].startswith('parent_'):
        notify(req['key']+'_FRESH','R3首轮fresh三档768完整生产JS局已完成；R2-b固定父同口径，P0/P1、greedy、道具/开路/真实链分母已保存；不发布。',result=str(dest))
    write_table()
    return result


def native_guard(req,result,parent):
    """Only actual native collector payment may trigger a coefficient request."""
    cp=Path(req['source']['checkpoint']);u=req['source']['update'];run=cp.parent.name
    history=cp.parent/'updates.jsonl'
    if not history.exists():return
    es=[r['exposure'] for r in map(json.loads,history.read_text().splitlines()) if u-80<r['new_update']<=u]
    assert all('r3_positive_pay' in e and 'r3_inherited_positive_pay' in e for e in es)
    items=sum(sum(e['r3_positive_pay'][:5]) for e in es)
    total=sum(sum(e['r3_positive_pay'])+e['r3_inherited_positive_pay'] for e in es)
    cg=result['metrics']['holdout']['greedy'];bg=parent['metrics']['holdout']['greedy']
    share=items/total if total>0 else None
    atomic_json(cp.parent/'NATIVE_SHAPING_RATIO.json',dict(item_road=items,total_positive_bundle=total,share=share,
        interval=[u-79,u],source='actual per-environment per-tick positive ledger settlement, accumulated before collector reduction; not JS reward reconstruction',
        limitation='inherited approach/topology settle as one existing signed ledger; positive part taken at its actual settlement',unix=time.time()))
    if share is not None and share>.60 and all(cg[s]['wins']<=bg[s]['wins'] for s in ('P0','P1')):
        protocol=read(OUT/'protocols'/f'{run}.json');control=Path(protocol['r3_item_coefficient_request'])
        if not control.exists():
            atomic_json(control,dict(protocol_id='R3-ITEMS-ROUTE-CHAIN-20261030',factor=.5,update=u,
                share=share,comparison='matched parent greedy holdout wins not increased in either seat',
                apply='next40 boundary, future candidates only, prior pending unchanged',unix=time.time()))
            notify(run+f'_SHARE_u{u}', 'R3道具/开路正实付份额>60%且双席胜局未增；按合同请求未来道具/开路系数减半，既有pending不重估，未改R1/R2。')


def write_table():
    columns=['五类拾取','香蕉/胶释放','终局泡/威/速','持包自踩香蕉','完整滑行省tick估算/30tick被炸',
        '胶敌触发/关键路径/50tick结果','两笼初终BFS/最少砖','自身因果路径砖',
        'live泡mean/P90/max','live批次mean/max','真连锁/批次数','0泡存活时间占比','空爆/实际爆炸','自踩胶/局']
    atomic_json(OUT/'REPLAY_TABLE_COLUMNS.json',dict(columns=columns,local_only=True,no_publish=True))
    lines=['','## R3异步候选评估（原始分片可复算）','','|请求|集|模式|席位|局数|胜/平/负|拾取五类|蕉/胶释放|持包自踩蕉|自踩胶|空爆/爆炸|真连锁|','|---|---|---|---|---:|---|---|---|---:|---:|---|---:|']
    for f in sorted((OUT/EVAL_DIR).glob('*/COMPLETE.json')):
        for c,ms in read(f)['metrics'].items():
            for m,ss in ms.items():
                for s,v in ss.items():lines.append(f"|{f.parent.name}|{c}|{m}|{s}|{v['games']}|{v['wins']}/{v['draws']}/{v['losses']}|{v['pickup_events']}|{v['release_counts']}|{v['carrying_own_banana_contacts']}|{v['glue_self_contacts']}|{v['empty_resolved_bombs']}/{v['resolved_bombs']}|{v['chain_events']}|")
    Path('reports/r3_ownside_evaluation_20261031.md' if OWN_SIDE else 'reports/r3_candidate_evaluation_20261030.md').write_text('\n'.join(lines)+'\n')


def main():
    guard=lock(OUT/'EVAL_OWNER.lock');p=freeze();assert all(sha(Path(f))==h for f,h in p['source_sha256'].items())
    with ThreadPoolExecutor(max_workers=4) as pool:
        active={};models={}
        while True:
            for fut in list(active):
                if fut.done():fut.result();del active[fut]
            rr=requests()
            for req in rr:
                jobs=specifications(req,p)
                if (OUT/EVAL_DIR/req['key']/'COMPLETE.json').exists():continue
                for job in jobs:
                    k=(req['key'],job['condition'],job['mode'],job['start'])
                    if Path(str(job_path(req,job))+'.ACCEPTED.json').exists() or any(v[0]==k for v in active.values()):continue
                    if len(active)>=4:break
                    if req['key'] not in models:models[req['key']]=model_for(req)
                    core=next(c for c in p['resource']['cpu_cores'] if c not in [v[1] for v in active.values()])
                    active[pool.submit(collect,req,job,models[req['key']],core)]=(k,core)
                finish(req,jobs,p)
                if len(active)>=4:break
            atomic_json(OUT/'EVAL_EXECUTION_STATUS.json',dict(status='RUNNING_CPU_ONLY',workers=len(active),
                requests=len(rr),completed=len(list((OUT/EVAL_DIR).glob('*/COMPLETE.json'))),unix=time.time()))
            time.sleep(5)


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(OUT/'EVAL_BLOCKED.json',dict(status='BLOCKED',reason='inspect EVAL_OWNER.log; GPU independent',unix=time.time()));raise
