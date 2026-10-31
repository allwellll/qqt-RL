"""Asynchronous native-v2 leaderboard; separate from the immutable v1 queue."""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
from scripts.bun_native_v2_diagnostic import collect,path
from scripts.bun_native_v2_owner import OUT,read,notification
OWN_SIDE=Path('runs/v2_ownside_selection_20261031/EVALUATION_SWITCH.json').exists()
EVAL_DIR='evaluation_ownside_v1' if OWN_SIDE else 'evaluation'
PROTOCOL_FILE='R2_OWNSIDE_EVALUATION_PROTOCOL.json' if OWN_SIDE else 'R2_EVALUATION_PROTOCOL.json'


def export(source,key):
    record=OUT/'eval_models'/f'{key}.json'
    if record.exists():
        result=read(record);assert sha(Path(result['model_json']))==result['model_sha256'];return result
    cp=Path(source['checkpoint']);assert sha(cp/'policy.pt')==source['policy_sha256']
    dest=OUT/'eval_exports'/key;dest.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    code='from pathlib import Path;from scripts.je08_bun_authority import export;import sys;export(Path(sys.argv[1]),Path(sys.argv[2]).resolve())'
    with (dest/'EXPORT.log').open('ab') as log:
        subprocess.run(['taskset','-c','172-175',sys.executable,'-c',code,str(cp/'policy.pt'),str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    f=dest/'model.json';assert read(f)['meta']['checkpoint_sha256']==source['policy_sha256']
    result=dict(source,model_json=str(f),model_sha256=sha(f));atomic_json(record,result)
    return result


def specifications(key,source,kind,proto):
    if kind=='dev':conditions={'dev':dict(proto['dev'],tier='hard',games=64)};modes=['greedy','categorical']
    elif kind=='holdout':conditions={'holdout':dict(proto['holdout'],tier='hard',games=128)};modes=['greedy','categorical']
    else:conditions={t:dict(v,tier=t,games=256) for t,v in proto['fresh'].items()};modes=['greedy']
    return [dict(eval_key=key,evaluation_directory=EVAL_DIR,rule='bun-native-v2',condition=c,opponent='bun.hunter',start=i,mode=mode,**v)
            for c,v in conditions.items() for mode in modes for i in range(0,v['games'],8)]


def summary(rows):
    out={}
    for seat in (0,1):
        es=[e for e in rows if e['seat']==seat]
        kills=[k for e in es for k in e['kills'] if k['attacker']==seat]
        farm=sum(k['victim_carried']>=0 or k['within3_attacker_home'] or k['in_attacker_cage'] for k in kills)
        carrier_kills=sum(k['victim_carried']>=0 for k in kills)
        enemy_pickups=sum(e['antifarm']['stock_pickups'][1-seat] for e in es)
        total=lambda k:sum(e['antifarm'][k][seat] for e in es)
        out[f'P{seat}']=dict(games=len(es),wins=sum(e['winner']==seat for e in es),losses=sum(e['winner']==1-seat for e in es),
            draws=sum(e['winner'] in (None,-1) for e in es),win_minus_loss=sum((e['winner']==seat)-(e['winner']==1-seat) for e in es),
            unilateral_zero_bomb=sum(e['placed'][seat]==0 and e['placed'][1-seat]>0 for e in es),
            zero_bomb=sum(e['placed'][seat]==0 for e in es),bombs=sum(e['placed'][seat] for e in es),
            opponent_delivery_losses=sum(e['winner']==1-seat and any(v['type']=='delivery' and
                v['player']==1-seat and v['origin']==seat and v['tick']==e['ticks'] for v in e['antifarm']['events']) for e in es),
            self_deaths=sum(e['self_deaths'][seat] for e in es),deaths=sum(e['deaths'][seat] for e in es),
            causal_kills=len(kills),farm_union=farm,farm_fraction=farm/len(kills) if kills else None,
            enemy_stock_pickups=enemy_pickups,enemy_stock_pickups_per_game=enemy_pickups/len(es) if es else None,
            carrier_kills=carrier_kills,carrier_kills_per_game=carrier_kills/len(es) if es else None,
            carrier_kills_per_enemy_stock_pickup=carrier_kills/enemy_pickups if enemy_pickups else None,
            active_stock_pickups=total('stock_pickups'),
            active_deliveries=sum(v['type']=='delivery' and v['player']==seat and v['origin']==1-seat for e in es for v in e['antifarm']['events']),
            all_origin_deposits=total('deliveries'),enemy_cage_entries=total('enemy_cage_entries'),
            alive_ticks=total('alive_ticks'),near_home_ticks=total('near_home_ticks'),actionable_ticks=total('actionable_ticks'),
            near_home_alive_fraction=total('near_home_ticks')/total('alive_ticks') if total('alive_ticks') else None)
    return out


def gate(candidate,parent):
    decisions={}
    for mode in ('greedy','categorical'):
        c=candidate['metrics'].get('holdout',{}).get(mode,{})
        b=parent['metrics'].get('holdout',{}).get(mode,{})
        for seat in ('P0','P1'):
            if seat not in c or seat not in b:continue
            x,y=c[seat],b[seat]
            scale=64/max(1,x['games'])
            delta={k:(x[k]-y[k])*scale for k in ('win_minus_loss','unilateral_zero_bomb','self_deaths')}
            trend_ok=delta['win_minus_loss']>=-6 and delta['unilateral_zero_bomb']<=3 and delta['self_deaths']<=3
            delivery_ok=x['active_deliveries']>0 and (y['active_deliveries']==0 or x['active_deliveries']>=.5*y['active_deliveries'])
            farm_ok=x['farm_fraction'] is not None and y['farm_fraction'] is not None and x['farm_fraction']<=y['farm_fraction']
            # Same protocol and same seat count: all P1 zero-bomb games,
            # not only unilateral zero-bomb and not a per64 approximation.
            assert x['games']==y['games'],'gate needs matched sample counts'
            zero_ok=seat!='P1' or x['zero_bomb']<=y['zero_bomb']+3
            custody_data=all(k in z for z in (x,y) for k in ('enemy_stock_pickups','carrier_kills','wins'))
            farming=custody_data and x['enemy_stock_pickups']>1.5*y['enemy_stock_pickups'] and x['carrier_kills']>y['carrier_kills'] and x['wins']<=y['wins']
            custody_ok=custody_data and not farming
            decisions[mode+'_'+seat]=dict(trend_guard=trend_ok,active_delivery_guard=delivery_ok,farm_guard=farm_ok,delta_per64=delta,
                custody_antifarm_guard=custody_ok,custody_data_available=custody_data,custody_farming_detected=farming,
                P1_zero_bomb_guard=zero_ok,zero_bomb_candidate=x['zero_bomb'],zero_bomb_parent=y['zero_bomb'],
                farm_candidate=x['farm_fraction'],farm_parent=y['farm_fraction'],status='PASS' if trend_ok and delivery_ok and farm_ok and zero_ok and custody_ok else 'FAIL')
    return dict(status='PASS' if len(decisions)==4 and all(d['status']=='PASS' for d in decisions.values()) else 'NOT_QUALIFIED',
        comparisons=decisions,not_statistical_superiority=True,no_promotion_or_publication=True)


def requests():
    reg=read(OUT/'R2_REGISTRATION.json')
    if not reg:return []
    source=reg['common_parent'];out=[]
    for kind in ('holdout','dev','fresh'):
        out.append(dict(key='common_parent_'+kind,source=source,kind=kind,
                        priority={'holdout':0,'dev':1,'fresh':4}[kind]))
    for folder in (OUT/'formal').glob('r2_*'):
        p=read(OUT/'protocols'/f'{folder.name}.json',{})
        if not p:continue
        if OWN_SIDE and p.get('spawn_side_version')!='own-side-native-v1':continue
        accepted=sorted((folder/'eval_queue').glob('*.json'),reverse=True)
        latest=accepted[0].name if accepted else None
        for req in accepted:
            m=read(req);u=m['update']
            if req.name==latest or u%80==0:
                out.append(dict(key=folder.name+f'_u{u:06d}_dev',source=m,kind='dev',priority=3 if req.name==latest else 5))
            if u%40==0:
                out.append(dict(key=folder.name+f'_u{u:06d}_holdout',source=m,kind='holdout',priority=2 if req.name==latest else 6))
        done=read(folder/'COMPLETE.json')
        if done:
            out.append(dict(key=folder.name+'_fresh',source=done,kind='fresh',priority=1))
            out.append(dict(key=folder.name+f"_u{done['update']:06d}_holdout",source=done,kind='holdout',priority=1))
    return sorted(out,key=lambda r:(r['priority'],-r['source']['update'],r['key']))


def finish(req,jobs,proto):
    f=OUT/EVAL_DIR/req['key']/'COMPLETE.json'
    if f.exists():return read(f)
    rows=[];files={}
    for job in jobs:
        accepted=Path(str(path(job))+'.ACCEPTED.json')
        if not accepted.exists():return
        record=read(accepted)
        for filename,h in record['files'].items():assert sha(Path(filename))==h;files[filename]=h
        for e in read(str(path(job))+'.outcome.json')['episodes']:rows.append((job['condition'],job['mode'],e))
    unique={(c,m,e['seed'],e['seat']) for c,m,e in rows};assert len(unique)==len(rows)
    metrics={}
    for c,m,e in rows:
        metrics.setdefault(c,{}).setdefault(m,[]).append(e)
    metrics={c:{m:summary(es) for m,es in modes.items()} for c,modes in metrics.items()}
    result=dict(status='COMPLETE',kind=req['kind'],source=req['source'],games=len(rows),metrics=metrics,files=files,
        protocol_sha256=sha(OUT/PROTOCOL_FILE),unix=time.time())
    atomic_json(f,result)
    if req['kind']=='holdout' and not req['key'].startswith('common_parent'):
        parent=read(OUT/'evaluation/common_parent_holdout/COMPLETE.json')
        if parent:atomic_json(f.parent/'GATE.json',gate(result,parent))
    if req['kind']=='fresh' and not req['key'].startswith('common_parent'):
        notification(req['key']+'_FRESH','bun-native-v2 '+req['key']+' 首轮fresh三档768完整局已完成，P0/P1分列、反刷与主动交付单独记账；结果见独立v2 evaluation目录，R1继续、网页规则未变。',dict(result=str(f),games=len(rows)))
    write_metrics_report()
    return result


def upgrade_custody_metrics():
    """Recompute from saved outcome shards only; never rerun an episode."""
    guard=lock(OUT/'CUSTODY_METRICS_MIGRATION.lock')
    for f in sorted((OUT/EVAL_DIR).glob('*/COMPLETE.json')):
        result=read(f)
        if result.get('custody_metrics_version')==2:continue
        groups={};seen=set()
        for name,h in result['files'].items():
            if not name.endswith('.outcome.json'):continue
            path=Path(name);assert sha(path)==h
            mode=path.parent.name;condition=path.parent.parent.name
            assert mode in ('greedy','categorical')
            for e in read(path)['episodes']:
                k=(condition,mode,e['seed'],e['seat']);assert k not in seen;seen.add(k)
                groups.setdefault(condition,{}).setdefault(mode,[]).append(e)
        assert len(seen)==result['games']
        prior=f.parent/'PRE_CUSTODY_MONITORING_COMPLETE.json'
        if not prior.exists():atomic_json(prior,result)
        result.update(metrics={c:{m:summary(es) for m,es in modes.items()} for c,modes in groups.items()},
            custody_metrics_version=2,recomputed_from_saved_outcomes=True)
        atomic_json(f,result)
    parent=read(OUT/'evaluation/common_parent_holdout/COMPLETE.json')
    if parent:
        for f in (OUT/EVAL_DIR).glob('*/COMPLETE.json'):
            result=read(f)
            if result['kind']=='holdout' and not f.parent.name.startswith('common_parent'):
                atomic_json(f.parent/'GATE.json',gate(result,parent))


def write_metrics_report():
    """Recomputable local scoreboard; pending gates never masquerade as PASS."""
    lines=['# bun-native-v2 R2 异步评估 — 20261030','',
        '仅v2同协议比较；起点为20:50编排指定u2721，非旧网页诊断模型或原u2760。P1全部零泡门禁=同协议父点+3，保留原单方零泡护栏。敌交付负局须敌源包在终局tick送达；己笼驻留分母为存活tick。','',
        '|请求|update|集/难度|模式|席位|局数|胜/平/负|全部零泡|单方零泡|敌交付负局|己笼3格/存活tick|驻留比例|主动敌源交付|刷分并集/因果击杀|holdout门禁|',
        '|---|---:|---|---|---|---:|---|---:|---:|---:|---|---|---:|---|---|']
    for f in sorted((OUT/EVAL_DIR).glob('*/COMPLETE.json')):
        result=read(f);g=read(f.parent/'GATE.json',{})
        for condition,modes in result['metrics'].items():
            for mode,seats in modes.items():
                for seat,m in seats.items():
                    decision=g.get('comparisons',{}).get(mode+'_'+seat,{}).get('status','PENDING/NA')
                    fraction=m.get('near_home_alive_fraction')
                    fraction='unknown' if fraction is None else f'{fraction:.6f}'
                    lines.append(f"|{f.parent.name}|{result['source']['update']}|{condition}|{mode}|{seat}|{m['games']}|{m['wins']}/{m['draws']}/{m['losses']}|{m['zero_bomb']}|{m['unilateral_zero_bomb']}|{m['opponent_delivery_losses']}|{m['near_home_ticks']}/{m['alive_ticks']}|{fraction}|{m['active_deliveries']}|{m['farm_union']}/{m['causal_kills']}|{decision}|")
    lines+=['','## 归属反刷监控（同版本、席位、动作模式；原始分母）','',
        '|请求|版本|模式|席位|局数|敌取己包/局|击杀持包者/局|击杀/敌取包|较前点敌取包/局变化|主动敌源交付|胜/平/负|反刷门禁|',
        '|---|---|---|---|---:|---:|---:|---|---|---:|---|---|']
    previous={}
    results=[(f,read(f)) for f in (OUT/EVAL_DIR).glob('*/COMPLETE.json')]
    for f,result in sorted(results,key=lambda x:(x[1]['source']['update'],x[0].parent.name)):
        if result['kind']!='holdout':continue
        run=Path(result['source']['checkpoint']).parent.name;p=read(OUT/'protocols'/f'{run}.json',{})
        version=p.get('r2_reward_version','original-v2' if p else 'common-parent')
        lineage=(p.get('r2_arm','parent'),version);g=read(f.parent/'GATE.json',{})
        for mode,seats in result['metrics'].get('holdout',{}).items():
            for seat,m in seats.items():
                if 'enemy_stock_pickups_per_game' not in m:continue
                key=(*lineage,mode,seat);prior=previous.get(key);v=m['enemy_stock_pickups_per_game']
                trend='NA' if prior is None else f'{v-prior:+.6f}';previous[key]=v
                ratio=m['carrier_kills_per_enemy_stock_pickup'];ratio='NA (0敌取包)' if ratio is None else f'{ratio:.6f}'
                guard=g.get('comparisons',{}).get(mode+'_'+seat,{}).get('custody_antifarm_guard','NA')
                lines.append(f"|{f.parent.name}|{version}|{mode}|{seat}|{m['games']}|{v:.6f}|{m['carrier_kills_per_game']:.6f}|{ratio}|{trend}|{m['active_deliveries']}|{m['wins']}/{m['draws']}/{m['losses']}|{guard}|")
    target=OUT/('R2_OWNSIDE_METRICS_20261031.md' if OWN_SIDE else 'R2_METRICS_20261030.md');temp=target.with_suffix('.md.tmp')
    temp.write_text('\n'.join(lines)+'\n');temp.replace(target)


def main():
    guard=lock(OUT/'R2_EVAL_OWNER.lock');proto=read(OUT/PROTOCOL_FILE)
    upgrade_custody_metrics()
    models={};active={};cores=list(range(172,176)) if OWN_SIDE else list(range(144,160))
    workers=4 if OWN_SIDE else 16
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            for future in list(active):
                if future.done():future.result();del active[future]
            pending=[]
            for req in requests():
                key=req['key'];jobs=specifications(key,req['source'],req['kind'],proto)
                if (OUT/EVAL_DIR/key/'COMPLETE.json').exists():continue
                if finish(req,jobs,proto):continue
                pending.append(req)
                if len(active)>=workers:continue
                if key not in models:models[key]=export(req['source'],key)
                for job in jobs:
                    if len(active)>=workers:break
                    if Path(str(path(job))+'.ACCEPTED.json').exists() or any(job==v[0] for v in active.values()):continue
                    core=next(c for c in cores if c not in [v[1] for v in active.values()])
                    active[pool.submit(collect,job,dict(model=models[key]),core)]=(job,core)
            atomic_json(OUT/'R2_EVALUATION_STATUS.json',dict(status='RUNNING',pending_requests=len(pending),active=[v[0] for v in active.values()],
                GPU_training_blocked=False,cpu_only=True,unix=time.time()))
            time.sleep(10)


if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic_json(OUT/'R2_EVAL_BLOCKED.json',dict(status='BLOCKED',error_type=type(e).__name__,reason='inspect R2_EVAL_OWNER.log; GPU remains independent',unix=time.time()))
        raise
