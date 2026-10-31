"""Latest six A3/control dev checkpoints by actual lineage and physical seat."""
import json,time,statistics
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
B=Path('runs/bomb_threat_v1')


def kill_efficiency(candidate,controls):
    """Passive rate per100 actual placements, not action requests or means of ratios."""
    def rate(m):
        bombs=m.get('actual_bombs')
        return 100*m['causal_kills']/bombs if bombs and bombs>0 else None
    own=rate(candidate);rates={s:rate(m) for s,m in controls.items()}
    pooled=100*sum(m['causal_kills'] for m in controls.values())/sum(m['actual_bombs'] for m in controls.values()) if len(controls)==2 and all(v is not None for v in rates.values()) else None
    return dict(kills_per100_actual_bombs=own,controls_per100=rates,pooled_control_per100=pooled,
        below70percent=own<.7*pooled if own is not None and pooled is not None else None)


def low_rate_triplets(points):
    return [points[i-2:i+1] for i in range(2,len(points))
        if all(p['below70percent'] is True for p in points[i-2:i+1])
        and points[i]['update']-points[i-1]['update']==points[i-1]['update']-points[i-2]['update']==40]


def main():
    from scripts.bomb_threat_gate_report import recompute_natural
    records=[]
    for p in (B/'eval').glob('*/NATURAL_COMPLETE.json'):
        marker=json.loads(p.read_text());cp=marker.get('checkpoint')
        if not cp or marker.get('split')!='dev' or marker.get('seed')!=2026122900:continue
        cp=Path(cp);run=cp.parent.name;proto=B/'protocols'/f'{run}.json'
        if not proto.exists() or not (cp/'COMMITTED.json').exists():continue
        cfg=json.loads(proto.read_text());slot=cfg.get('slot');arm=cfg.get('iter2_arm');update=int(cp.name[1:])
        if slot not in (0,1,3,4,7) or update%40:continue
        if slot in (1,4,7) and arm!='A3':continue
        if slot in (0,3) and arm is not None:continue
        if any((d/f).exists() for d in (cp.parent,p.parent) for f in ('EXCLUDED_EVALUATION.json','EXCLUDED_INVALID.json','INVALID_STARTING_POINT.json')):continue
        cm=json.loads((cp/'COMMITTED.json').read_text());assert cm['policy_sha256']==marker['policy_sha256']
        _,m=recompute_natural(p.parent)
        records.append(dict(slot=slot,arm=arm,update=update,checkpoint=str(cp),train_unix=cm['metrics']['unix'],
            marker=str(p),marker_sha256=sha(p),metrics=m['greedy']))
    latest={};dedup={}
    for r in sorted(records,key=lambda x:x['train_unix']):dedup[(r['slot'],r['update'])]=r
    records=list(dedup.values())
    for slot in (1,4,7,0,3):latest[str(slot)]=sorted([r for r in records if r['slot']==slot],key=lambda r:r['train_unix'])[-6:]
    rate_points=[]
    for r in sorted([r for r in records if r['slot']==1],key=lambda r:r['train_unix']):
        matched={}
        for s in (0,3):
            cs=[c for c in records if c['slot']==s and 0<=r['train_unix']-c['train_unix']<=600]
            if cs:matched[str(s)]=max(cs,key=lambda c:c['train_unix'])
        rate_points.append(dict(update=r['update'],checkpoint=r['checkpoint'],marker=r['marker'],marker_sha256=r['marker_sha256'],
            actual_bombs=r['metrics']['actual_bombs'],causal_kills=r['metrics']['causal_kills'],
            matched_markers={s:c['marker'] for s,c in matched.items()},
            **kill_efficiency(r['metrics'],{s:c['metrics'] for s,c in matched.items()})))
    rate_triples=low_rate_triplets(rate_points)
    rate_observation=dict(status='REPORT_ONLY_LOW_RATE_3_POINTS' if rate_triples else 'NO_THREE_KNOWN_CONSECUTIVE_LOW_RATE',
        points=rate_points,trigger_triplets=rate_triples,automatic_rollback=False,automatic_recipe_change=False,
        definition='causal kills/actual placements*100; controls00+03 pooled actual counts; latest preceding commits<=600s; missing/zero denominator unknown; contiguous40 updates')
    # Existing matched-control window, using00/03 per fresh steer. Compare
    # whole64 metrics for warning; never infer missing seats or controls as0.
    points=[]
    for r in sorted([r for r in records if r['slot']==4],key=lambda x:x['train_unix']):
        matched={}
        for s in (0,3):
            candidates=[c for c in records if c['slot']==s and 0<=r['train_unix']-c['train_unix']<=600]
            if candidates:matched[str(s)]=max(candidates,key=lambda c:c['train_unix'])
        control={k:statistics.mean(c['metrics'][k] for c in matched.values()) if len(matched)==2 else None for k in ('bombs_per_game','causal_kills')}
        m=r['metrics'];qualifies=m['bombs_per_game']>control['bombs_per_game']+5 and m['causal_kills']<=control['causal_kills'] if len(matched)==2 else None
        points.append(dict(update=r['update'],checkpoint=r['checkpoint'],train_unix=r['train_unix'],bombs=m['bombs_per_game'],kills=m['causal_kills'],control=control,
            matched_markers={s:c['marker'] for s,c in matched.items()},qualifies=qualifies))
    triples=[points[i-2:i+1] for i in range(2,len(points)) if all(p['qualifies'] is True for p in points[i-2:i+1]) and points[i]['update']-points[i-1]['update']==points[i-1]['update']-points[i-2]['update']==40]
    slots=json.loads((B/'SLOTS.json').read_text());s4=slots['4'];old_replaced=s4.get('iter2_arm')=='A3S'
    observation=dict(status='TRIGGERED_OLD_ARM_SUPERSEDED_BY_AUTHORIZED_A3S' if triples and old_replaced else 'TRIGGER_AT_NEXT_ATOMIC_BOUNDARY' if triples else 'NO_THREE_KNOWN_CONSECUTIVE_TRIGGER',
        points=points,trigger_triplets=triples,active_slot04=s4['run'],active_arm=s4.get('iter2_arm'),
        disposition='old A3 slot04 completed640 and new A3S initialized at full chunk boundary; old warning not transferred to new recipe' if old_replaced else 'read-only; existing owner rollback remains authoritative')
    value=dict(status='PASSIVE_OBSERVATION_COMPLETE',latest_six=latest,slot04_observation=observation,slot01_kill_efficiency=rate_observation,definitions='dev productionJS greedy64; per-seat32; zero=own actual0 enemy>0; draw kill diff=sum own-enemy causal kills only drawn games',unix=time.time())
    o=B/'fresh256_slot01_u800';o.mkdir(exist_ok=True);atomic_json(o/'RECENT6_AND_SLOT04.json',value)
    text='# ITER-2 第8轮观察：fresh256与A3三个seed（20261029）\n\n'
    text+='本轮只读观察，A3S按17:20批准执行。下表每个dev完整点64局，P0/P1各32；argmax分开列出，不用categorical替代。严格按训练提交时间取各slot最近6个已完成评估点，三条A3与slot00/03对照，未评最新点不补零。checkpoint可追溯SHA见`runs/bomb_threat_v1/fresh256_slot01_u800/RECENT6_AND_SLOT04.json`。\n\n'
    text+='|slot/配方|checkpoint|seat/局数|胜-负|因果击杀|泡/局|单方零泡|平局击杀差|\n|---|---|---|---:|---:|---:|---:|---:|\n'
    for slot,rows in latest.items():
        for r in rows:
            for seat in ('P0','P1'):
                m=r['metrics']['by_seat'][seat]
                text+=f"|{slot}/{r['arm'] or 'control'}|{Path(r['checkpoint']).parent.name}/{Path(r['checkpoint']).name}|{seat}/{m['games']}|{m['win_minus_loss']}|{m['causal_kills']}|{m['bombs_per_game']:.3f}|{m['zero_bomb_games']}|{m['draw_kill_difference']}|\n"
        for seat in ('P0','P1'):
            if not rows:continue
            mm=[r['metrics']['by_seat'][seat] for r in rows]
            text+=f'|{slot}/最近{len(mm)}点均值|不同训练点重复固定dev集，非独立新局|{seat}|'+'|'.join(f'{statistics.mean(m[k] for m in mm):.3f}' for k in ('win_minus_loss','causal_kills','bombs_per_game','zero_bomb_games','draw_kill_difference'))+'|\n'
    text+='\nslot04连续三点监控：**'+observation['status']+'**。匹配00/03最近不晚于候选的训练点且各≤600秒，缺失记unknown；全64局检验“泡>对照+5且击杀≤对照”，逐座位表仅观察。\n\n'
    text+=f'发现{len(triples)}组三连；当前slot04={s4["run"]}/{s4.get("iter2_arm")}。'+observation['disposition']+'。未改任何奖励/权重。\n\n'
    text+='B fresh256状态另见`reports/bomb_threat_v1_fresh256_20261029.md`。未完成不判PASS；禁止晋升BEST/发布。\n'
    text+='\n19:45修订：A3S固定u640，dev及holdout各自P1胜-负+5、P0下降≤3、总单方零泡不增；不放宽。暂不PBT复制，slot00/03对照保留，等待编排决定。\n\n'
    text+='slot01击杀/百泡只观察：'+rate_observation['status']+'。分母为真实放置数；对照按00/03实际总击杀/总放泡合并，单独列两路。连续40间隔三点低于70%仅通知，不自动回退。缺参照或零分母为unknown。\n\n'
    text+=f"历史三连{len(rate_triples)}组；末次历史触发更新{[p['update'] for p in rate_triples[-1]] if rate_triples else []}；最近三点触发={bool(low_rate_triplets(rate_points[-3:]))}。历史与当前分列，不把早期预警当作最新触发。\n\n"
    text+='|slot01点|真实击杀/真实泡数|击杀/百泡|00对照|03对照|合并对照|低于70%|\n|---|---|---:|---:|---:|---:|---|\n'
    for r in rate_points[-6:]:
        text+=f"|{r['checkpoint']}|{r['causal_kills']}/{r['actual_bombs']}|{r['kills_per100_actual_bombs']}|{r['controls_per100'].get('0')}|{r['controls_per100'].get('3')}|{r['pooled_control_per100']}|{r['below70percent']}|\n"
    Path('reports/bomb_threat_v1_iter2_round8_fresh256_20261029.md').write_text(text)
    if rate_triples:
        last=rate_triples[-1][-1];name='SLOT01_KILL_PER100_LOW_'+Path(last['checkpoint']).parent.name+'_'+str(last['update'])
        q=B/'notifications'/(name+'.json')
        if not q.exists():
            atomic_json(q,dict(status='REPORT_ONLY',triplet=rate_triples[-1],no_automatic_rollback=True,unix=time.time()))
            from scripts.iter_pbt_owner import notify
            notify(name,'A3 slot01连续3点评估击杀/百泡低于00/03合并对照70%；仅上报观察，未改配方或回退，训练继续。')
    print(json.dumps(dict(status=value['status'],rows={k:len(v) for k,v in latest.items()},slot04=observation['status'],triggers=len(triples))))

if __name__=='__main__':main()
