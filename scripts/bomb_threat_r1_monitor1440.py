"""Passive every40 holdout and native-paid monitoring, explicit missing data."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.bomb_threat_round15 import read
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1';OUT=BASE/'iter2/r1_attack_20261030/items_v2/monitor1440'


def actual_fraction(numerator,denominator):
    return numerator/denominator if denominator>0 else None


def refresh():
    from scripts.bomb_threat_eval import summarize
    rows=[]
    hands=BASE/'iter2/r1_attack_20261030/items_v2/handoffs'
    for hand in hands.glob('*.json'):
        h=read(hand);run=h['successor'];p=BASE/'formal'/run
        if read(p/'EXCLUDED_TECHNICAL.json'):continue
        history=p/'updates.jsonl'
        if not history.exists():continue
        updates=[json.loads(l) for l in history.read_text().splitlines()]
        for cm in sorted(p.glob('u*/COMMITTED.json')):
            u=int(cm.parent.name[1:])
            if u%40 or u<=h['source']['update']:continue
            target=OUT/run/f'u{u:06d}.json'
            previous=read(target,{})
            if previous.get('holdout_complete'):rows.append(previous);continue
            interval=[x['exposure'] for x in updates if u-40<x['new_update']<=u]
            total=sum(x['total_reward'] for x in interval);absolute=sum(x['absolute_total_reward'] for x in interval)
            item=sum(x.get('r1_item_pay',0) for x in interval);attack=sum(x.get('r1_attack_pay',0) for x in interval)
            rawapproach=sum(x.get('approach',0) for x in interval)
            new_component='r1_approach_pay' in interval[-1] if interval else False
            approach=sum(x.get('r1_approach_pay',0) for x in interval) if new_component else None
            threat=sum(x.get('r1_legacy_threat_pay',0) for x in interval) if new_component else None
            nat=dict(total_reward=total,absolute_total_reward=absolute,item_paid=item,attack_bundle_paid=attack,total_shaping_paid=item+attack,
                shaping_over_signed_total=actual_fraction(item+attack,total),item_over_signed_total=actual_fraction(item,total),
                shaping_over_absolute_total=actual_fraction(item+attack,absolute),item_over_positive_shaping=actual_fraction(item,max(0,attack)+item),
                approach_paid=approach,threat_paid=threat,approach_raw_signed_candidate=rawapproach,
                approach_over_total=None if approach is None else actual_fraction(approach,total),threat_over_total=None if threat is None else actual_fraction(threat,total),
                source='actual on-policy collector logs; Bank and natural mixed',steps=len(interval)*262144,
                missing=[] if new_component else ['legacy frozen worker does not retain per-component paid queue attribution; raw approach is not actual pay', 'native reward numerators by seat unavailable in frozen worker logs'],
                item_pending_measurement='per-tick outstanding balance sum is exposure, never cumulative created reward')
            nat['paid_by_seat']={field:[sum(x.get(field+'_by_seat',[0.,0.])[seat] for x in interval) for seat in (0,1)] for field in ('r1_legacy_pay','r1_legacy_threat_pay','r1_legacy_chain_pay','r1_legacy_topology_pay','r1_approach_pay','r1_item_pay','r1_additions_pay','r1_total_shaping_pay')} if new_component else None
            if new_component:
                den=[sum(x.get('r1_total_reward_by_seat',[0.,0.])[seat] for x in interval) for seat in (0,1)]
                nat['total_reward_by_seat']=den
                nat['paid_over_total_by_seat']={field:[actual_fraction(v[seat],den[seat]) for seat in (0,1)] for field,v in nat['paid_by_seat'].items()}
            evaluation=BASE/'eval'/f'{run}_u{u:06d}_holdout';data={};files={}
            for mode in ('greedy','categorical'):
                rr=[]
                for f in sorted(evaluation.glob(f'natural_{mode}_[0-9][0-9][0-9].json.outcome.json')):
                    x=read(f)
                    if len(x.get('episodes',[]))!=8:continue
                    assert not x['teacher'] and x['engine']=='production_JS'
                    rr+=x['episodes'];files[str(f.relative_to(ROOT))]=sha(f)
                if len(rr)==64 and len({r['seed'] for r in rr})==64 and all(r['done'] for r in rr):data[mode]=summarize(rr)
            value=dict(status='COMPLETE_PASSIVE' if len(data)==2 else 'AWAITING_CPU_HOLDOUT',run=run,slot=h['slot'],update=u,
                holdout_complete=len(data)==2,holdout=data,native_reward=nat,files=files,protocol_sha256=sha(BASE/'protocols'/f'{run}.json'),unix=time.time())
            atomic_json(target,value);rows.append(value)
            # Conservative item guard uses frozen parent and actual paid share.
            reference=read(BASE/'eval/r1_reference_slot05_u2320_dev/NATURAL_COMPLETE.json',{}).get('natural_js_full2400',{}).get('greedy',{})
            item_share=nat['item_over_positive_shaping']
            if item_share is not None and item_share>.6 and 'greedy' in data:
                baseline=read(BASE/'eval/r1_reference_slot05_u2320_holdout/NATURAL_COMPLETE.json',{}).get('natural_js_full2400',{}).get('greedy',{})
                cur=data['greedy']
                if baseline and cur['causal_kills']<=baseline['causal_kills'] and cur['win_minus_loss']<=baseline['win_minus_loss']:
                    proto=read(BASE/'protocols'/f'{run}.json');q=ROOT/proto['r1_item_coefficient_request']
                    if not q.exists():
                        atomic_json(q,dict(protocol_id='A3SR-ATTACK-R1-ITEMS-V2',factor=.5,red=True,split='holdout',update=u,share=item_share,comparison='no gain in kills or W-L vs frozen same-split parent',unix=time.time()))
            if approach is not None and 'greedy' in data:
                positive=sum(max(0,x.get('r1_approach_pay',0))+max(0,x.get('r1_item_pay',0))+max(0,x.get('r1_multi_pay',0))+max(0,x.get('r1_extra_chain_pay',0))+max(0,x.get('r1_legacy_pay',0)) for x in interval)
                approach_share=actual_fraction(max(0,approach),positive)
                nat['approach_over_positive_shaping']=approach_share
                baseline=read(BASE/'eval/r1_reference_slot05_u2320_holdout/NATURAL_COMPLETE.json',{}).get('natural_js_full2400',{}).get('greedy',{})
                cur=data['greedy']
                if approach_share is not None and approach_share>.6 and baseline and cur['causal_kills']<=baseline['causal_kills'] and cur['win_minus_loss']<=baseline['win_minus_loss']:
                    proto=read(BASE/'protocols'/f'{run}.json');q=ROOT/proto['r1_coefficient_request']
                    if not q.exists():
                        atomic_json(q,dict(protocol_id='A3SR-ATTACK-R1-20261030',factor=.5,red=True,split='holdout',update=u,approach_share=approach_share,apply='new addition ledger only; frozen legacy untouched',unix=time.time()))
                        from scripts.iter_pbt_owner import notify
                        notify('R1_APPROACH_HALF_'+run+str(u),'slot03新增接近实付占正塑形>60%且holdout击杀/W-L未涨；按已授权规则请求未来新项减半，旧A3SR奖励不变。')
            if len(data)==2:
                before=[r for r in rows if r['run']==run and r['update']<u and 'greedy' in r['holdout']]
                if before:
                    prev=max(before,key=lambda x:x['update'])['holdout']['greedy'];cur=data['greedy']
                    if cur['draw_fraction']>prev['draw_fraction'] and cur['bombs_per_game']<prev['bombs_per_game']:
                        alarm=OUT/'alerts'/f'{run}_u{u:06d}.json'
                        if not alarm.exists():
                            atomic_json(alarm,dict(status='OBSERVATIONAL_ALERT_NOT_PROVEN_HACK',condition='draw fraction up and bombs/game down',current=cur,previous=prev,
                                missing_causal_evidence=['close-range-without-bomb temporal linkage','circle/net approach causality','item detour or delay causality'],unix=time.time()))
                            from scripts.iter_pbt_owner import notify
                            notify('R1_1440_TREND_'+run+str(u),'R1 holdout出现平局率上升且泡/局下降：已写趋势警报；贴脸/绕圈/刷道具因果尚无充分证据，不伪称已证实hack。')
            atomic_json(target,value)
    atomic_json(OUT/'STATUS.json',dict(status='RUNNING_PASSIVE',points=len(rows),completed_holdout=sum(r['holdout_complete'] for r in rows),
        monitoring_limits='old running worker per-component paid numerator and seat-native reward are DATA_GAP; JS never reconstructs training reward',unix=time.time()))
