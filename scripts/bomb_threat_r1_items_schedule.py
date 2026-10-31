"""Consume the revised item contract through the existing sole GPU owner."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.bomb_threat_round15 import read,pin
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'
OLD=BASE/'iter2/r1_attack_20261030';OUT=OLD/'items_v2';TARGETS=(0,3,4,7)
MODULE='scripts.bomb_threat_r1_progress_train'
NAMESPACE='A3SR-ATTACK-R1-ITEMS-V2'


def apply_switch(slot,entry,chunk):
    if slot not in TARGETS or entry.get('r1_smoke_fallback'):return None
    if entry.get('r1_items_applied'):
        failed=BASE/'formal'/entry['run']
        if not read(failed/'EXCLUDED_TECHNICAL.json') or list(failed.glob('u*/COMMITTED.json')):return None
    ready=read(OUT/'READY_FOR_BOUNDARY.json',{})
    if ready.get('status')!='PASS':return None
    for f,h in ready['prerequisites'].items():assert sha(OUT/f)==h,'new item gate changed'
    previous=BASE/'formal'/entry['run'];done=read(previous/'COMPLETE.json',{})
    replaced_smoke=bool(read(previous/'EXCLUDED_TECHNICAL.json',{})) and entry.get('r1_applied')
    if not done and not replaced_smoke:return None
    if done:
        assert sha(ROOT/done['checkpoint']/'policy.pt')==done['policy_sha256']
    frozen=read(OLD/'PROTOCOL.json');source=frozen['source'];donor=Path(source['checkpoint']).parent.name
    template=read(BASE/'protocols'/f'{donor}.json');recipe=dict(template['arms'][donor])
    attack=bool(frozen['arms'][str(slot)]['attack']);seed=frozen['seeds'][str(slot)]
    weights=[.05,.25,.25,.20,.15,.10] if slot==4 else [.05,.35,.30,.15,.10,.05]
    recipe.update(r1_attack=attack,r1_bank_fix=True,opponent_weights=weights,r1_arm='BANK_ONLY' if not attack else 'ATTACK_SELFPLAY45' if slot==4 else 'ATTACK')
    name=f'bt_slot{slot:02d}_c{chunk:06d}';path=BASE/'protocols'/f'{name}.json';assert not path.exists()
    seal=read(OUT/'SOURCE_SEAL.json');assert all(sha(ROOT/f)==h for f,h in seal['files'].items())
    hashes=dict(template['source_sha256']);assert all(sha(ROOT/f)==h for f,h in hashes.items())
    hashes.update(seal['files'])
    proto=dict(template)
    for key in ('round14_authorization','round15_authorization','round16_authorization','a3sr_promotion_authorization'):
        if key in proto:proto['ancestor_'+key]=proto.pop(key)
    proto.update(namespace_id=NAMESPACE,arms={name:recipe},sources={str(seed):source},source_sha256=hashes,
        continuation_source=source,chunk_start_update=source['update'],chunk_updates=640,environment_steps_per_run=(source['update']+640)*262144,
        slot=slot,chunk=chunk,worker_module=MODULE,r1_attack=attack,r1_seed=seed,A1_weights=weights,
        r1_bank_manifest=str((OUT/'BANK_WITH_HISTORY.json').relative_to(ROOT)),r1_bank_manifest_sha256=sha(OUT/'BANK_WITH_HISTORY.json'),
        r1_parent_export=str((OUT/'PARENT_EXPORT.json').relative_to(ROOT)),
        r1_coefficient_request=str((OUT/'coefficient_requests'/f'slot{slot:02d}.json').relative_to(ROOT)),
        r1_item_coefficient_request=str((OUT/'item_coefficient_requests'/f'slot{slot:02d}.json').relative_to(ROOT)),
        reward_version='r1-attack-cap030-independent-item-highwater024-v2' if attack else template['reward_version'],
        rule_version='r1-native-items-stock-window-v2',r1_authorization=str((OUT/'AMENDMENT03.json').relative_to(ROOT)),
        pbt_lineage=dict(source=source,Actor_Critic_Adam_ring_exact=True,env_aux='fresh natural episode boundary with native typed inventory',RNG='independent seed',old_controls_unchanged=[5,1,6]))
    atomic_json(path,proto)
    atomic_json(OUT/'handoffs'/f'{name}.json',dict(status='BOUNDARY_HANDOFF',previous_run=entry['run'],previous_final=done or read(previous/'EXCLUDED_TECHNICAL.json'),successor=name,slot=slot,source=source,unix=time.time()))
    pin([str((BASE/'formal'/name/f'u{u:06d}').relative_to(ROOT)) for u in range(source['update']+40,source['update']+641,40)],'R1 item-v2 fixed budget checkpoints')
    return dict(run=name,chunk=chunk,gpu=slot,source=source,role='R1_ITEMS_V2_FIXED640',iter2_arm='A3SR',r1_applied=True,r1_items_applied=True,r1_arm=recipe['r1_arm'],authorized_shaping_override=True,supersedes=entry['run'])


def refresh():
    rows=[]
    for hand in sorted((OUT/'handoffs').glob('*.json')):
        h=read(hand);run=h['successor'];p=read(BASE/'protocols'/f'{run}.json');out=BASE/'formal'/run
        if read(out/'EXCLUDED_TECHNICAL.json'):continue
        commits=sorted(out.glob('u*/COMMITTED.json'));latest=read(commits[-1],{}) if commits else {}
        updates=[json.loads(l) for l in (out/'updates.jsonl').read_text().splitlines()] if (out/'updates.jsonl').exists() else []
        valid=read(out/'SMOKE.json',{}).get('passed') and read(out/'RESTORE.json',{}).get('passed')
        formal=[x for x in updates if x['new_update']>h['source']['update']+2]
        first=OUT/'first_updates'/f'{run}.json'
        if valid and formal and not first.exists():
            x=formal[0];assert x['finite'] and min(x['update_l2'].values())>0
            atomic_json(first,dict(status='PASS',run=run,slot=h['slot'],update=x['new_update'],environment_steps=x['actual_environment_steps_this_update'],
                update_l2=x['update_l2'],finite=True,env_per_second=262144/x['cycle_seconds'],latest_atomic=latest,unix=time.time()))
            from scripts.iter_pbt_owner import notify
            notify('R1_ITEMS_FIRST_'+run,f'R1新规则slot{h["slot"]:02d}正式u{x["new_update"]} finite，Actor/Critic非零，262144真实环境步；原生道具独立high-water奖励，其他对照不变。')
        for cm in commits:
            u=int(cm.parent.name[1:])
            if u%40 or u<=h['source']['update']:continue
            for split in ('dev','holdout'):
                name=f'{run}_u{u:06d}_{split}';req=BASE/'eval_requests'/f'{name}.json'
                if not req.exists():atomic_json(req,dict(name=name,checkpoint=str(cm.parent.relative_to(ROOT)),checkpoint_sha256=read(cm)['policy_sha256'],run=run,update=u,slot=h['slot'],split=split,
                    initial='none',r1_metrics=True,r1_items_v2=True,iter2_arm='A3SR',reward_version=p['reward_version'],training_rule_version=p['rule_version'],training_namespace=p['namespace_id'],reason='item-v2 every40 fixed cadence, CPU async'))
        if p['r1_attack']:
            from scripts.bomb_threat_r1_schedule import monitor_gate
            points=[];item_points=[]
            baseline=read(BASE/'eval/r1_reference_slot05_u2320_dev/NATURAL_COMPLETE.json',{}).get('natural_js_full2400',{}).get('greedy',{})
            for cm in commits:
                u=int(cm.parent.name[1:])
                if u%40 or u<=h['source']['update']:continue
                name=f'{run}_u{u:06d}_dev';evaluation=BASE/'eval'/name
                summaries=[]
                for f in evaluation.glob('natural_greedy_*.json.r1_metrics.json'):
                    extra=read(f,{})
                    if extra.get('version')!='r1-passive-bomb-lifetime-items-v3' or extra.get('source_sha256')!=sha(ROOT/'scripts/bomb_threat_r1_progress_eval.js'):continue
                    summaries.extend(extra['episodes'])
                if len(summaries)!=64 or len({x['seed'] for x in summaries})!=64 or not all(x['done'] for x in summaries):continue
                den=sum(x['learner']['resolved_bombs'] for x in summaries)
                point=dict(update=u,empty_bomb_fraction=sum(x['learner']['empty_bombs'] for x in summaries)/den if den else 0.,
                    kills=sum(x['learner']['kills'] for x in summaries),self_harm=sum(x['learner']['self_harm'] for x in summaries))
                points.append(point)
                marker=read(evaluation/'NATURAL_COMPLETE.json',{}).get('natural_js_full2400',{}).get('greedy',{})
                interval=[x['exposure'] for x in updates if u-40<x['new_update']<=u]
                paid=sum(x.get('r1_item_pay',0.) for x in interval);attack=sum(max(0.,x.get('r1_attack_pay',0.)) for x in interval)
                share=paid/(paid+attack) if paid+attack>0 else None
                item_points.append(dict(update=u,item_paid=paid,attack_positive_paid=attack,item_share=share,
                    kills=marker.get('causal_kills'),win_minus_loss=marker.get('win_minus_loss'),
                    denominator='actual on-policy interval paid item plus positive attack shaping; not JS reward reconstruction'))
                if baseline and marker and share is not None and share>.6 and marker['causal_kills']<=baseline['causal_kills'] and marker['win_minus_loss']<=baseline['win_minus_loss']:
                    request=ROOT/p['r1_item_coefficient_request']
                    if not request.exists():
                        atomic_json(request,dict(protocol_id='A3SR-ATTACK-R1-ITEMS-V2',factor=.5,red=True,point=item_points[-1],baseline=baseline,
                            apply='future progress events at next atomic40 boundary; highwater and historical/pending balances preserved',unix=time.time()))
                        from scripts.iter_pbt_owner import notify
                        notify('R1_ITEMS_WEIGHT_HALF_'+run,'R1道具支付份额>60%且相对冻结父dev击杀/W-L未涨：已标红并请求独立W减半，训练不停。')
            gate=monitor_gate(points)
            atomic_json(OUT/'anti_hack'/f'{run}.json',dict(**gate,points=points,item_points=item_points,baseline_available=bool(baseline),unix=time.time()))
            if gate['status']=='HALVE':
                request=ROOT/p['r1_coefficient_request']
                if not request.exists():
                    atomic_json(request,dict(protocol_id='A3SR-ATTACK-R1-20261030',factor=.5,gate=gate,points=points,unix=time.time()))
                    from scripts.iter_pbt_owner import notify
                    notify('R1_ITEMS_ATTACK_HALF_'+run,'R1进攻防hack触发，已请求原子40边界仅进攻系数减半；独立道具W由自己的份额门禁管理，训练不停。')
        done=read(out/'COMPLETE.json',{});term=OUT/'terminal'/f'{run}.json'
        if done and not term.exists():
            assert done['update']==h['source']['update']+640
            assert sha(ROOT/done['checkpoint']/'state.pkl')==done['state_sha256']
            continuation=read(OLD/'continuation_1805/AUTHORIZATION.json',{}).get('status')=='AUTHORIZED'
            next_action='same-configuration640 continuation authorized; persistent stop gates take priority' if continuation else 'bounded budget complete, no automatic extension'
            atomic_json(term,dict(status='COMPLETE_FIXED640',run=run,last_atomic=done,new_updates=640,new_environment_steps=167772160,next_action=next_action,unix=time.time()))
            from scripts.iter_pbt_owner import notify
            next_text='按18:05授权同配置自动续640，持久停止优先。' if continuation else '无自动扩预算。'
            notify('R1_ITEMS_TERMINAL_'+run,f'R1道具v2 {run} 固定640更新原子完成；CPU评估继续，'+next_text)
        rows.append(dict(run=run,slot=h['slot'],update=updates[-1]['new_update'] if updates else latest.get('update'),last_atomic_update=latest.get('update'),smoke=read(out/'SMOKE.json',{}).get('passed'),restore=read(out/'RESTORE.json',{}).get('passed'),first_formal=first.exists(),complete=bool(done)))
    newest={r['slot']:r for r in rows}
    status='ALL_FOUR_COMPLETE_CPU_EVAL_ASYNC' if len(newest)==4 and all(r['complete'] for r in newest.values()) else 'RUNNING' if rows else 'PREPARING_REVISED_GATES'
    atomic_json(OUT/'EXECUTION_STATUS.json',dict(status=status,arms=rows,old_controls_unchanged=[5,1,6],unix=time.time()))
    atomic_json(OUT/'TASK_CONSUMED.json',dict(status=status,
        instruction='.hermes/TASK_a3sr_attack_r1_20261030.md revised item5 plus native death correction',
        first_formal_runs=[r['run'] for r in rows if r['first_formal']],
        complete_runs=[r['run'] for r in rows if r['complete']],
        owner='scripts.bomb_threat_owner',old_controls_unchanged=[5,1,6],training_worker_signals=0,publication=False,unix=time.time()))
