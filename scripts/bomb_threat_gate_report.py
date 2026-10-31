"""Recompute revised gates from immutable, hashed natural-game evidence."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha,lock
from scripts.bomb_threat_selection import shaping_trend
from scripts.bomb_threat_gates_v2 import GATE_VERSION,evaluate
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'


def recompute_natural(out):
    from scripts.bomb_threat_eval import summarize
    marker=json.loads((out/'NATURAL_COMPLETE.json').read_text())
    assert marker['measurement_version']==2 and marker['rule_version']=='stock-window-v1'
    for file,digest in marker['files'].items():assert sha(out/file)==digest,(out.name,file)
    result={}
    for mode,values in marker['natural_js_full2400'].items():
        rows=[]
        for start in range(0,64,8):
            primary=json.loads((out/f'natural_{mode}_{start:03d}.json').read_text())['results']['natural_full2400']['episodes']
            side=json.loads((out/f'natural_{mode}_{start:03d}.json.outcome.json').read_text())
            assert side['measurement_version']==2 and len(side['episodes'])==len(primary)==8
            for e,r in zip(primary,side['episodes']):
                assert e['seed']==r['seed'] and e['team']==r['seat'] and e['ticks']==r['ticks'] and r['done']
                assert e['causal_kill']==r['causal_kills'][r['seat']]
                rows.append(r)
        assert len(rows)==len({r['seed'] for r in rows})==64
        measured=summarize(rows)
        result[mode]=dict(values,**{k:measured[k] for k in ('win_minus_loss','draw_own_causal_kills','draw_opponent_causal_kills','draw_kill_difference',
            'zero_bomb_games','structural_zero_bomb_games','raw_zero_bomb_games','opponent_bomb_opportunity_games','zero_bomb_measurement','by_seat')})
    return marker,result


def reward_monitor(cp,update):
    from scripts.bomb_threat_eval import native_reward_exposures
    exposures=native_reward_exposures(cp,update)
    extra=sum(e.get('chain_reward',0) for e in exposures)
    total=sum(e.get('total_reward',0) for e in exposures)
    absolute=sum(e.get('absolute_total_reward',0) for e in exposures)
    episodes=sum(e.get('resets',0) for e in exposures)
    terminal_available=bool(exposures) and all(len(e.get('native_parts',[]))==90 for e in exposures)
    terminal=sum(e['native_parts'][2]+e['native_parts'][9]+e['native_parts'][61] for e in exposures) if terminal_available else None
    return dict(threat_reward_fraction_abs_total=extra/absolute if absolute else None,
        threat_reward_fraction_total=extra/total if total>0 else None,
        threat_topology_reward=extra if exposures else None,total_reward=total if exposures else None,
        terminal_kill_delivery_reward=terminal,
        terminal_components={'settled_kill':2,'transport_capture':9,'control_residual_reward':61},
        reward_share_gate_pass=(extra<=.3*total and extra<=terminal) if terminal_available and total>0 else None,
        relative_reward_share_gate_pass=(extra<=.3*total) if exposures and total>0 else None,
        topology_reward_events_per_training_episode=sum(e.get('topology_events',0) for e in exposures)/episodes if episodes else None,
        training_episode_denominator=episodes if exposures else None,
        scope='new-rule reward prefix within target chunk; partial episodes; not full ancestry; legacy histories excluded',
        JS_training_reward_unavailable=True)


def promotion_failures(absolute,split):
    return list(absolute)+(['HOLDOUT_REQUIRED_FOR_BEST'] if split!='holdout' else [])


def request_holdout(name,marker,relative_eligible):
    if not relative_eligible or marker.get('split')!='dev' or not marker.get('checkpoint'):return None
    path=BASE/'eval_requests'/(name+'_holdout.json')
    if not path.exists():
        atomic_json(path,dict(name=path.stem,checkpoint=marker['checkpoint'],split='holdout',
            rule_version='stock-window-v1',reward_version=marker.get('reward_version','bomb-threat-v1'),
            **({'iter2_arm':marker['iter2_arm']} if marker.get('iter2_arm') else {}),
            reason='independent holdout required before BEST; same checkpoint, no promotion performed'))
    return path


def round1_holdout_priority():
    """Keep the two 14:00 A3 holdouts ahead of later gate refresh priorities."""
    decision_path=BASE/'iter2/DECISION_ROUND1.json'
    decision=json.loads(decision_path.read_text()) if decision_path.exists() else {}
    if decision.get('status')!='AUTHORIZED':return []
    names=[]
    for update in (680,720):
        name=f'bt_slot01_c000015_u{update:06d}_holdout'
        if (BASE/'eval_requests'/f'{name}.json').exists() and not (BASE/'eval'/name/'NATURAL_COMPLETE.json').exists():
            names.append(name)
    return names


def monitor_priority():
    """Retain explicitly authorized monitoring order across report refreshes."""
    path=BASE/'MONITOR_20261030_0200.json'
    decision=json.loads(path.read_text()) if path.exists() else {}
    if decision.get('status')!='AUTHORIZED':return []
    round14=BASE/'iter2/a3sr/round14/AUTHORIZATION.json'
    extra=['fresh256_a3sr_u640','bt_slot05_c000026_u000600_holdout'] if round14.exists() else []
    round16=BASE/'iter2/a3sr/round16/EVAL_ORDER.json'
    r16=json.loads(round16.read_text()) if round16.exists() else {}
    if r16.get('status')=='RELEASED_AFTER_FRESH_COMPLETE':extra=r16.get('names',[])+extra
    return list(dict.fromkeys(name for name in extra+decision.get('priority_names',[])
        if isinstance(name,str) and Path(name).name==name
        and (BASE/'eval_requests'/(name+'.json')).exists()))


def refresh_all():
    try:guard=lock(BASE/'GATE_REFRESH.lock')
    except BlockingIOError:return {}
    markers={};naturals={}
    for path in sorted((BASE/'eval').glob('*/NATURAL_COMPLETE.json')):
        if (path.parent/'EXCLUDED_INVALID.json').exists():continue
        marker,natural=recompute_natural(path.parent)
        markers[path.parent.name]=marker;naturals[path.parent.name]=natural
    reference_names=('reference_transport1536','reference_selfplay3072')
    if all(n in markers for n in reference_names) and 'baseline_parent' in markers:
        parent=markers['baseline_parent']
        assert all(all(markers[n].get(k)==parent.get(k) for k in
            ('split','seed','spawn_sha256','simulator_sha256','rule_version')) for n in reference_names)
        atomic_json(BASE/'OLD_WEB_REFERENCES_ACCEPTANCE.json',dict(status='PASS',
            games=256,modes=['greedy','categorical'],split=parent['split'],seed=parent['seed'],
            matched_new_rule_parent=True,raw_release_byte_identity_claimed=False,
            sources={n:dict(natural_sha256=sha(BASE/'eval'/n/'NATURAL_COMPLETE.json'),
                policy_sha256=markers[n]['policy_sha256'],export_sha256=markers[n]['export_sha256']) for n in reference_names},unix=time.time()))
    baselines_complete=all(n in markers for n in ('baseline_parent','baseline_u1120','baseline_slot01_latest'))
    reports={}
    for name,marker in markers.items():
        out=BASE/'eval'/name;natural=naturals[name];split=marker['split']
        parent_name='baseline_parent' if split=='holdout' else 'baseline_parent_dev'
        parent=markers.get(parent_name)
        matching=bool(parent and all(marker.get(k)==parent.get(k) for k in
            ('rule_version','split','seed','spawn_sha256','simulator_sha256')))
        reference=naturals.get(parent_name) if matching else None
        cp=ROOT/marker['checkpoint'] if marker.get('checkpoint') else None
        cm=json.loads((cp/'COMMITTED.json').read_text()) if cp else {}
        hack=reward_monitor(cp,cm['update']) if cp else {}
        reference_set={n:naturals[n] for n in reference_names if n in markers and all(
            markers[n].get(k)==marker.get(k) for k in ('rule_version','split','seed','spawn_sha256','simulator_sha256'))}
        decision=evaluate(natural,reference,hack,split,reference_set)
        absolute=decision['absolute']['failures'];relative=decision['relative']['failures']
        old_path=out/'GATE_METRICS.pre_gate_amend2_20261029_0900.json'
        current_path=out/'GATE_METRICS.json'
        zero_backup=out/'GATE_METRICS.pre_zero_bomb_20261029_1420.json'
        if current_path.exists() and not zero_backup.exists():zero_backup.write_bytes(current_path.read_bytes())
        if not old_path.exists() and current_path.exists():
            current=json.loads(current_path.read_text())
            if current.get('gate_version')!=GATE_VERSION:old_path.write_bytes(current_path.read_bytes())
        old=json.loads(old_path.read_text()) if old_path.exists() else {}
        if marker.get('role')=='web_reference':
            absolute.append('REFERENCE_ONLY');relative.append('REFERENCE_ONLY')
        g=natural['greedy']
        reports[name]=dict(gate_version=GATE_VERSION,natural_js_full2400=natural,hack_monitor=hack,
            checkpoint=marker.get('checkpoint'),model_json=marker.get('model_json'),policy_sha256=marker['policy_sha256'],
            reward_version=marker.get('reward_version','bomb-threat-v1'),iter2_arm=marker.get('iter2_arm'),
            source_natural_sha256=sha(out/'NATURAL_COMPLETE.json'),matched_parent=parent_name if matching else None,
            absolute_gate_pass=decision['absolute']['status']=='PASS',absolute_gate_failures=absolute,
            absolute_gate_status=decision['absolute']['status'],absolute_gate_missing=decision['absolute']['missing'],
            best_eligible=False,best_gate_status=decision['best']['status'],
            best_gate_failures=decision['best']['failures'],best_gate_missing=decision['best']['missing'],
            pbt_selection_eligible=decision['relative']['status']=='PASS' and marker.get('role')!='web_reference',pbt_gate_failures=relative,
            pbt_gate_status=decision['relative']['status'],pbt_gate_missing=decision['relative']['missing'],
            previous_gate_version=old.get('gate_version'),
            previous_best_gate=('PASS' if old.get('best_eligible') else 'FAIL') if old else None,
            previous_pbt_gate=('PASS' if old.get('pbt_selection_eligible') else 'FAIL') if old else None,
            selection_primary_order=[g['win_minus_loss'],g['causal_kills'],-g['deaths']],
            selection_status='GATE_DECISION_ONLY; EXECUTED_PBT_COPIES_RECORDED_SEPARATELY; NO_BEST_PROMOTION',
            limitations=['self_deaths counts direct ownBombDefeat only; indirect harms not identified',
                          'JS exact training rewards unavailable','reward shares are within-chunk prefixes, partial episodes',
                          'chain_reward is aggregate threat/topology/chain payout; not three separable channels',
                          'fresh 256/tier review and external publication approval remain required; no BEST promotion'],
            unix=time.time())
    # Match by physical slot, full causal protocol ancestry, split and seed.
    # Excluded/restarted branches and duplicate policies never count as two points.
    trend_groups={}
    for name,report in reports.items():
        marker=markers[name];cp=marker.get('checkpoint') or '';run=Path(cp).parent.name
        if not run.startswith('bt_slot'):continue
        slot=run.split('_')[1];key=(slot,marker['split'],marker['seed'],marker['spawn_sha256'])
        trend_groups.setdefault(key,[]).append((run,int(Path(cp).name[1:]),name))
    for key,items in trend_groups.items():
        parent_name='baseline_parent' if key[1]=='holdout' else 'baseline_parent_dev'
        reference=naturals.get(parent_name,{}).get('greedy')
        items.sort();seen=set();points=[]
        previous_run=None
        for run,update,name in items:
            r=reports[name];digest=r['policy_sha256']
            if digest in seen:continue
            seen.add(digest)
            # A fresh-parent restart is not a continuation of the preceding chunk.
            proto_path=BASE/'protocols'/(run+'.json')
            proto=json.loads(proto_path.read_text())
            source=next(iter(proto['sources'].values()))['checkpoint']
            parent_run=Path(source).parent.name
            if previous_run is not None and run!=previous_run and parent_run!=previous_run:
                points=[]
            points.append(r['natural_js_full2400']['greedy']);previous_run=run
            trend=shaping_trend(points,reference)
            r['shaping_trend']=trend
            if trend['pause_required']:
                # Owner consumes this at the next atomic chunk boundary; no
                # mutation of a running worker or silent coefficient adjustment.
                slot=key[0].replace('slot','')
                path=BASE/'shaping_pause_requests'/f'slot{slot}_{key[1]}.json'
                first_request=not path.exists()
                atomic_json(path,dict(status='SHAPING_PAUSE_REQUIRED',slot=int(slot),checkpoint=r['checkpoint'],
                    evaluated_points=[x[2] for x in items if x[:2]<=(run,update)],
                    report=str((BASE/'eval'/name/'GATE_METRICS.json').relative_to(ROOT)),
                    action='block next shaping chunk at atomic boundary; report orchestrator; do not retune',unix=time.time()))
                if first_request:
                    from scripts.iter_pbt_owner import notify
                    notify('BOMB_THREAT_SHAPING_PAUSE_'+key[0]+'_'+key[1],
                        f'BOMB-THREAT {key[0]}/{key[1]} 连续两点评估泡上升但击杀/胜局未升，已请求在原子chunk边界暂停后续塑形；系数不擅改。',request=str(path.relative_to(ROOT)))
    for name,r in reports.items():
        out=BASE/'eval'/name
        atomic_json(out/'GATE_METRICS.json',r)
        for filename in ('SUMMARY.json','COMPLETE.json'):
            source=out/filename;backup=out/(source.stem+'.pre_gate_amend2_20261029_0900.json')
            if source.exists() and not backup.exists():backup.write_bytes(source.read_bytes())
        summary_path=out/'SUMMARY.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text())
            patch=dict(gate_version=GATE_VERSION,natural_js_full2400=r['natural_js_full2400'],
                hack_monitor=r['hack_monitor'],best_eligible=r['best_eligible'],best_gate_failures=r['best_gate_failures'],
                pbt_selection_eligible=r['pbt_selection_eligible'],pbt_gate_failures=r['pbt_gate_failures'],
                selection_eligible=r['pbt_selection_eligible'],hard_gate_failures=r['best_gate_failures'],
                selection_primary_order=r['selection_primary_order'],selection_status=r['selection_status'],
                best_gate_status=r['best_gate_status'],absolute_gate_status=r['absolute_gate_status'],
                best_gate_missing=r['best_gate_missing'],pbt_gate_status=r['pbt_gate_status'],pbt_gate_missing=r['pbt_gate_missing'],
                gate_metrics='GATE_METRICS.json')
            if any(summary.get(k)!=v for k,v in patch.items()):
                archive=out/'SUMMARY.pre_gate_amend_20261029_0240.json'
                if not archive.exists():archive.write_bytes(summary_path.read_bytes())
                complete_path=out/'COMPLETE.json'
                complete=json.loads(complete_path.read_text()) if complete_path.exists() else None
                if complete and complete.get('summary_sha256'):
                    backup=out/'COMPLETE.pre_gate_amend_20261029_0240.json'
                    if not backup.exists():backup.write_bytes(complete_path.read_bytes())
                summary.update(patch);summary['gate_updated_unix']=time.time()
                atomic_json(summary_path,summary)
                if complete and complete.get('summary_sha256'):
                    complete['summary_sha256']=sha(summary_path)
                    complete['gate_version']=GATE_VERSION
                    atomic_json(complete_path,complete)

    rankings={}
    for split in ('dev','holdout'):
        eligible=[n for n,r in reports.items() if markers[n]['split']==split and r['pbt_selection_eligible']]
        eligible.sort(key=lambda n:tuple(reports[n]['selection_primary_order']),reverse=True)
        rankings[split]=[dict(name=n,checkpoint=reports[n]['checkpoint'],
            selection_primary_order=reports[n]['selection_primary_order']) for n in eligible]
    top=[r['name'] for r in rankings['dev'] if r['checkpoint'] and Path(r['checkpoint']).parent.name.startswith('bt_slot')][:3]
    mandatory=[n for n,m in markers.items() if m.get('split')=='dev' and m.get('checkpoint') and
               Path(m['checkpoint']).parent.name.startswith('bt_slot00_') and Path(m['checkpoint']).name=='u000960']
    round1_priority=round1_holdout_priority()
    priority=list(dict.fromkeys(round1_priority+top+mandatory));requests=[]
    for name in priority:
        path=request_holdout(name,markers[name],True)
        if path and not (BASE/'eval'/path.stem/'NATURAL_COMPLETE.json').exists():requests.append(path.stem)
    atomic_json(BASE/'GATE_AMEND2_HOLDOUT_PRIORITY.json',dict(gate_version=GATE_VERSION,
        names=list(dict.fromkeys(monitor_priority()+round1_priority+requests)),monitor_priority=monitor_priority(),
        top3=top,mandatory_slot00_u960=mandatory,
        round1_a3_holdout=round1_priority,union=priority,unix=time.time()))
    copy_path=BASE/'PBT_COPY_ACCEPTANCE_20261029_0700.json'
    copied=json.loads(copy_path.read_text()) if copy_path.exists() else {}
    atomic_json(BASE/'PBT_SELECTION.json',dict(gate_version=GATE_VERSION,rankings=rankings,
        status='SELECTION_DECISION_ONLY',executed_copy_count=len(copied.get('slots',[])) if copied.get('status')=='PASS' else 0,
        executed_copy_receipt=str(copy_path.relative_to(ROOT)) if copied else None,
        executed_copy_receipt_sha256=sha(copy_path) if copied else None,
        round_copy_limit=2,excluded_copy_slots=[5,6],additional_copies_authorized_this_round=0,
        web_or_best_changed=False,unix=time.time()))
    atomic_json(BASE/'GATE_AMEND_ACCEPTANCE.json',dict(status='PASS',gate_version=GATE_VERSION,
        verified_natural_markers=len(markers),reports={n:sha(BASE/'eval'/n/'GATE_METRICS.json') for n in reports},
        immutable_natural_markers_preserved=True,no_games_rerun=True,unix=time.time()))
    return reports

if __name__=='__main__':refresh_all()
