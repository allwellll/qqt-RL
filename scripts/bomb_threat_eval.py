"""CPU-only new-rule full-game evaluation, independent from training owner."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',QQT_STOCK_RULE='stock-window-v1')
import sys,json,time,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scripts.population_run_state import atomic_json,sha,lock
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'

def native_reward_exposures(cp,update):
    records={}
    if cp.is_relative_to(BASE/'formal') and (cp.parent/'updates.jsonl').exists():
        for line in (cp.parent/'updates.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['new_update']<=update and 'threat_events' in r.get('exposure',{}):
                records[r['new_update']]=r
    return [r['exposure'] for _,r in sorted(records.items())]

def summarize(rows,include_seats=True):
    games=len(rows);draws=[r for r in rows if r['winner'] in (None,-1)];kills=sum(r['causal_kills'][r['seat']] for r in rows);cage=sum(r['cage_kills'][r['seat']] for r in rows)
    result=dict(games=games,win=sum(r['winner']==r['seat'] for r in rows),draw=sum(r['winner'] is None or r['winner']==-1 for r in rows),
        loss=sum(r['winner'] not in (None,-1,r['seat']) for r in rows),
        win_minus_loss=sum(r['winner']==r['seat'] for r in rows)-sum(r['winner'] not in (None,-1,r['seat']) for r in rows),
        draw_own_causal_kills=sum(r['causal_kills'][r['seat']] for r in draws),
        draw_opponent_causal_kills=sum(r['causal_kills'][1-r['seat']] for r in draws),
        draw_kill_difference=sum(r['causal_kills'][r['seat']]-r['causal_kills'][1-r['seat']] for r in draws),
        actual_bombs=sum(r['placed'][r['seat']] for r in rows),bombs_per_game=sum(r['placed'][r['seat']] for r in rows)/games,
        zero_bomb_games=sum(r['placed'][r['seat']]==0 and r['placed'][1-r['seat']]>0 for r in rows),
        structural_zero_bomb_games=sum(r['placed']==[0,0] for r in rows),
        raw_zero_bomb_games=sum(r['placed'][r['seat']]==0 for r in rows),
        opponent_bomb_opportunity_games=sum(r['placed'][1-r['seat']]>0 for r in rows),
        zero_bomb_measurement='opponent_actual_bombs_positive_v1',bomb_requests=sum(r['requests'][r['seat']] for r in rows),
        causal_kills=kills,cage_kills=cage,cage_kill_denominator=kills,cage_kill_fraction=cage/kills if kills else None,
        deaths=sum(r['deaths'][r['seat']] for r in rows),self_deaths=sum(r['self_deaths'][r['seat']] for r in rows),delivery=sum(r['stored'][r['seat']][1-r['seat']] for r in rows),draw_fraction=sum(r['winner'] in (None,-1) for r in rows)/games)
    if include_seats:
        result['by_seat']={f'P{seat}':summarize(selected,False) if selected else None
            for seat in (0,1) for selected in [[r for r in rows if r['seat']==seat]]}
    return result

def main():
    req=json.loads(Path(sys.argv[1]).read_text());out=BASE/'eval'/req['name'];out.mkdir(parents=True,exist_ok=True);guard=lock(out/'worker.lock')
    if (out/'COMPLETE.json').exists():return
    from scripts import je08_bun_authority as A
    if req.get('model_json'):
        cp=None;model=ROOT/req['model_json'];assert sha(model)==req['model_sha256']
        cm={'policy_sha256':req.get('source_policy_sha256',req['model_sha256'])}
    else:
        cp=ROOT/req['checkpoint'];cm=json.loads((cp/'COMMITTED.json').read_text());assert sha(cp/'policy.pt')==cm['policy_sha256']
        model=A.export(cp/'policy.pt',out/'model')
    split=req.get('split','dev');spawn=BASE/f'{split}_spawns.json';seed=json.loads(spawn.read_text())['seed']
    files={};natural={}
    for mode in ('greedy','categorical'):
        def run(start):
            p=out/f'natural_{mode}_{start:03d}.json'
            side_path=Path(str(p)+'.outcome.json')
            reusable=False
            if p.exists() and side_path.exists():
                try:
                    old=json.loads(p.read_text()); side_old=json.loads(side_path.read_text())
                    eps=side_old.get('episodes',[])
                    reusable=(side_old.get('measurement_version')==2 and side_old.get('rule_version')=='stock-window-v1' and
                              side_old.get('engine')=='production_JS' and len(eps)==8 and
                              all(bool(e.get('done')) and 0<int(e.get('ticks',0))<=2400 for e in eps) and
                              len({int(e.get('seed')) for e in eps})==8 and
                              len(old.get('results',{}).get('natural_full2400',{}).get('episodes',[]))==8)
                except (OSError,ValueError,TypeError,KeyError):
                    reusable=False
            if not reusable:
                # Preserve pre-fix evidence; rerun the exact shard with the
                # corrected versioned simulator and frozen spawn frame.
                for old_path in (p,side_path,Path(str(p)+'.partial.json')):
                    if old_path.exists():
                        archived=old_path.with_name(old_path.name+'.pre_fix_20261029')
                        if not archived.exists(): old_path.replace(archived)
                env=dict(os.environ,JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='64',JE10_ROUTE_START=str(start),JE10_ROUTE_END=str(start+8),JE10_ROUTE_SEED_START=str(seed))
                with p.with_suffix('.log').open('ab') as f:subprocess.run(['node',('scripts/bomb_threat_r1_progress_eval.js' if req.get('r1_items_v2') else 'scripts/bomb_threat_r1_eval.js' if req.get('r1_metrics') else 'scripts/bomb_threat_eval.js'),mode,str(model.relative_to(ROOT)),str(p.relative_to(ROOT)),'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',str(spawn.relative_to(ROOT))],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            v=json.loads(p.read_text());side=json.loads(Path(str(p)+'.outcome.json').read_text())
            episodes=v['results']['natural_full2400']['episodes'];assert len(episodes)==len(side['episodes'])==8
            for e,r in zip(episodes,side['episodes']):
                assert e['ticks']==r['ticks'] and r['done'] and e['seed']==r['seed'] and e['team']==r['seat']
                assert e['causal_kill']==r['causal_kills'][e['team']]
                r.update(game=e['game'])
            return p,episodes,side['episodes']
        with ThreadPoolExecutor(max_workers=int(os.environ.get("QQT_EVAL_SHARDS","4"))) as pool:parts=list(pool.map(run,range(0,64,8)))
        allrows=[r for _,_,rr in parts for r in rr];assert len({r['seed'] for r in allrows})==64
        natural[mode]=summarize(allrows)
        natural[mode]['final_enemy_buns_stored']=natural[mode]['delivery']
        natural[mode]['delivery']=sum(e['delivery'] for _,ee,_ in parts for e in ee)
        natural[mode]['window_stock_pickup']=sum(e['stock_pickup_dead']+e['stock_pickup_native_sugar'] for _,ee,_ in parts for e in ee)
        natural[mode]['window_stock_delivery']=sum(e['window_stock_delivery'] for _,ee,_ in parts for e in ee)
        natural[mode]['free_stock_pickup']=sum(e['stock_pickup_free_alive'] for _,ee,_ in parts for e in ee)
        assert natural[mode]['free_stock_pickup']==0,'new rule semantic regression'
        for p,_,_ in parts:files[str(p.relative_to(out))]=sha(p);files[str(p.relative_to(out))+'.outcome.json']=sha(Path(str(p)+'.outcome.json'))
        atomic_json(out/'PROGRESS.json',dict(phase='full_natural_complete',mode=mode,games=64,unix=time.time()))
    natural_marker=dict(checkpoint=req.get('checkpoint'),model_json=req.get('model_json'),role=req.get('role','checkpoint'),export_sha256=req.get('model_sha256'),model_identity=req.get('model_identity'),policy_sha256=cm['policy_sha256'],
        rule_version='stock-window-v1',reward_version=req.get('reward_version','bomb-threat-v1'),iter2_arm=req.get('iter2_arm'),split=split,seed=seed,spawn_sha256=sha(spawn),
        simulator_sha256=sha(ROOT/'web/sim-stock-window-v1.js'),measurement_version=2,
        natural_js_full2400=natural,files=files,unix=time.time())
    if req.get('iter2_arm') and not req.get('r1_metrics'):
        # Extra diagnostic column only; never changes argmax gate/selection.
        half=out/'TEMPERATURE_HALF_COMPLETE.json'
        if not half.exists():
            from scripts.bomb_threat_iter2 import accept_shard
            def run_half(start):
                p=out/f'natural_tau0.5_{start:03d}.json'
                if not Path(str(p)+'.ACCEPTED.json').exists():
                    env=dict(os.environ,JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='64',
                        JE10_ROUTE_START=str(start),JE10_ROUTE_END=str(start+8),JE10_ROUTE_SEED_START=str(seed),
                        ITER2_TAU='.5',ITER2_RANK='0')
                    with p.with_suffix('.log').open('ab') as f:
                        subprocess.run(['node','scripts/bomb_threat_iter2_actions.js','categorical',str(model.relative_to(ROOT)),
                            str(p.relative_to(ROOT)),'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',str(spawn.relative_to(ROOT))],
                            env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                    rows,proof=accept_shard(p,sha(model),start,seed,.5)
                    atomic_json(str(p)+'.ACCEPTED.json',proof)
                else:
                    proof=json.loads(Path(str(p)+'.ACCEPTED.json').read_text())
                    for file,h in proof['files'].items():assert sha(ROOT/file)==h
                    rows,_=accept_shard(p,sha(model),start,seed,.5)
                return rows,proof
            with ThreadPoolExecutor(max_workers=int(os.environ.get('QQT_EVAL_SHARDS','3'))) as pool:half_parts=list(pool.map(run_half,range(0,64,8)))
            half_rows=[r for rr,_ in half_parts for r in rr]
            atomic_json(half,dict(policy_sha256=cm['policy_sha256'],tau=.5,games=64,metrics=summarize(half_rows),
                files={f:h for _,proof in half_parts for f,h in proof['files'].items()},gate_input=False,unix=time.time()))
    # Natural-completion is immutable: a later quickscreen completion must not
    # silently invalidate its acceptance SHA or rerun the finished games.
    if (out/'NATURAL_COMPLETE.json').exists():
        previous=json.loads((out/'NATURAL_COMPLETE.json').read_text())
        assert previous['files']==files and previous['policy_sha256']==cm['policy_sha256']
        for mode,values in previous['natural_js_full2400'].items():
            # Raw completion markers retain their original measurement. Gates
            # are recomputed from the same hashed episodes without replaying.
            assert all(natural[mode].get(k)==v for k,v in values.items() if k not in ('zero_bomb_games','by_seat'))
    else:atomic_json(out/'NATURAL_COMPLETE.json',natural_marker)
    if '--natural-only' in sys.argv:
        from scripts.bomb_threat_gate_report import refresh_all
        if os.environ.get("QQT_GATE_DEFER")!="1":refresh_all()
        if req.get('role')=='web_reference':
            atomic_json(out/'COMPLETE.json',dict(passed=True,natural_sha256=sha(out/'NATURAL_COMPLETE.json'),files=files,role='web_reference',unix=time.time()))
        return
    # Retained quickscreen uses the SAME corrected engine and frozen new spawns.
    hunter={}
    for tier in ('hard','normal','easy'):
        rows=[]
        def run_clear(start):
            p=out/f'hunter_{tier}_{start:03d}.json'
            existing= json.loads(p.read_text()) if p.exists() else {}
            if not p.exists() or existing.get('source_hashes',{}).get('web/sim-stock-window-v1.js')!=sha(ROOT/'web/sim-stock-window-v1.js'):
                with p.with_suffix('.log').open('ab') as f:subprocess.run(['node','scripts/je05_web_hunter.js','--model',str(model.relative_to(ROOT)),'--difficulty',tier,'--spawns',str(spawn.relative_to(ROOT)),'--start',str(start),'--end',str(start+8),'--seed',str(seed),'--checkpoint-sha',cm['policy_sha256'],'--out',str(p.relative_to(ROOT))],check=True,stdout=f,stderr=subprocess.STDOUT)
            v=json.loads(p.read_text());return p,v['episodes']
        with ThreadPoolExecutor(max_workers=int(os.environ.get("QQT_EVAL_SHARDS","4"))) as pool:
            for p,rr in pool.map(run_clear,range(0,64,8)):rows+=rr;files[p.name]=sha(p)
        hunter[tier]=A.summarize_hunter(rows)
    from scripts.bomb_threat_gate_report import reward_monitor,promotion_failures
    hack=reward_monitor(cp,cm['update']) if cp else {}
    flags=[]
    if hack['reward_share_gate_pass'] is False:flags.append('HIGH_THREAT_REWARD_SHARE')
    if hack['reward_share_gate_pass'] is None:flags.append('NATIVE_REWARD_SHARE_UNAVAILABLE')
    if any(v['draw_fraction']>.5 for v in natural.values()):flags.append('HIGH_DRAW_FRACTION')
    parent=BASE/'eval'/('baseline_parent' if split=='holdout' else 'baseline_parent_dev')/'NATURAL_COMPLETE.json'
    if parent.exists():
        reference=json.loads(parent.read_text())['natural_js_full2400']
        for mode,v in natural.items():
            p=reference[mode]
            if v['bombs_per_game']>p['bombs_per_game'] and v['causal_kills']<=p['causal_kills'] and v['win']<=p['win']:flags.append(mode+'_BOMBS_UP_WITHOUT_KILLS')
            if v['draw_fraction']>p['draw_fraction']:flags.append(mode+'_DRAW_REGRESSION')
    from scripts.bomb_threat_gates_v2 import evaluate,GATE_VERSION
    reference=json.loads(parent.read_text())['natural_js_full2400'] if parent.exists() else None
    refs={}
    for name in ('reference_transport1536','reference_selfplay3072'):
        file=BASE/'eval'/name/'NATURAL_COMPLETE.json'
        if file.exists():
            m=json.loads(file.read_text())
            if all(m.get(k)==req.get(k,{'seed':seed,'rule_version':'stock-window-v1','split':split}.get(k)) for k in ('seed','rule_version','split')):
                refs[name]=m['natural_js_full2400']
    decision=evaluate(natural,reference,hack,split,refs)
    failures=decision['absolute']['failures'];relative=decision['relative']['failures']
    g=natural['greedy'];selection=g['win']-g['loss']
    summary=dict(checkpoint=req.get('checkpoint'),policy_sha256=cm['policy_sha256'],rule_version='stock-window-v1',reward_version='bomb-threat-v1',seed=seed,split=split,teacher=False,
        natural_js_full2400=natural,hunter_js_clear=hunter,hack_monitor=hack,hack_flags=flags,selection_score=selection,
        gate_version=GATE_VERSION,selection_eligible=decision['relative']['status']=='PASS',pbt_selection_eligible=decision['relative']['status']=='PASS',pbt_gate_failures=relative,
        pbt_gate_status=decision['relative']['status'],pbt_gate_missing=decision['relative']['missing'],
        absolute_gate_pass=decision['absolute']['status']=='PASS',absolute_gate_failures=failures,absolute_gate_status=decision['absolute']['status'],absolute_gate_missing=decision['absolute']['missing'],
        best_eligible=False,best_gate_status=decision['best']['status'],best_gate_missing=decision['best']['missing'],best_gate_failures=decision['best']['failures'],hard_gate_failures=decision['best']['failures'],
        selection_primary_order=[selection,g['causal_kills'],-g['deaths']],
        selection_status='GATE_DECISION_ONLY; EXECUTED_PBT_COPIES_RECORDED_SEPARATELY; NO_BEST_PROMOTION',spawn_sha256=sha(spawn),simulator_sha256=sha(ROOT/'web/sim-stock-window-v1.js'),
        missing_metrics=['native_training_threat_share' if hack.get('threat_reward_fraction_total') is None else None],unavailable_metrics=['JS_exact_training_rewards'],unix=time.time())
    atomic_json(out/'SUMMARY.json',summary);atomic_json(out/'COMPLETE.json',dict(passed=True,summary_sha256=sha(out/'SUMMARY.json'),files=files))
    from scripts.bomb_threat_gate_report import refresh_all
    if os.environ.get("QQT_GATE_DEFER")!="1":refresh_all()
    # The unique CPU owner publishes the completed batch and retries upload
    # independently of training; workers never launch another owner.

if __name__=='__main__':main()
