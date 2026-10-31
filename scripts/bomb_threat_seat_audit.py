"""Passive seat exposure and outcome audit; no training scheduler mutation."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha

BASE=Path('runs/bomb_threat_v1');OUT=BASE/'iter2/seat_asym_audit'
NAMES=('hunter_tier3a','hunter_tier3b','jax_hunter_hard','selfplay_current','selfplay_ring','selfplay_fixed')


def collect_exposure(paths):
    totals={name:dict(P0_ticks=0,P1_ticks=0,P0_live_ticks=0,P1_live_ticks=0,episodes=0) for name in NAMES}
    records=[]
    for path in paths:
        data=path.read_bytes();rows={}
        for line in data.splitlines():
            try:r=json.loads(line)
            except ValueError:continue  # atomic appended last line may be in flight
            e=r.get('exposure',{})
            if len(e.get('pool_ticks',[]))!=6:continue
            rows[r['new_update']]=r
        per={name:dict(P0_ticks=0,P1_ticks=0,P0_live_ticks=0,P1_live_ticks=0,episodes=0) for name in NAMES}
        for r in rows.values():
            e=r['exposure']
            for i,name in enumerate(NAMES):
                by_seat=e.get('pool_seat_ticks')
                p0=e['pool_ticks'][i] if by_seat is None else by_seat[i][0]
                p1=0 if by_seat is None else by_seat[i][1]
                live_seat=e.get('pool_seat_live_ticks')
                live0=e['pool_live_ticks'][i] if live_seat is None else live_seat[i][0]
                live1=0 if live_seat is None else live_seat[i][1]
                for key,value in [('P0_ticks',p0),('P1_ticks',p1),('P0_live_ticks',live0),('P1_live_ticks',live1),('episodes',e['pool_new_episodes'][i])]:
                    totals[name][key]+=value;per[name][key]+=value
        import hashlib
        records.append(dict(run=path.parent.name,updates=len(rows),first=min(rows) if rows else None,
            last=max(rows) if rows else None,log_prefix_bytes=len(data),log_prefix_sha256=hashlib.sha256(data).hexdigest(),by_opponent=per))
    for value in totals.values():
        den=value['P0_ticks']+value['P1_ticks'];value['learner_P1_fraction']=value['P1_ticks']/den if den else None
    return dict(by_opponent=totals,runs=records,scope='available BOMB-THREAT formal update records; deduplicated new_update within run, cumulative copies not resummed',
        denominators=dict(ticks='all collector environment transitions',live_ticks='~aux.terminal; not own_alive',
            episodes='collector done/reset events, not independent complete source games'),
        opponent_label_caveat='recorded rule3 kind0 and jax_hunter kind1 both dispatch bun_jax_bots tier3=hunter; kind2 dispatches tier4=hunter_hard; no relabeling or training change',
        learner_seat_basis='legacy collector P0-only; A3S explicit pool_seat_ticks follows physical learner_seat; opponent inference never PPO sample',
        P1_policy_inference_is_opponent_not_PPO_sample=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    active=json.loads((BASE/'SLOTS.json').read_text());runs=sorted({x['run'] for x in active.values()})
    # Current run actual exposure plus all valid prior formal chunks, separately.
    from scripts.bomb_threat_owner import invalid_ancestry
    paths=[p for p in (BASE/'formal').glob('*/updates.jsonl')
        if not invalid_ancestry(p.parent.name)
        and not any((p.parent/n).exists() for n in ('EXCLUDED_INVALID.json','EXCLUDED_EVALUATION.json'))]
    all_exposure=collect_exposure(sorted(paths))
    current=collect_exposure([BASE/'formal'/n/'updates.jsonl' for n in runs if (BASE/'formal'/n/'updates.jsonl').exists()])
    sources=['scripts/carrier_mechanism_engine.py','scripts/bomb_threat_collector.py',
        'scripts/bomb_threat_collector_diagnostics_v1.py','scripts/combat_mix_opponents.py',
        'scripts/bomb_threat_iter2_diagnostics_train.py','web/app.js','web/sim-stock-window-v1.js',
        'jax_bomb/je07_bun_env.py','jax_bomb/jax_train.py']
    atomic_json(OUT/'EXPOSURE.json',dict(status='AUDIT_COMPLETE_P1_UNDERSAMPLED',all_available=all_exposure,current=current,
        source_sha256={s:sha(s) for s in sources},unix=time.time(),new_sampling_implemented=False))
    selected={}
    from scripts.bomb_threat_gate_report import recompute_natural
    for name in ('baseline_parent_dev','baseline_parent','bt_slot01_c000015_u000720','bt_slot01_c000015_u000720_holdout'):
        marker,values=recompute_natural(BASE/'eval'/name)
        selected[name]=dict(policy_sha256=marker['policy_sha256'],split=marker['split'],seed=marker['seed'],metrics=values)
    atomic_json(OUT/'SELECTED_SEATS.json',dict(status='COMPLETE',selected=selected,gate_changed=False,unix=time.time()))
    print(json.dumps(dict(runs=len(paths),current=current['by_opponent'])))


if __name__=='__main__':main()
