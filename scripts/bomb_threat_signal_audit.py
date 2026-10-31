"""Read-only native signal accounting from frozen, committed training windows.

This does not replay or train policies. It deliberately distinguishes counters
that exist in the native collector from event-level links absent in old logs.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.population_run_state import atomic_json,sha

ROOT=Path(__file__).resolve().parents[1]


def summarize_window(rows):
    if not rows:raise ValueError('empty native audit window')
    sums={}
    for row in rows:
        if row.get('finite') is not True:raise ValueError('nonfinite or unaudited update')
        for key,value in row.get('exposure',{}).items():
            if isinstance(value,(int,float)):
                sums[key]=sums.get(key,0)+value
    requests=sums.get('learner_bomb_requests',0)
    placed=sums.get('learner_bombs_placed',0)
    rejected=sums.get('learner_bomb_requests_rejected',0)
    if requests!=placed+rejected:raise ValueError('request/place/reject accounting mismatch')
    terminal_available=all(len(r.get('exposure',{}).get('native_parts',[]))==90 for r in rows)
    terminal=(sum(sum(r['exposure']['native_parts'][i] for i in (2,9,61)) for r in rows)
        if terminal_available else None)
    bonus=sums.get('chain_reward',0);total=sums.get('total_reward',0)
    absolute=sums.get('absolute_total_reward',0)
    return dict(updates=len(rows),start_update=rows[0]['new_update'],end_update=rows[-1]['new_update'],
        raw_counters=sums,request_accounting_exact=True,
        actual_place_per_request=placed/requests if requests else None,
        first_safe_threat_events_per_actual_place=sums.get('threat_events',0)/placed if placed else None,
        topology_events_per_actual_place=sums.get('topology_events',0)/placed if placed else None,
        signal_union_events_per_actual_place=sums.get('chain_candidates',0)/placed if placed else None,
        positive_payment_transitions=sums.get('chain_settled',0),
        payment_over_net_reward=bonus/total if total>0 else None,
        payment_over_absolute_reward=bonus/absolute if absolute>0 else None,
        terminal_kill_delivery_reward=terminal,
        payment_over_terminal_reward=bonus/terminal if terminal and terminal>0 else None,
        semantics=dict(chain_candidates='union of accepted threat/chain/topology flags per env-tick; not chain-only',
            chain_settled='env-ticks with strictly positive payment; not episodes and not event-linked conversion',
            topology_paid_events='credited topology-event count settled in this window; can include opening carry-in',
            total_ticks='all sampled environment ticks, including terminal bookkeeping',
            resets='collector done/reset transitions, not cause-labelled full natural games'),
        missing_metrics=['legal_bomb_opportunity_ticks','safe_placement_opportunity_ticks',
            'chain_only_events','pending_created_amount','pending_void_amount_by_reason',
            'pending_opening_closing_balance','event_id_to_payment_or_terminal_link',
            'native_phase_action_probabilities_and_entropy'])


def audit_manifest(manifest,output):
    config=json.loads(manifest.read_text());output.mkdir(parents=True,exist_ok=True)
    results=[]
    for item in config['windows']:
        run=ROOT/item['run'];end=int(item['end_update']);width=int(item.get('updates',40))
        cp=run/f'u{end:06d}';commit=json.loads((cp/'COMMITTED.json').read_text())
        assert commit['passed'] and commit['update']==end
        from scripts.checkpoint_policy_artifact import verify_policy
        verify_policy(cp)
        assert sha(cp/'policy.pt')==commit['policy_sha256']
        selected={}
        for line in (run/'updates.jsonl').read_text().splitlines():
            row=json.loads(line);u=row['new_update']
            if end-width<u<=end:
                if u in selected and selected[u]!=row:raise ValueError('conflicting duplicate update')
                selected[u]=row
        assert set(selected)==set(range(end-width+1,end+1)),item
        rows=[selected[u] for u in sorted(selected)]
        snapshot=output/f"slot{int(item['slot']):02d}_u{end:06d}_rows.json"
        atomic_json(snapshot,rows)
        result=summarize_window(rows)
        result.update(slot=item['slot'],run=item['run'],atomic_checkpoint=str(cp.relative_to(ROOT)),
            state_sha256=commit['state_sha256'],policy_sha256=commit['policy_sha256'],
            selected_rows=str(snapshot.relative_to(ROOT)),selected_rows_sha256=sha(snapshot))
        results.append(result)
    result=dict(status='NATIVE_AGGREGATE_AUDIT_COMPLETE_EVENT_LINKS_MISSING',
        manifest=str(manifest.relative_to(ROOT)),manifest_sha256=sha(manifest),
        training_or_reward_changed=False,new_training_steps=0,windows=results,
        interpretation='Existing native counters are exact within selected windows; no per-signal conversion or cancellation claim is possible from them.')
    atomic_json(output/'RESULT.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('manifest');parser.add_argument('output')
    args=parser.parse_args();result=audit_manifest(Path(args.manifest).resolve(),Path(args.output).resolve())
    print(json.dumps(dict(status=result['status'],windows=len(result['windows'])),ensure_ascii=False))
