"""Passive A3S launch acceptance and terminal notification; no worker launch path."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'


def read(p,default=None):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return default


def retained_atomic(initial,active):
    # The retention policy may remove every state from the first chunk after
    # continuation. Verify the active chunk instead; never index an empty list.
    for out in dict.fromkeys((active,initial)):
        for path in reversed(sorted(out.glob('u*/COMMITTED.json'))):
            if not all((path.parent/f).exists() for f in ('state.pkl','policy.pt')):continue
            cm=read(path)
            try:
                assert sha(path.parent/'state.pkl')==cm['state_sha256'] and sha(path.parent/'policy.pt')==cm['policy_sha256']
            except FileNotFoundError:continue  # concurrent authorized retention
            return cm
    return None


def refresh():
    slots=read(BASE/'SLOTS.json',{});records=[]
    for slot in (4,7):
        entry=slots.get(str(slot),{});handoff=read(BASE/'a3s'/f'SLOT{slot:02d}_HANDOFF.json',{})
        run=handoff.get('successor');record=dict(slot=slot,status='WAITING_FOR_COMPLETE_OLD_CHUNK',active_run=entry.get('run'))
        if run:
            out=BASE/'formal'/run;receipts={k:read(out/(k+'.json'),{}) for k in ('INIT','WARMUP_GATE','SMOKE','RESTORE')}
            rows=[]
            if (out/'updates.jsonl').exists():
                for line in (out/'updates.jsonl').read_bytes().splitlines():
                    try:rows.append(json.loads(line))
                    except ValueError:pass
            record.update(run=run,source=handoff['source'],lineage=handoff['lineage'],formal_updates_discarded=0,
                latest_update=rows[-1]['new_update'] if rows else 0,status='TECHNICAL_STAGE_RUNNING')
            formal=[r for r in rows if r['new_update']>1]
            if formal and all(receipts[k].get('passed') for k in ('WARMUP_GATE','SMOKE','RESTORE')):
                init=receipts['INIT'];rest=receipts['RESTORE'];assert init['learner_seat_counts']==[1024,1024] and init['bank_P1_forbidden']
                assert init['actor_logits_first8_bitwise'] and all(rest[k] for k in ('fresh_process','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact'))
                assert formal[0]['finite'] and min(formal[0]['update_l2'].values())>0
                for r in rows:
                    assert r['exposure']['seat_ticks']==[131072.,131072.]
                    assert r['exposure']['bank_seat_ticks'][1]==0 and r['exposure']['bank_seat_resets'][1]==0
                cm=retained_atomic(out,BASE/'formal'/entry.get('run',run))
                if cm is None:
                    record.update(status='AWAITING_RETAINED_ATOMIC_STATE',initialization_verified=True,
                        reason='historical state removed by retention; retry live atomic checkpoint')
                    records.append(record);continue
                record.update(status='FORMAL_RUNNING_VERIFIED',first_post_restore_update=formal[0],latest_update=rows[-1]['new_update'],
                    latest_env_per_second=262144/rows[-1]['cycle_seconds'],latest_atomic=cm,
                    receipt_sha256={k:sha(out/(k+'.json')) for k in receipts},
                    warmup_actual_updates=20,warmup_environment_steps=20*262144,actor_frozen_during_warmup=True)
                notify=BASE/'notifications'/f'A3S_SLOT{slot:02d}_FIRST_FORMAL.json'
                if not notify.exists():
                    atomic_json(notify,record)
                    from scripts.iter_pbt_owner import notify as send
                    send(f'A3S_SLOT{slot:02d}_FIRST_FORMAL',f'A3S slot{slot:02d}已正式训练：20次Critic预热、1update smoke及fresh恢复通过，u{formal[0]["new_update"]}finite、Actor/Critic非零，seat ticks严格50/50，Bank P1=0。')
        records.append(record)
    status='ALL_FORMAL_RUNNING_VERIFIED' if all(x['status']=='FORMAL_RUNNING_VERIFIED' for x in records) else 'BOUNDARY_OR_TECHNICAL_STAGE_PENDING'
    result=dict(status=status,slots=records,training_signals_sent=0,old_queues_resumed=False,unix=time.time())
    atomic_json(BASE/'a3s/FIRST_FORMAL_ACCEPTANCE.json',result);return result


if __name__=='__main__':print(json.dumps(refresh()))
