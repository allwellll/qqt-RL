"""Durable passive round15 completion/report monitor, never a training owner."""
import time
from scripts.population_run_state import atomic_json, lock, sha
from scripts import bomb_threat_round15 as R
from scripts.bomb_threat_round15_report import write_report


def main():
    guard=lock(R.OUT/'MONITOR.lock')
    while not (R.BASE/'USER_STOP_EVAL.json').exists():
        try:
            state=write_report()
            auth=R.read(R.AUTH,{})
            pending=[]
            for name in auth.get('candidates',[]):
                cp=R.ROOT/name
                out=R.BASE/'eval'/f"{auth['donor_run']}_{cp.name}"
                if not (cp/'COMMITTED.json').exists() or len(list(out.glob('natural_greedy_*.json.outcome.json')))!=8:
                    pending.append(name)
            no_donor=not auth.get('donor') and not pending
            if no_donor:
                target=R.OUT/'DONOR_BLOCKED.json'
                if not target.exists():
                    value=dict(status='BLOCKED_NO_ELIGIBLE_COMPLETE_STATE',
                        reason='all retained current-chain cadence candidates fail explicit dev admission; threshold not weakened',
                        admission=R.read(R.OUT/'DONOR_ADMISSION.json'),other_slots_unchanged=True,unix=time.time())
                    atomic_json(target,value)
                    atomic_json(R.BASE/'notifications/ROUND15_DONOR_BLOCKED.json',value)
                    from scripts.iter_pbt_owner import notify
                    notify('ROUND15_DONOR_BLOCKED','round15全部完整donor候选dev未满足胜负差>=25且零泡<=4，slot07复制受阻；未降低门槛，其余训练继续，报告已落盘。')
            complete=bool(state['first_formal']) and state['diagnostic_complete']
            receipt=dict(status='COMPLETED_TRAINING_CONTINUES' if complete else 'BLOCKED' if no_donor else state['status'],
                instruction=R.INSTRUCTION,correction=R.CORRECTION,
                report='reports/bomb_threat_v1_round15_20261030.md',
                report_sha256=sha(R.ROOT/'reports/bomb_threat_v1_round15_20261030.md'),
                diagnostic_complete=state['diagnostic_complete'],diagnostic_games=state['diagnostic_games'],
                first_formal=state['first_formal'],fresh_continues=state['fresh_collection_status']!='COMPLETE',
                training_worker_signals=0,no_promotion=True,no_publication=True,unix=time.time())
            atomic_json(R.OUT/'TASK_CONSUMED.json',receipt)
            if (complete or no_donor) and state['diagnostic_complete']:
                target=R.OUT/'TASK_NOTIFICATION.json'
                if not target.exists():
                    atomic_json(target,receipt)
                    atomic_json(R.BASE/'notifications/ROUND15_TASK.json',receipt)
                    from scripts.iter_pbt_owner import notify
                    notify('ROUND15_TASK',f"round15诊断58局完整匹配；slot07={receipt['status']}，fresh继续，不晋升不发布。",report=receipt['report'])
                if state['fresh_collection_status']=='COMPLETE':return
        except Exception as error:
            atomic_json(R.OUT/'MONITOR_RETRY.json',dict(status='RETRY_REQUIRED',
                error_type=type(error).__name__,detail=str(error),training_untouched=True,unix=time.time()))
        time.sleep(30)


if __name__=='__main__':main()
