"""Reuse the bounded SP owner and lock in an independent seed namespace."""
import os
import sys
import time
from pathlib import Path
from scripts.population_run_state import atomic_json
from scripts.r3_sp_seed2_train_20261031 import NAMESPACE


def main():
    os.environ['R3_SP_NAMESPACE']=NAMESPACE
    from scripts import r3_selfplay_owner_20261031 as owner
    owner.OUT=Path(NAMESPACE)
    owner.MODULE='scripts.r3_sp_seed2_owner_20261031'
    owner.WORKER='scripts.r3_sp_seed2_train_20261031'
    owner.EVALUATOR='scripts.r3_sp_seed2_eval_20261031'
    owner.evaluate=lambda:owner.read(owner.OUT/'PARENT_GATE.json')
    notify=owner.notify
    owner.notify=lambda key,message,**evidence:notify(key,'R3-sp第二seed2026323002：'+message,**evidence)
    try:
        if '--watchdog' in sys.argv:owner.watchdog()
        else:owner.main()
    except Exception as error:
        atomic_json(owner.OUT/'OWNER_BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,reason=str(error),unix=time.time()))
        owner.notify('TECHNICAL_BLOCKED','R3-sp第二seed技术阻塞，完整证据已保存；第一seed和原R3未发信号。')
        raise


if __name__=='__main__':main()
