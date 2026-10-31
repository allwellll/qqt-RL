"""CPU-only evaluator for the independent authorized SP second seed."""
import os
import time
from pathlib import Path
from scripts.r3_sp_seed2_train_20261031 import NAMESPACE
from scripts.population_run_state import atomic_json


def main():
    os.environ['R3_SP_NAMESPACE']=NAMESPACE
    from scripts import r3_selfplay_eval_20261031 as evaluator
    evaluator.OUT=Path(NAMESPACE)
    try:evaluator.main()
    except Exception as error:
        atomic_json(evaluator.OUT/'EVAL_BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,reason=str(error),unix=time.time()))
        raise


if __name__=='__main__':main()
