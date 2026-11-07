"""Finite queue on twelve free physical evaluation cores; exact cached games unchanged."""
import time
from scripts import cx_items_candidates_v1 as S
from scripts.population_run_state import atomic_json,sha

if __name__=='__main__':
    S.CPU['CC_STRONG_CPUS']='36-47'
    Original=S.concurrent.futures.ThreadPoolExecutor
    class EvaluationPool(Original):
        def __init__(self,max_workers=None,*args,**kw):
            super().__init__(12 if max_workers==8 else max_workers,*args,**kw)
    S.concurrent.futures.ThreadPoolExecutor=EvaluationPool
    atomic_json(S.ROOT/'CPU_SCHEDULING_V3.json',dict(status='TWELVE_FREE_PHYSICAL_EVAL_CORES_SHARED_LOCKED',
        cpus='36-47',max_children=12,training_affinities_untouched=True,protocol_games_models_rng_unchanged=True,
        scheduling_adapter_sha256=sha(__file__),unix=time.time()))
    S.main()
