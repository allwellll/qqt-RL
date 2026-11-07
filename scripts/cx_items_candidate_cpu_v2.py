"""Scheduling-only adapter: use eight free evaluation cores; exact games unchanged."""
import time
from scripts import cx_items_candidates_v1 as S
from scripts.population_run_state import atomic_json,sha

if __name__=='__main__':
    S.CPU['CC_STRONG_CPUS']='36-43'
    atomic_json(S.ROOT/'CPU_SCHEDULING.json',dict(status='EIGHT_SHARED_LOCKED_EVAL_CORES',
        cpus='36-43',previous_pool='24-31 occupied by existing evaluation queue',
        max_children=8,training_affinities_untouched=True,protocol_games_models_rng_unchanged=True,
        scheduling_adapter_sha256=sha(__file__),unix=time.time()))
    S.main()
