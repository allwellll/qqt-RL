"""Verify six immutable V2 endpoints; no repeat training or peak selection."""
import json
from pathlib import Path
import sys
from scripts.population_run_state import atomic_json,sha


def main():
    path=Path(sys.argv[1]);p=json.loads(path.read_text());v=json.loads(Path(p['training_protocol']).read_text());base=Path(v['namespace']);rows=[]
    for seed in v['paired_seeds']:
        for arm in v['arms']:
            out=base/'formal'/f'{arm}_s{seed}';done=json.loads((out/'COMPLETE.json').read_text());cp=Path(done['checkpoint'])
            assert done['passed'] and done['updates']==320 and done['actual_environment_steps']==5242880
            assert sha(cp/'state.pkl')==done['state_sha256'] and sha(cp/'policy.pt')==done['policy_sha256']
            rows.append(dict(seed=seed,arm=arm,**done))
    atomic_json(base/'six_endpoint_acceptance.json',dict(rows=rows,passed=True,scope='Full fixed training budgets and hashes; asynchronous evaluation tracked independently'))
    atomic_json(base/'SIX_RUN_COMPLETE.json',dict(passed=True,rows=6,actual_environment_steps=31457280,summary_sha256=sha(base/'six_endpoint_acceptance.json')))


if __name__=='__main__':main()
