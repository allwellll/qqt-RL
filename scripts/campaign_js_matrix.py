"""Matched production-JS natural full games, independent of GPU training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.population_run_state import atomic_json,lock,sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('protocol');a=p.parse_args()
    spec=json.loads(Path(a.protocol).read_text());out=Path(spec['namespace']);out.mkdir(parents=True,exist_ok=True)
    guard=lock(out/'owner.lock');global_guard=lock(Path(spec['JS_unique_lock']))
    for file,expected in spec['source_sha256'].items():assert sha(file)==expected,file
    rows=[]
    for name,model in spec['models'].items():
        assert sha(model['model_json'])==model['model_json_sha256']
        assert json.loads(Path(model['model_json']).read_text())['meta']['checkpoint_sha256']==model['policy_sha256']
        for mode in spec['action_modes']:
            for start in range(0,spec['games_per_condition'],8):
                end=min(start+8,spec['games_per_condition']);folder=out/name/mode/f'{start:04d}';folder.mkdir(parents=True,exist_ok=True)
                target=folder/'RESULT.json'
                if not target.exists():
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JE10_ROUTE_TASK='natural_full2400',
                        JE10_ROUTE_GAMES=str(spec['games_per_condition']),JE10_ROUTE_START=str(start),
                        JE10_ROUTE_END=str(end),JE10_ROUTE_SEED_START=str(spec['seed_start']))
                    subprocess.run(['node','scripts/expert_validation_js_eval.js',mode,model['model_json'],str(target),spec['regions']],env=env,check=True)
                result=json.loads(target.read_text());assert result['passed'] and result['model_json_sha256']==model['model_json_sha256']
                episodes=result['results']['natural_full2400']['episodes'];assert len(episodes)==end-start
                assert [e['seed'] for e in episodes]==list(range(spec['seed_start']+start,spec['seed_start']+end))
                rows.extend(dict(model=name,action_mode=mode,episode=e) for e in episodes)
                atomic_json(out/'PROGRESS.json',dict(completed_games=len(rows),total_games=len(spec['models'])*2*spec['games_per_condition'],last_model=name,action_mode=mode,unix=time.time()))
    atomic_json(out/'SUMMARY.json',dict(protocol=a.protocol,protocol_sha256=sha(a.protocol),rows=rows,
        engine='production JS web/sim.js',opponent='JS Hunter normal',teacher=False,
        training_reward_status='not recorded; observable reconstruction is not actual JAX reward',
        missing_metrics=['own_alive/actionable ticks and complete causal item branch net-benefit require follow-up'],no_peak_selection=True))
    atomic_json(out/'COMPLETE.json',dict(passed=True,total_games=len(rows),summary_sha256=sha(out/'SUMMARY.json'),scope='Fixed natural JS normal-Hunter matrix only; short combat Hunter and Python Tactical separate'))


if __name__=='__main__':main()
