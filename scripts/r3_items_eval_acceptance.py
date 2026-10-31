"""Full production-JS parent/zero-extension comparison, no training updates."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import json
import pickle
import subprocess
import time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha

OUT=Path('runs/r3_items_route_20261030')


def main():
    from scripts import r3_items_observation as O
    from scripts.r3_items_export import export_weights
    from scripts.je08_bun_authority import export
    receipt=json.loads((OUT/'R3_PARENT.json').read_text());cp=Path(receipt['pinned_checkpoint'])
    old=OUT/'eval_abi/old';old_model=export(cp/'policy.pt',old.resolve())
    parent=pickle.load((cp/'policy.pt').open('rb'));p=O.migrate(parent)
    new=OUT/'eval_abi/new';new.mkdir(parents=True,exist_ok=True)
    artifact=new/'diagnostic_zero_extended_actor.pkl'
    with artifact.open('wb') as f:pickle.dump(p,f,protocol=5)
    record=export_weights(p,new,dict(diagnostic_only=True,not_trained_checkpoint=True,policy_sha256=sha(artifact),
        source_policy_sha256=receipt['source']['policy_sha256'],source=receipt['source']['checkpoint']))
    models={'old':str(old_model.relative_to(Path.cwd())),'new':record['model_json']}
    v=json.loads((OUT/'BASELINE_PROTOCOL.json').read_text())['conditions']['hard'];results={};began=time.time()
    for key,model in models.items():
        results[key]={}
        for seat in (0,1):
            dest=OUT/'eval_abi'/f'{key}_P{seat}.json';side=Path(str(dest)+'.outcome.json')
            if not side.exists():
                env=dict(os.environ,QQT_STOCK_RULE='stock-window-v1',BUN_V2_RULE='bun-native-v2',BUN_V2_TIER='hard',BUN_V2_OPPONENT='bun.hunter',
                    JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='2',JE10_ROUTE_START=str(seat),JE10_ROUTE_END=str(seat+1),JE10_ROUTE_SEED_START=str(v['seed']))
                with dest.with_suffix('.log').open('ab') as log:
                    subprocess.run(['taskset','-c',str(188+seat),'node','scripts/r3_items_candidate_eval.js','greedy',model,str(dest),
                        'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',v['spawn_manifest']],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            d=json.loads(side.read_text());assert len(d['episodes'])==1 and d['episodes'][0]['done'] and not d['teacher']
            e=d['episodes'][0];assert e['seat']==seat and 'resolved_bombs' in e['r3']
            results[key][f'P{seat}']=dict(seed=e['seed'],ticks=e['ticks'],winner=e['winner'],joint_actions_sha256=e['r3']['joint_actions_sha256'],file=str(side),sha256=sha(side))
    assert all(results['old'][s]['joint_actions_sha256']==results['new'][s]['joint_actions_sha256'] for s in ('P0','P1'))
    atomic_json(OUT/'EVALUATION_ABI_ACCEPTANCE.json',dict(status='PASS',production_JS_full_games=4,seats=[0,1],teacher=False,
        zero_extended_parent_full_game_joint_actions_exact=True,results=results,wall_seconds=time.time()-began,
        new_input_schema=[25,13,15,37],not_ability_improvement_evidence=True))
    print('production JS evaluation ABI and full-game parent action parity PASS',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(OUT/'EVALUATION_ABI_BLOCKED.json',dict(status='BLOCKED',reason='inspect EVALUATION_ABI.log and eval_abi logs',unix=time.time()))
        raise
