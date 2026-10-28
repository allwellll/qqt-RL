"""Read-only Stage C evidence audit; writes only this round's dated report."""
import json,hashlib,fcntl,subprocess
from pathlib import Path
import numpy as np
BASE=Path('runs/je09_bun_20261015/stage_c_return_set_pfsp8_v8');ARM='bun392_bridge_pfsp8'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compact(x):return {k:v for k,v in x.items() if k!='episodes'}
def gated(r):return all(r[k]>=v for k,v in dict(first_pickup=13,progress25=12,home_half=8,capture=4).items())
def stage(out,seeds):
 result={}
 for name,key in [('STAGE_A','stage_a_main'),('STAGE_A_HOLDOUT','stage_a_holdout'),('STAGE_B','stage_b_main'),('STAGE_B_HOLDOUT','stage_b_holdout')]:
  p=out/(name+'.json');d=read(p);assert d['passed'] and d['assist'] is False and d['model_json_sha256']==sha(out/'model/model.json')
  episodes=d['results']['episodes'];assert len(episodes)==16 and [e['seed'] for e in episodes]==[seeds[key]+i*104729 for i in range(16)]
  result[name]=compact(d['results'])
  if name.startswith('STAGE_B'):
   funnel=['eligible_drop','toward_drop','arrival','first_pickup','pickup_return','progress25','progress50','home_half','capture']
   earliest=[next((k for k in funnel if not e[k]),'passed') for e in episodes]
   result[name].update(gate=gated(d['results']),earliest_funnel={k:earliest.count(k) for k in set(earliest)},reverse_cell_transitions=sum(e['reverse_cell_transitions'] for e in episodes),unique_return_cells=sum(e['unique_return_cells'] for e in episodes),downhill_cell_transitions=sum(e['downhill_cell_transitions'] for e in episodes),per_game=[dict(game=e['game'],seed=e['seed'],earliest=e['earliest'],first_pickup=e['first_pickup'],pickup_return=e['pickup_return'],progress25=e['progress25'],progress50=e['progress50'],home_half=e['home_half'],capture=e['capture'],reverse_cell_transitions=e['reverse_cell_transitions'],unique_return_cells=e['unique_return_cells'],initial=e['initial'],trace=e['trace']) for e in episodes])
 return result

def main():
 d=read(BASE/'ROUND_DECLARATION.json');p=read(BASE/'plan.json');lineage=read(BASE/'PARENT_LINEAGE.json');parent=Path(lineage['source_checkpoint']);hashes={}
 for name,h in lineage['source_hashes'].items():assert sha(parent/name)==h
 seal=read(BASE/'SOURCE_SEAL.json');assert sha(BASE/'plan.json')==seal['plan_sha256']
 for name,h in seal['sources'].items():assert sha(BASE/'inputs/sealed_sources'/name)==h and sha(Path(name))==h
 assert p['stage_b_reward']==dict(window=120,scale=2.) and p['combat_fraction']==.6 and p['bridge_fraction']==.25
 assert sha(BASE/'inputs/retention_parent.pkl')==sha(parent.parents[2]/'inputs/initial.pkl')
 for file in ['TDD_SUMMARY','SEED_AUDIT','RETURN_PRODUCTION_GRADIENT','EXPERT_PRODUCTION_GRADIENT','CPU_BRIDGE_PRODUCTION']:
  assert read(BASE/'validation'/(file+'.json'))['passed']
 gradient=read(BASE/'validation/RETURN_PRODUCTION_GRADIENT.json');assert gradient['raw_auxiliary_gradient']['actor']>0 and gradient['raw_auxiliary_gradient']['critic']==0 and gradient['actor_counterfactual_l2']>0
 assert gradient['train_trajectory']['recovery_rows_after_reverse']>0
 assert read(BASE/'PRODUCTION_SMOKE_COMPLETE.json')['passed']
 smoke=BASE/'production_smoke_stage_c';replay=read(smoke/'state/FRESH_GPU_REPLAY.json');restore=read(BASE/'validation/FORMAL_RESTORE.json');assert replay['next_update_exact'] and replay['full_state_digest']==restore['state_digest']
 assert read(BASE/'TRAINING_COMPLETE.json')['generation']==8 and read(BASE/'authority_je09_v1/EVALUATION_COMPLETE.json')['completed_generations']==[4,8]
 events=[json.loads(l) for l in (BASE/'formal'/ARM/'events.jsonl').read_text().splitlines()];assert len(events)==8 and all(e['finite'] and e['nonzero_update'] for e in events)
 for g in [0,1,4,8]:
  cp=BASE/'formal'/ARM/f'gen{g:06d}';m=read(cp/'COMMITTED.json');assert m['replicas_exact'] and m['adam_steps']==[g*128]*8
  for name,key in [('state.pkl','state_sha256'),('policy.pt','policy_sha256'),('ema_policy.pt','ema_policy_sha256')]:assert sha(cp/name)==m[key]
 final=read(BASE/'state/RESTORE_MONITOR.json');assert final['generation']==8 and final['full_state_digest']==read(BASE/'formal'/ARM/'gen000008/COMMITTED.json')['full_state_digest']
 authority=BASE/'authority_je09_v1';baseline=stage(authority/'baseline_parent',d['seeds']);points=[]
 for g in [4,8]:
  label=f'{ARM}_g{g:06d}';out=authority/label;complete=read(out/'JE09_COMPLETE.json');assert complete['passed'] and complete['assist'] is False
  for name,key in [('CONTINUOUS_BRIDGE.json','bridge_sha256'),('COMPLETE_WITH_BUN_TASKS.json','authority_sha256'),('STAGE_A.json','stage_a_sha256'),('STAGE_A_HOLDOUT.json','holdout_sha256')]:assert sha(out/name)==complete[key]
  metrics=read(out/'COMPLETE.json');assert metrics['policy_sha256']==sha(BASE/'formal'/ARM/f'gen{g:06d}'/'policy.pt')
  for name,h in metrics['shards'].items():assert sha(out/name)==h
  additional=read(out/'COMPLETE_WITH_BUN_TASKS.json')
  for name,h in additional['additional_shards'].items():assert sha(out/name)==h
  topology=BASE/'topology_diagnostics_v1'/label/'COMPLETE.json';assert sha(topology)==complete['diagnostics_sha256']
  points.append(dict(generation=g,stage=stage(out,d['seeds']),hunter=metrics['hunter'],tactical=metrics['tactical'],carry={k:compact(v) for k,v in read(topology)['results'].items()},legacy={k:compact(v) for k,v in read(out/'CONTINUOUS_BRIDGE.json')['results'].items()}))
 for f in [BASE/'ROUND_DECLARATION.json',BASE/'SOURCE_SEAL.json',BASE/'PARENT_LINEAGE.json',*BASE.glob('validation/*.json'),*authority.glob('*/STAGE*.json'),*authority.glob('*/JE09_COMPLETE.json')]:hashes[str(f)]=sha(f)
 counts=np.sum([e['actual_scene_valid_ticks'] for e in events],0);combat=float(counts[0]/counts.sum());assert combat>=.4
 previous=np.zeros((11,6));pool=np.zeros_like(previous)
 for e in events:
  current=np.asarray(e['pfsp_counts']);pool+=(current-.95*previous)/.05;previous=current
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True);gpurows=[list(map(int,l.split(','))) for l in gpu.splitlines()];assert all(r[1]==0 for r in gpurows)
 for file in [BASE/'gpu_worker.lock',BASE/'formal'/ARM/'worker.lock',authority/'owner.lock']:
  with file.open() as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
 result=dict(passed=True,namespace=str(BASE),experiment_id=d['experiment_id'],scope='evidence completeness is distinct from promotion; curriculum real drop is distinct from full natural chain',budget=dict(formal_generations=8,formal_ticks=16777216,technical_gpu_ticks=4194304,consumed=True),return_expert=p['return_expert'],baseline=baseline,points=points,production=gradient,scan=read(BASE/'validation/STAGE_C_SCAN.json'),combat_actual_fraction=combat,bridge_actual_fraction=float(counts[9]/counts.sum()),dynamic_pfsp_fraction=float(pool[[0,7,8,9,10],0].sum()/pool[:,0].sum()),compute_seconds=events[-1]['compute_seconds'],wall_seconds=events[-1]['wall_seconds'],final_digest=final['full_state_digest'],hashes=hashes,terminal=dict(gpu_memory_mib=gpurows,locks_released=True))
 target=Path('reports/je09_bun_20261015/stage_c_audit_20261015.json');target.write_text(json.dumps(result,indent=2)+'\n');print(target)
if __name__=='__main__':main()
