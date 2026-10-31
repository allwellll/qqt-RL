"""Bounded CPU backfill of requested pre-acquisition safety on frozen cases."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,time,subprocess,concurrent.futures,shutil
from scripts.combat_item_run import B,R,read,notify
from scripts.population_run_state import atomic_json,sha,lock
from scripts.retention_route_eval import merge_episodes


def evaluate(cp):
 out=cp/'phase_audit';out.mkdir(exist_ok=True)
 def shard(start):
  folder=out/f'{start:03d}';folder.mkdir(exist_ok=True);target=folder/'RESULT.json'
  if not target.exists():
   env=dict(os.environ,JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='64',JE10_ROUTE_SEED_START='2026102700',JE10_ROUTE_START=str(start),JE10_ROUTE_END=str(start+8))
   subprocess.run(['node','scripts/combat_item_phase_audit.js',str((cp/'retention/model/model.json').relative_to(R)),str(target.relative_to(R)),str((B/'cpu_eval_inputs/regions.json').relative_to(R))],env=env,check=True,stdout=(folder/'run.log').open('a'),stderr=subprocess.STDOUT)
  return read(target)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:parts=list(pool.map(shard,range(0,64,8)))
 result=merge_episodes('natural_full2400',parts);before=read(cp/'retention/natural_holdout64.json')['results']['natural_full2400']['episodes'];after=result['results']['natural_full2400']['episodes']
 assert len(before)==len(after)==64
 for a,c in zip(before,after):assert a=={k:v for k,v in c.items() if k not in ['phase_safety','pre_first_acquisition_deaths']},'passive metric parity failure'
 original=next((cp/'holdout/retention').glob('fixed-*.trace.json'));repeated=next((out/'000').glob('fixed-*.trace.json'))
 x,y=read(original),read(repeated);assert x==y,'fixed first episode action/state/diagnostic parity failure'
 atomic_json(out/'RESULT.json',result);atomic_json(out/'COMPLETE.json',dict(passed=True,policy_sha256=sha(cp/'policy.pt'),all64_recorded_episode_metrics_exact=True,fixed_first_action_state_trace_exact=True,unique_new_cases=0,repeat_cases=64,no_parameter_selection=True))
 return cp


def run():
 unique=lock(B/'phase_audit.lock');global_unique=lock(R/'runs/je10_bun_20261018/evaluation.lock')
 assert read(B/'CPU_EVALUATION_COMPLETE.json')['passed']
 cps=[B/'formal/no_update/u000']+[B/'formal'/f'{a}_s{s}/u032' for a in ['control','combat','glue','banana'] for s in read(B/'PROTOCOL.json')['paired_seeds']]
 canonical={};aliases={}
 for cp in cps:
  h=sha(cp/'policy.pt')
  if h in canonical:aliases[cp]=canonical[h]
  else:canonical[h]=cp
 pending=[cp for cp in canonical.values() if not (cp/'phase_audit/COMPLETE.json').exists()]
 atomic_json(B/'PHASE_AUDIT_STATUS.json',dict(stage='CPU_REPEATS',unique_owner=True,parallel_models=4,pending=[str(cp.relative_to(B)) for cp in pending],new_unique_eval_cases=0,controls_gpu=False))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for cp in pool.map(evaluate,pending):notify(f'固定CPU阶段安全补记完成：{cp.parent.name}，原64局计数及固定首局动作/hash精确一致；不新增独立样本。')
 for cp,source in aliases.items():
  original=read(cp/'retention/natural_holdout64.json')['results']['natural_full2400']['episodes'];truth=read(source/'retention/natural_holdout64.json')['results']['natural_full2400']['episodes'];assert original==truth
  (cp/'phase_audit').mkdir(exist_ok=True);shutil.copyfile(source/'phase_audit/RESULT.json',cp/'phase_audit/RESULT.json')
  atomic_json(cp/'phase_audit/COMPLETE.json',dict(passed=True,alias=str(source.relative_to(B)),same_policy_sha256=sha(source/'policy.pt'),original_all64_episode_metrics_exact=True,new_unique_eval_cases=0,no_repeat_execution=True))
 rows=[]
 for cp in cps:
  es=read(cp/'phase_audit/RESULT.json')['results']['natural_full2400']['episodes'];stages={}
  for e in es:
   for phase,values in e['phase_safety'].items():
    dest=stages.setdefault(phase,{})
    for k,v in values.items():dest[k]=dest.get(k,0)+v
  assert sum(x['deaths'] for x in stages.values())==sum(x['death'] for x in es)
  rows.append(dict(checkpoint=str(cp.relative_to(B)),stages=stages,pre_first_acquisition_deaths=sum(e['pre_first_acquisition_deaths'] for e in es)))
 atomic_json(B/'PHASE_SAFETY_COMPLETE.json',dict(passed=True,rows=rows,canonical_models=len(canonical),repeat_episodes=64*len(canonical),new_unique_cases=0,selection='all fixed final holdout cases; diagnostic backfill requested by original protocol, no tuning',stage_scope='decision before each action; pre-first-acquisition counts all phases before first real pickup, not synonymous with combat phase',all64_recorded_metric_parity=True,fixed_first_full_trace_parity=True,old_continuous_resume=False))
 notify('CPU阶段安全补记终态：固定13模型完整覆盖，原指标逐局一致，新增独立样本0；GPU无新训练。')


if __name__=='__main__':run()
