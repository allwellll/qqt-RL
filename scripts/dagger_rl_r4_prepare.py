"""Materialize conditional three-seed eviction screen; no training dispatch."""
import shutil,time
from scripts.dagger_rl_r4_run import B,R,read
from scripts.population_run_state import atomic_json,sha


def prepare():
 prior=R/'runs/dagger_rl_20261024/r3_confirmation';p=read(B/'PROTOCOL.json')
 assert p['anchor_coefficient']==.05 and p['total_environment_steps']==1572864
 assert p['total_optimizer_steps']==6144 and p['arms']==['control','reservoir']
 assert sha(R/'runs/dagger_rl_20261024/r1v2/ANCHOR_CANDIDATE_TRAIN_SCALE.json')==p['train_scale_sha256']
 for folder in ['inputs','cpu_eval_inputs','inputs/sealed_sources']:(B/folder).mkdir(parents=True,exist_ok=True)
 shutil.copyfile(prior/'inputs/parent.pkl',B/'inputs/parent.pkl')
 assert sha(B/'inputs/parent.pkl')==p['parent_input_sha256']
 cp=R/p['parent']['checkpoint']
 assert sha(cp/'policy.pt')==p['parent']['policy_sha256'] and sha(cp/'state.pkl')==p['parent']['state_sha256']
 for name in ['spawns.json','regions.json']:shutil.copyfile(prior/'cpu_eval_inputs'/name,B/'cpu_eval_inputs'/name)
 assert read(B/'cpu_eval_inputs/spawns.json')['seed']==p['evaluation']['hunter_seed']
 protocol=dict(read(prior/'CPU_EVAL_PROTOCOL.json'),holdout_seed_start=p['evaluation']['natural_holdout_seed'])
 assert protocol['spawns_sha256']==sha(B/'cpu_eval_inputs/spawns.json')
 atomic_json(B/'CPU_EVAL_PROTOCOL.json',protocol)
 files=['ACTION_CHAIN_AUDIT.json','TEACHER_CLOSED_LOOP.json','audit_student_opportunities.pkl','audit_native_branches.pkl']
 for name in files:shutil.copyfile(prior/name,B/name)
 atomic_json(B/'AUDIT_REUSE_RECEIPT.json',dict(new_independent_examples=0,source_namespace=str(prior.relative_to(R)),files={name:sha(B/name) for name in files},teacher_sha256=sha(R/'scripts/dagger_rl_oracle.py'),scope='exact unchanged banana teacher and causal input proof, not new capability evidence'))
 atomic_json(B/'MODE_IDENTITY.json',dict(mode=p['mode'],parent_policy_sha256=p['parent']['policy_sha256'],no_other_game_lineage=True))
 # Keep a complete immutable source copy so later rounds cannot silently alter it.
 sources={str(path.relative_to(R)):sha(path) for directory in ['scripts','jax_bomb'] for path in sorted((R/directory).glob('*.py'))}
 for name in ['scripts/dagger_rl_route.js','scripts/dagger_rl_metrics.js','web/sim.js','web/bun_hunter_bot.js','web/review_protocol.js']:
  sources[name]=sha(R/name)
 for name in sources:
  target=B/'inputs/sealed_sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/name,target)
 atomic_json(B/'SOURCE_SEAL.json',dict(sources=sources,protocol_sha256=sha(B/'PROTOCOL.json')))
 atomic_json(B/'SCOPED_GPU_AUTHORIZATION.json',dict(authorized_by='explicit_user_instruction',old_continuous_resume=False,old_global_pause_sha256=sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json'),bounded_protocol_sha256=sha(B/'PROTOCOL.json')))
 atomic_json(B/'MATERIALIZED.json',dict(passed=True,unix=time.time(),formal_not_started=True,old_continuous_resume=False))
 print(p['id'],sha(B/'PROTOCOL.json'))
if __name__=='__main__':prepare()
