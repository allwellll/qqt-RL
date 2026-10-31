"""Materialize isolated source screen without starting GPU or old queues."""
import shutil
import time
from scripts.dagger_rl_r9_run import B, R, read
from scripts.population_run_state import atomic_json, sha


def prepare():
    prior = R/'runs/dagger_rl_20261024/r6_opponent_candidate'
    p = read(B/'PROTOCOL.json')
    assert p['arms']==['control','lr_quarter']
    assert p['learning_rate_by_arm']==dict(control=7.5e-5,lr_quarter=1.875e-5)
    assert p['dagger_coefficient_by_arm']==dict(control=.2,lr_quarter=.2)
    assert p['anchor_coefficient']==.05 and p['total_environment_steps']==1572864
    assert p['total_optimizer_steps']==6144
    for folder in ['inputs','cpu_eval_inputs','inputs/sealed_sources']:
        (B/folder).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prior/'inputs/parent.pkl', B/'inputs/parent.pkl')
    assert sha(B/'inputs/parent.pkl')==p['parent_input_sha256']
    cp = R/p['parent']['checkpoint']
    assert sha(cp/'policy.pt')==p['parent']['policy_sha256']
    assert sha(cp/'state.pkl')==p['parent']['state_sha256']
    for name in ['spawns.json','regions.json']:
        shutil.copyfile(prior/'cpu_eval_inputs'/name, B/'cpu_eval_inputs'/name)
    ev = dict(read(prior/'CPU_EVAL_PROTOCOL.json'),
              holdout_seed_start=p['evaluation']['natural_holdout_seed'])
    assert ev['spawns_sha256']==sha(B/'cpu_eval_inputs/spawns.json')
    assert ev['metric_sha256']==sha(R/'scripts/dagger_rl_metrics_v2.js')
    atomic_json(B/'CPU_EVAL_PROTOCOL.json', ev)
    names = ['ACTION_CHAIN_AUDIT.json','TEACHER_CLOSED_LOOP.json',
             'audit_student_opportunities.pkl','audit_native_branches.pkl']
    for name in names:
        shutil.copyfile(prior/name, B/name)
    atomic_json(B/'AUDIT_REUSE_RECEIPT.json', dict(
        new_independent_examples=0, source_namespace=str(prior.relative_to(R)),
        files={n:sha(B/n) for n in names}, teacher_sha256=sha(R/'scripts/dagger_rl_oracle.py'),
        scope='unchanged teacher and native causal train-state proof; not new capability evidence'))
    sources = {str(path.relative_to(R)):sha(path)
               for directory in ['scripts','jax_bomb']
               for path in sorted((R/directory).glob('*.py'))}
    for name in ['scripts/dagger_rl_route.js','scripts/dagger_rl_metrics.js',
                 'scripts/dagger_rl_metrics_v2.js','scripts/dagger_rl_metrics_v2_preload.js',
                 'web/sim.js','web/bun_hunter_bot.js','web/review_protocol.js',
                 'tests/test_dagger_rl_opponent_mix.py','tests/test_dagger_rl_r9_step_scale.py']:
        sources[name]=sha(R/name)
    for name in sources:
        target=B/'inputs/sealed_sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(R/name, target)
    atomic_json(B/'SOURCE_SEAL.json', dict(sources=sources, protocol_sha256=sha(B/'PROTOCOL.json')))
    old_pause=read(R/'runs/dagger_rl_20261024/r7_update_mechanism/SCOPED_GPU_AUTHORIZATION.json')['old_global_pause_sha256']
    assert sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json')==old_pause
    atomic_json(B/'SCOPED_GPU_AUTHORIZATION.json', dict(
        authorized_by='explicit_user_instruction', old_continuous_resume=False,
        old_global_pause_sha256=old_pause,
        bounded_protocol_sha256=sha(B/'PROTOCOL.json')))
    atomic_json(B/'MATERIALIZED.json', dict(passed=True, formal_not_started=True,
                scientific_selection_required=True, old_continuous_resume=False, unix=time.time()))
    print(p['id'], sha(B/'PROTOCOL.json'))


if __name__=='__main__':
    prepare()
