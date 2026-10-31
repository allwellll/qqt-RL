"""Pre-register the isolated R18 item-chain teacher experiment."""
import copy, shutil, time
from pathlib import Path
from scripts.dagger_rl_r18_run import B, R, read
from scripts.population_run_state import atomic_json, sha

PRIOR = R/'runs/dagger_rl_20261025/r16_retention_schedule'
PARENT = PRIOR/'formal/no_update/u000'
SEEDS = [2026102661, 2026102662, 2026102663]

def prepare():
    assert not (B/'PROTOCOL.json').exists(), 'R18 namespace already registered'
    parent_meta = read(PARENT/'COMMITTED.json')
    assert parent_meta['update'] == 0 and sha(PARENT/'state.pkl') == parent_meta['state_sha256']
    old = read(PRIOR/'PROTOCOL.json')
    p = dict(
        id='DAGGER-RL-20261025-R18-ITEM-CHAIN', mode='causal_paired_three_arm',
        namespace='r18_item_chain', parent=dict(checkpoint=str(PARENT.relative_to(R)),
            state_sha256=sha(PARENT/'state.pkl'), policy_sha256=sha(PARENT/'policy.pt')),
        parent_input_sha256=sha(PARENT/'state.pkl'), arms=['ppo_anchor','sequential_anchor','joint_anchor'],
        paired_seeds=SEEDS, updates=32, switch_update=16,
        single_variable='without a certified complete item plan, route supervision removes only ability=2 release; certified item plans retain the full set',
        schedule=dict(ppo_anchor='fresh PPO/value/entropy+parent KL, no DAgger adoption',
                      sequential_anchor='u1..16 DAgger+parent KL with Critic and moments frozen; u17..32 fresh PPO/value/entropy+parent KL, DAgger=0',
                      joint_anchor='fresh PPO/value/entropy+DAgger+parent KL throughout'),
        common='R16 reward, combat handling, item oracle, movement/action masks, curriculum, opponents, network, optimizer, LR, bank, reset mixture and budget; no banana/glue/reward change; no synthetic carry',
        hypothesis='Removing uncertified route release labels improves action-chain validity and avoids learning release as ordinary movement, while certified item-chain supervision remains available.',
        falsification='If route release opportunity telemetry changes but natural release/complete-chain or safety does not improve against matched controls, the label omission is not sufficient; do not tune on holdout.',
        oracle='route set-valued safety labels plus item oracle; item oracle overrides only when complete public-state placement/leave/contact/settlement plan is certified',
        train_mixture=old['train_mixture'], opponents=dict(current=0,ring=0,fixed=1.,identity='JAX rule3'),
        envs=64, steps=128, environment_steps_per_run=524288, total_environment_steps=4718592,
        optimizer_steps_per_run=1024, total_optimizer_steps=9216, checkpoints=[0,1,8,16,17,24,32],
        optimizer=old['optimizer'], lr=old['lr'], gamma=old['gamma'], gae_lambda=old['gae_lambda'], clip=old['clip'], value=old['value'], entropy=old['entropy'], gradient_clip=old['gradient_clip'], epochs=old['epochs'], minibatch=old['minibatch'], anchor_coefficient=old['anchor_coefficient'],
        dagger_coefficient_by_arm=dict(ppo_anchor=0., sequential_anchor=.2, joint_anchor=.2), learning_rate_by_arm={a:old['lr'] for a in ['ppo_anchor','sequential_anchor','joint_anchor']}, auxiliary_greedy={a:False for a in ['ppo_anchor','sequential_anchor','joint_anchor']}, release_qualification={a:False for a in ['ppo_anchor','sequential_anchor','joint_anchor']}, memory=copy.deepcopy(old['memory']),
        bank_source=old['bank_source'], bank_sha256=old['bank_sha256'], bank_limits=old['bank_limits'],
        technical_smoke=dict(total_executed_environment_steps=3072,total_executed_optimizer_steps=18,adopted_formal_steps=0,checks='three branches finite, real Actor/Critic/ability updates, sequential DAgger Critic/moments frozen, fresh full state/RNG/Adam/history restore exact; no capability gate'),
        evaluation=copy.deepcopy(old['evaluation']), reporting='natural dev/holdout64; Python Tactical64; real JS Hunter difficult/normal/easy64; Bank and Bank-P separately; route/item chain opportunity denominators, release logits/sample/greedy/complete settlement, combat/safety and worst seed; no peak selection',
        successor='After three paired runs, continue only with a separately preregistered bounded question chosen from first-deviation and action-chain evidence; never restore old O5/continuous/reserve queues.',
        stopping='fixed nine run32 budgets; only technical failure or explicit stop interrupts; no midpoint capability gate',
        old_r17_exclusion='R17 attempt03 EXCLUDED_INVALID; no state or result reused', unix=time.time())
    p['evaluation']['natural_holdout_seed']=2026111800
    p['evaluation']['checkpoints']='immutable0/1/8/16/17/24/32; same protocol dev/holdout, Tactical, Hunter, Bank/P'
    B.mkdir(parents=True, exist_ok=True)
    for d in ['inputs','cpu_eval_inputs','inputs/sealed_sources','cache','CPU_QUEUE','notifications']:
        (B/d).mkdir(parents=True, exist_ok=True)
    atomic_json(B/'PROTOCOL.json', p)
    shutil.copyfile(PARENT/'state.pkl', B/'inputs/parent.pkl')
    assert sha(B/'inputs/parent.pkl') == p['parent_input_sha256']
    for n in ['spawns.json','regions.json']:
        shutil.copyfile(PRIOR/'cpu_eval_inputs'/n, B/'cpu_eval_inputs'/n)
    ev=copy.deepcopy(read(PRIOR/'CPU_EVAL_PROTOCOL.json')); ev['holdout_seed_start']=2026111800
    atomic_json(B/'CPU_EVAL_PROTOCOL.json',ev)
    # Seal only current source files plus the newly tested adapter; no run/checkpoint files enter Git.
    names=['scripts/dagger_rl_r18_run.py','scripts/dagger_rl_r18_train.py','scripts/dagger_rl_r18_cpu.py','scripts/dagger_rl_r18_bank.py','scripts/dagger_rl_r18_report.py','scripts/dagger_rl_r18_diagnostics.py','scripts/dagger_rl_item_chain_teacher.py','scripts/dagger_rl_r16_train.py','scripts/dagger_rl_r16_run.py','scripts/dagger_rl_retention_loop.py','scripts/dagger_rl_retention_schedule.py','scripts/dagger_rl_r12_train.py','scripts/dagger_rl_r10_train.py','scripts/dagger_rl_r4_train.py','scripts/combat_item_engine.py','scripts/dagger_rl_oracle.py','scripts/dagger_rl_release_qualification.py','tests/test_dagger_rl_item_chain_teacher.py']
    sources={n:sha(R/n) for n in names}
    atomic_json(B/'SOURCE_SEAL.json',dict(sources=sources,protocol_sha256=sha(B/'PROTOCOL.json')))
    for n,h in sources.items():
        d=B/'inputs/sealed_sources'/n; d.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(R/n,d)
    for src in PRIOR.glob('cache/*'):
        if src.is_file(): shutil.copyfile(src,B/'cache'/src.name)
    atomic_json(B/'COMPILED_CACHE_REUSE.json',dict(compiler_files_only=True,no_state_or_Adam=True))
    pause=sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json')
    atomic_json(B/'SCOPED_GPU_AUTHORIZATION.json',dict(authorized_by='explicit_user_instruction',protocol_sha256=sha(B/'PROTOCOL.json'),old_global_pause_sha256=pause,old_continuous_resume=False,old_queues_remain_paused=True))
    atomic_json(B/'PREREGISTRATION.json',dict(passed=True,protocol_sha256=sha(B/'PROTOCOL.json'),formal_updates=0,selection_pending=True))
    print('R18 prepared',sha(B/'PROTOCOL.json'))

if __name__=='__main__': prepare()
