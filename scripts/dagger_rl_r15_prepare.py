"""Materialize the bounded candidate only after an explicit full-evidence choice."""
import shutil
import time
from pathlib import Path
from scripts.dagger_rl_r15_run import B,R,read
from scripts.population_run_state import atomic_json,sha


def prepare():
    prior=R/'runs/dagger_rl_20261024/r14_item_observation'
    assert read(prior/'COMPLETE_EVIDENCE_CONSUMED.json')['passed']
    choice=read(prior/'NEXT_SELECTED_EXPERIMENT.json')
    assert choice['execute_carry_mask_question'] is True
    assert choice['R14_evidence_sha256']==sha(prior/'COMPLETE_EVIDENCE_SYNTHESIS.json')
    assert choice['R14_complete_receipt_sha256']==sha(prior/'COMPLETE_EVIDENCE_CONSUMED.json')
    assert choice['common_item_observation_alignment'] is False
    assert not (B/'PROTOCOL.json').exists(),'do not overwrite an existing experiment'
    assert read(prior/'NEXT_MASK_PREPARATION_ACCEPTANCE.json')['passed']
    p=read(prior/'PROTOCOL.json')
    arms=['legacy_mask','actual_mask'];weights={a:.2 for a in arms}
    p.update(id='DAGGER-RL-20261025-R15-CARRY-MASK',arms=arms,
        paired_seeds=[2026102641,2026102642,2026102643],
        common_item_observation_alignment=False,
        treatment='collector P0 full-lesson alive free genuine carrier movement legality: legacy growth-speed probe vs existing actual-physics-speed probe; only train.legal_mask binding',
        single_variable='carry-only learner movement mask in both streams and teacher inputs; env.legal_mask/step/reward untouched',
        hypothesis='known production/JAX carrier mask-speed mismatch may suppress real return directions; isolate its contribution, not assume all degradation is explained',
        controls='same clean r19 saved params/Adam, common legacy ch7, reward, opponents, resets, labels, architecture, reservoir, learning rate and budgets; no adoption of rejected ch7 treatment or trained endpoint',
        common='joint on-policy PPO, DAgger .2, combat parent-KL .05; reservoir8192, item/route4096/4096, old/new .5/.5, fixed JAX rule3',
        opponents=dict(current=0,ring=0,fixed=1.,legacy_mask='JAX rule3 100%',actual_mask='JAX rule3 100%'),
        prior_evidence='full R14 paired natural/retention/fixed Bank evidence and independent26-state mask-input/832 same-action transition proof; R13 numerical boundary remains disclosed',
        preregistration_stage='after complete R14 evidence and before R15 smoke/formal; no holdout scale selection',
        conditional_execution='full R14 consumed, selected carry-mask question, focused correctness and scoped user continuous authorization',
        scale_selection='none; same .2 DAgger/.05 KL/7.5e-5 lr and saved r19 Adam',
        common_recipe_change_from_R10='same two categorical streams; no R12 greedy or R14 ch7 treatment',
        oracle='same public-state joint safe teacher, now given treatment learner mask; no new targets, safe geometry unchanged; glue never native control window',
        bank_limits='prior-used map806 P0 allfar; common production-aligned ch7 and carry-only actual-speed P0 collector mask for all tested models; versioned vs R14 Bank',
        successor='if credible whole-chain improvement with parent combat/safety retention, five new paired confirmation seeds; otherwise frozen zero-step audit of remaining natural item-chain coverage/label conflict and common-baseline retention, then choose a separately bounded question; no automatic unrelated GPU filler',
        stopping='six fixed32-update runs plus declared Bank/BankP; only technical invalidity/user stop interrupts; complete evidence selects next bounded question, never old queues',
        preregistration_R14_evidence_sha256=choice['R14_evidence_sha256'],
        dagger_coefficient_by_arm=weights,learning_rate_by_arm={a:7.5e-5 for a in arms},
        auxiliary_greedy={a:False for a in arms},release_qualification={a:False for a in arms},
        unix=time.time())
    p['memory']['teacher_loss_coefficient_by_arm']=weights
    p['memory']['source_metadata']='aligned to actual accepted learner arrivals; no inferred legacy provenance or policy input'
    p['evaluation']['natural_holdout_seed']=2026111600
    p['evaluation']['input_schema']='production JS unchanged, teacher off. Native Bank all models common production ch7+carry actual-speed P0 input mask; versioned vs R14. Tactical common original legacy input, separate.'
    p['evaluation']['selection']='all3 paired fixed final/tail/worst vs control and untrained parent, combat/safety and opportunity-normalized whole-chain; no winning-seed selection'
    B.mkdir(parents=True,exist_ok=True)
    atomic_json(B/'PROTOCOL.json',p)
    for folder in ['inputs','cpu_eval_inputs','inputs/sealed_sources','cache']:
        (B/folder).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(prior/'inputs/parent.pkl',B/'inputs/parent.pkl')
    assert sha(B/'inputs/parent.pkl')==p['parent_input_sha256']
    parent=R/p['parent']['checkpoint']
    assert sha(parent/'policy.pt')==p['parent']['policy_sha256']
    assert sha(parent/'state.pkl')==p['parent']['state_sha256']
    for name in ['spawns.json','regions.json']:
        shutil.copyfile(prior/'cpu_eval_inputs'/name,B/'cpu_eval_inputs'/name)
    ev=dict(read(prior/'CPU_EVAL_PROTOCOL.json'),holdout_seed_start=2026111600)
    assert sha(B/'cpu_eval_inputs/spawns.json')==ev['spawns_sha256']
    assert sha(R/'scripts/dagger_rl_metrics_v2.js')==ev['metric_sha256']
    atomic_json(B/'CPU_EVAL_PROTOCOL.json',ev)
    shutil.copyfile(prior/'audit_student_opportunities.pkl',B/'audit_student_opportunities.pkl')
    atomic_json(B/'AUDIT_REUSE_RECEIPT.json',dict(new_independent_examples=0,
        TRAIN_source_sha256=sha(B/'audit_student_opportunities.pkl'),
        mask_preparation_receipt_sha256=sha(prior/'NEXT_MASK_PREPARATION_ACCEPTANCE.json'),
        same_action_proof_sha256=sha(prior/'NEXT_MASK_SAME_ACTION_PARITY.json'),
        collector_input_proof_sha256=sha(prior/'NEXT_MASK_COLLECTOR_INPUT_PROOF.json'),
        scope='same26 previously used real TRAIN states;832 executed technical replays,0 adopted training steps'))
    sources=read(prior/'SOURCE_SEAL.json')['sources']
    for name,h in list(sources.items()):assert sha(R/name)==h,name
    for name in [f'scripts/dagger_rl_r15_{kind}.py' for kind in ['prepare','run','train','cpu','bank','report']]+[
        'scripts/dagger_rl_movement_probe.py','tests/test_dagger_rl_movement_probe.py','tests/test_dagger_rl_r15_setup.py']:
        sources[name]=sha(R/name)
    for name in sources:
        target=B/'inputs/sealed_sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(R/name,target)
    atomic_json(B/'SOURCE_SEAL.json',dict(sources=sources,protocol_sha256=sha(B/'PROTOCOL.json')))
    caches=[]
    for path in (prior/'cache').glob('*'):
        if not path.is_file():continue
        target=B/'cache'/path.name;shutil.copyfile(path,target)
        assert sha(target)==sha(path)
        caches.append(dict(name=path.name,sha256=sha(target)))
    atomic_json(B/'COMPILED_CACHE_REUSE.json',dict(compiler_files_only=True,no_params_Adam_state_or_RNG=True,files=caches))
    pause=sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json')
    assert pause=='358dc9c05479ea351ac8434a79ffc53333c896d0f04f84f7671d5c8333ffdbea'
    atomic_json(B/'SCOPED_GPU_AUTHORIZATION.json',dict(authorized_by='explicit_user_instruction',
        old_continuous_resume=False,old_global_pause_sha256=pause,bounded_protocol_sha256=sha(B/'PROTOCOL.json')))
    atomic_json(B/'SCIENTIFIC_SELECTION.json',dict(choice,protocol_sha256=sha(B/'PROTOCOL.json')))
    atomic_json(B/'NEXT_SCIENTIFIC_QUESTIONS.json',dict(
        confirmation=dict(condition='credible whole-chain/retention gain at all three fixed final/tail/worst endpoints',
            paired_seeds=list(range(2026102651,2026102656)),runs=10,updates=32,
            environment_steps=5242880,optimizer_steps=10240,holdout_seed_start=2026111700),
        failure=dict(question='zero-step remaining real natural item-chain opportunity/label conflict and common-baseline retention audit before selecting any next GPU recipe',
            new_environment_steps=0,optimizer_steps=0,no_automatic_multiplier_or_reward_sweep=True),old_continuous_resume=False))
    atomic_json(B/'MATERIALIZED.json',dict(passed=True,formal_not_started=True,old_continuous_resume=False,
        protocol_sha256=sha(B/'PROTOCOL.json'),unix=time.time()))
    print(p['id'],sha(B/'PROTOCOL.json'),flush=True)


if __name__=='__main__':prepare()
