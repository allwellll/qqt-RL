"""Freeze the kind-only goal eligibility experiment before its first update."""
import copy
import shutil
import time
from scripts.dagger_rl_r17_run import B, R, PRIOR, read
from scripts.population_run_state import atomic_json, sha


def register():
    assert not (B/'PROTOCOL.json').exists(), 'immutable namespace already registered'
    p = copy.deepcopy(read(PRIOR/'PROTOCOL.json'))
    p.update(id='DAGGER-RL-20261025-R17-GOAL-KIND', attempt=3,
        namespace=B.name, arms=['sequential_anchor'],
        single_variable='sequential route teacher additionally requires existing goal kind>0',
        goal_kind_guard=True, whole_legacy_active_guard=False,
        goal_guard_source_sha256=sha(R/'scripts/dagger_rl_route_goal_eligibility.py'),
        total_environment_steps=1572864, total_optimizer_steps=3072,
        matched_controls='reuse all three R16 sequential runs with identical paired seeds and full initial state; not independent new-seed confirmation',
        common='same R16 sequential16/17 schedule, parent/Adam, reward, input/mask, item teacher, '
               'opponent, reset mixture, optimizer, reservoir and safety; only remove route labels when kind0',
        hypothesis='Does removing demonstrably adopted goal0 fallback supervision improve sustained real '
                   'route and natural retention? Do not assume it dominates prior failures.',
        oracle='original temporal safe joint route teacher AND goal kind>0; preserve legitimate '
               'cargo escape even if legacy active0; item priority unchanged; stock only native death/sugar',
        scale_selection='none; no coefficient, switch, reward, sampling, input or network tuning',
        technical_smoke=dict(total_executed_environment_steps=3072, total_executed_optimizer_steps=12,
            adopted_formal_steps=0, checks='two loss branches, finite, real Actor/ability gradient and update, '
            'DAgger Critic/moments frozen and PPO Critic nonzero, fresh full-state next update exact'),
        reporting='matched R16 sequential controls, R16 PPO contextual control, untrained parent; '
                  'all fixed finals/tails/worst seeds, opportunity denominators, safe joint actions, '
                  'raw reward/GAE and fixed trace limits, source-clustered Bank/P',
        successor='If all three-seed natural whole-chain and battle retention are credible against both '
                  'untrained parent and controls, separately register five new paired seeds. Otherwise '
                  'use fixed earliest native action/cause/coverage diagnostics to select one bounded '
                  'question. No blind multiplier/reward changes, no unrelated GPU filler or old queues.',
        stopping='three new fixed32 runs and Bank/P; no capability/admission gates or peak selection; '
                 'only technical invalidity or user stop interrupts. Report evidence-dependent boundary immediately.',
        unix=time.time())
    p['dagger_coefficient_by_arm']={'sequential_anchor': .2}
    for field in ['learning_rate_by_arm', 'auxiliary_greedy', 'release_qualification']:
        p[field]={'sequential_anchor':p[field]['sequential_anchor']}
    p['evaluation']['natural_holdout_seed']=2026111800
    p['evaluation']['selection']='fixed same-seed R16 controls, prior-used same holdout disclosed; no tuning or new-independent confirmation'
    p['memory']['teacher_loss_coefficient_by_arm']=p['dagger_coefficient_by_arm']
    B.mkdir(parents=True, exist_ok=True)
    for folder in ['inputs/sealed_sources', 'cpu_eval_inputs', 'cache']:
        (B/folder).mkdir(parents=True, exist_ok=True)
    atomic_json(B/'PROTOCOL.json', p)
    shutil.copyfile(PRIOR/'inputs/parent.pkl', B/'inputs/parent.pkl')
    assert sha(B/'inputs/parent.pkl') == p['parent_input_sha256']
    for name in ['spawns.json', 'regions.json']:
        shutil.copyfile(PRIOR/'cpu_eval_inputs'/name,B/'cpu_eval_inputs'/name)
    shutil.copyfile(PRIOR/'CPU_EVAL_PROTOCOL.json',B/'CPU_EVAL_PROTOCOL.json')
    inherited = read(PRIOR/'SOURCE_SEAL.json')['sources']
    for name, h in inherited.items():
        assert sha(R/name)==h, name
    sources=dict(inherited)
    names=[f'scripts/dagger_rl_r17_{s}.py' for s in ['prepare','run','train','cpu','bank','report','diagnostics','monitor','handoff']]
    names+=['scripts/dagger_rl_route_goal_eligibility.py','scripts/dagger_rl_goal_guard_collector.py',
            'tests/test_dagger_rl_route_goal_eligibility.py','tests/test_dagger_rl_r17_authorization.py',
            'tests/test_dagger_rl_goal_guard_collector.py']
    for name in names: sources[name]=sha(R/name)
    for name in sources:
        dest=B/'inputs/sealed_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(R/name,dest)
    atomic_json(B/'SOURCE_SEAL.json',dict(sources=sources,protocol_sha256=sha(B/'PROTOCOL.json')))
    cache=[]
    for src in (PRIOR/'cache').glob('*'):
        if src.is_file():
            dst=B/'cache'/src.name;shutil.copyfile(src,dst)
            assert sha(src)==sha(dst);cache.append(dict(name=src.name,sha256=sha(dst)))
    atomic_json(B/'COMPILED_CACHE_REUSE.json',dict(compiler_files_only=True,no_state_or_Adam=True,files=cache))
    pause=sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json')
    assert pause=='358dc9c05479ea351ac8434a79ffc53333c896d0f04f84f7671d5c8333ffdbea'
    atomic_json(B/'SCOPED_GPU_AUTHORIZATION.json',dict(authorized_by='explicit_user_instruction',
        protocol_sha256=sha(B/'PROTOCOL.json'),old_global_pause_sha256=pause,
        namespace=B.name,old_continuous_resume=False))
    atomic_json(B/'PREREGISTRATION.json',dict(passed=True,protocol_sha256=sha(B/'PROTOCOL.json'),
        formal_updates=0,selection_pending_full_R16=True))
    print('R17 pre-registered',sha(B/'PROTOCOL.json'),flush=True)


def select():
    from scripts.dagger_rl_r17_handoff import consume_prior
    evidence=consume_prior()
    p=read(B/'PROTOCOL.json');controls=[]
    for seed in p['paired_seeds']:
        cp=PRIOR/'formal'/f'sequential_anchor_s{seed}'/'u032'
        m=read(cp/'COMMITTED.json')
        assert sha(cp/'state.pkl')==m['state_sha256'] and sha(cp/'policy.pt')==m['policy_sha256']
        controls.append(dict(seed=seed,checkpoint=str(cp.relative_to(R)),
            state_sha256=m['state_sha256'],policy_sha256=m['policy_sha256'],
            initial_state_sha256=sha(cp.parent/'u000/state.pkl')))
    atomic_json(B/'MATCHED_CONTROLS.json',dict(controls=controls,reused=True,independent_new_seed_confirmation=False))
    atomic_json(B/'SCIENTIFIC_SELECTION.json',dict(execute_goal_kind=True,
        protocol_sha256=sha(B/'PROTOCOL.json'),R16_comparison_sha256=sha(PRIOR/'FINAL_COMPARISON.json'),
        R16_supplement_sha256=sha(PRIOR/'SUPPLEMENTAL_EVIDENCE_CONSUMED.json'),
        matched_controls_sha256=sha(B/'MATCHED_CONTROLS.json'),prior_complete_evidence_sha256=sha(PRIOR/'FULL_HANDOFF_EVIDENCE.json'),
        reason='specific causal label-eligibility omission adopted twice in u9 replay; '
               'preserve real-cargo escape, no claim it explains all natural decline',evidence=evidence))


if __name__=='__main__':
    import sys
    select() if len(sys.argv)>1 and sys.argv[1]=='select' else register()
