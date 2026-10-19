"""Continuous fixed-budget relay with evidence-selected, predeclared mixtures.

This administrative owner can attach to an already running immutable worker.
It never modifies running training sources/plans or uses capability admission.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def choose_recipe(plan,summaries,proposal=None):
    combat=plan['combat_fraction'];bridge=plan['bridge_fraction']
    baseline=summaries[0] if summaries else {};latest=summaries[-1] if summaries else {}
    retention=[]
    for name,prior in baseline.get('hunter_js',{}).items():
        current=latest.get('hunter_js',{}).get(name,{})
        if current.get('safe_win',0)<prior.get('safe_win',0):retention.append('JS Hunter '+name)
    if latest.get('tactical_python',{}).get('safe_win',0)<baseline.get('tactical_python',{}).get('safe_win',0):retention.append('Python Tactical v2')
    if proposal is not None:
        assert proposal['recipe'] in ['native_one_step_certificate','continuous_route_metric','route_objective_consistency','prestock_preparation','phase_balanced_replay','navigation_replay','carrier_route_fields','return_dagger_weight']
        recipe=dict(combat_fraction=combat,bridge_fraction=bridge,natural_full_fraction=.35,
            route_mass=round(1-combat-bridge-.35,8),reason=proposal['reason'],
            changed_factor=proposal['changed_factor'],
            retention_diagnostics=retention,blocks_training=False,
            evidence_generations=[r.get('generation') for r in summaries],
            proposal=proposal)
        if proposal['recipe']=='native_one_step_certificate' or plan.get('oracle_certificate'):
            recipe['native_one_step_certificate']=True
        if proposal['recipe']=='continuous_route_metric' or plan.get('continuous_metric'):
            recipe['continuous_route_metric']=True
        if proposal['recipe']=='route_objective_consistency' or plan.get('route_objective_alignment'):
            recipe['route_objective_consistency']=True
        if proposal['recipe']=='prestock_preparation' or plan.get('prestock_preparation'):
            recipe['prestock_preparation']=True
        if proposal['recipe'] in ['phase_balanced_replay','navigation_replay'] or plan.get('phase_balanced_replay'):
            recipe['phase_balanced_replay']=True
        if proposal['recipe']=='navigation_replay' or plan.get('navigation_replay'):
            recipe['navigation_replay']=True
        if proposal['recipe']=='carrier_route_fields' or plan.get('carrier_route_fields'):
            recipe['carrier_route_fields']=True
        if proposal.get('normalized_advantage_diagnostic') or plan.get('normalized_advantage_diagnostic'):
            recipe['normalized_advantage_diagnostic']=True
        if proposal['recipe']=='return_dagger_weight' or plan.get('return_dagger_weight'):
            recipe['return_dagger_weight']=True
        if proposal.get('return_query_diversity') or plan.get('return_query_diversity'):
            recipe['return_query_diversity']=True
        if proposal.get('return_error_replay') or plan.get('return_error_replay'):
            recipe['return_error_replay']=True
        if proposal.get('replay_exposure') or plan.get('replay_exposure'):
            recipe['replay_exposure']=True
        if proposal.get('replay_batch') or plan.get('replay_batch'):
            recipe['replay_batch']=True
        if proposal.get('native_return_retention') or plan.get('native_return_retention'):
            recipe['native_return_retention']=True
        if proposal.get('replay_horizon') or plan.get('replay_horizon'):
            recipe['replay_horizon']=True
        return recipe
    # The changed factor is the scene mixture. Rewards/DAgger/query budgets,
    # networks, PPO and native dynamics remain immutable.
    if 1-combat-bridge-.35<.05:
        combat,bridge=.35,.20;reason='Zero independent native route initialization support; reserve .10 mass while retaining .35 natural full and combat.'
    elif retention:
        combat,bridge=.40,.15;reason='Observed within-protocol combat retention loss; allocate combat mass while preserving .10 route and .35 full.'
    else:
        combat,bridge=.30,.20;reason='Continue the weakest route funnel with .15 route initialization mass, retaining .35 full and .30 combat; no stop/promotion decision.'
    recipe=dict(combat_fraction=combat,bridge_fraction=bridge,natural_full_fraction=.35,
        route_mass=round(1-combat-bridge-.35,8),reason=reason,
        retention_diagnostics=retention,blocks_training=False,
        changed_factor='scene mixture only',evidence_generations=[r.get('generation') for r in summaries])
    if plan.get('oracle_certificate'):recipe['native_one_step_certificate']=True
    if plan.get('continuous_metric'):recipe['continuous_route_metric']=True
    if plan.get('route_objective_alignment'):recipe['route_objective_consistency']=True
    if plan.get('prestock_preparation'):recipe['prestock_preparation']=True
    if plan.get('phase_balanced_replay'):recipe['phase_balanced_replay']=True
    if plan.get('navigation_replay'):recipe['navigation_replay']=True
    if plan.get('carrier_route_fields'):recipe['carrier_route_fields']=True
    if plan.get('normalized_advantage_diagnostic'):recipe['normalized_advantage_diagnostic']=True
    if plan.get('return_dagger_weight'):recipe['return_dagger_weight']=True
    if plan.get('return_query_diversity'):recipe['return_query_diversity']=True
    if plan.get('return_error_replay'):recipe['return_error_replay']=True
    if plan.get('replay_exposure'):recipe['replay_exposure']=True
    if plan.get('replay_batch'):recipe['replay_batch']=True
    if plan.get('native_return_retention'):recipe['native_return_retention']=True
    if plan.get('replay_horizon'):recipe['replay_horizon']=True
    return recipe


def alive(pid):
    try:
        state=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
        return state!='Z'
    except (OSError,IndexError):return False


def predeclare_continuity(base,plan,declaration):
    """Concrete same-recipe fallback, replaceable by a tested next proposal.

    Written before the run's technical smoke. A prepared specific proposal has
    priority; scores and CPU evaluation completion never control this fallback.
    """
    import hashlib
    from scripts.population_run_state import atomic_json
    base=Path(base);proposal=base/'NEXT_RECIPE_PROPOSAL.json'
    if proposal.exists():return json.loads(proposal.read_text())
    number=declaration['round_number'];following=number+1
    identity=f'JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r{following:02d}'
    if plan.get('clean_causal_init'):identity+='-clean-causal-init'
    recipe=('return_dagger_weight' if plan.get('return_dagger_weight') else
        'carrier_route_fields' if plan.get('carrier_route_fields') else
        'navigation_replay' if plan.get('navigation_replay') else
        'phase_balanced_replay' if plan.get('phase_balanced_replay') else
        'prestock_preparation' if plan.get('prestock_preparation') else
        'route_objective_consistency' if plan.get('route_objective_alignment') else
        'continuous_route_metric' if plan.get('continuous_metric') else 'native_one_step_certificate')
    current=declaration.get('namespace',f'runs/je10_bun_20261018/control_route_v38_r{number:02d}')
    result=dict(recipe=recipe,experiment_id=identity,successor=declaration['terminal_relay']['successor'],
        parent=current+'/formal/bun392_bridge_pfsp8/gen000128',
        seeds={role:int.from_bytes(hashlib.sha256((identity+':'+role).encode()).digest()[:4],'big') for role in ['train','technical']},
        reason='Predeclared same-recipe fixed continuation; a concrete evidence-backed tested next factor may replace it before terminal. No final evaluation wait.',
        changed_factor='No new numerical factor; fixed-budget same-recipe continuation',
        scene_mixture=dict(combat=plan['combat_fraction'],bridge=plan['bridge_fraction'],natural_full=.35,
            other_native_routes=round(1-plan['combat_fraction']-plan['bridge_fraction']-.35,8)),
        fixed_budget=dict(devices=8,generations=128,fresh_ticks=268435456),evaluation=declaration['evaluation'],
        initialization_scope='Final raw Actor/Critic and full pool weights; fresh Adam, EMA-to-raw, RNG/env, FIFO and counters',
        async_ownership=dict(global_unique_cpu_js_owner=True,recovery_every=4,
            consume_marker='authority_je10/gXXXXXX/COMPLETE.json',cpu_gpu_independent=True),
        predeclared_fallback=True,capability_gate=False,evaluation_blocks_training=False,
        install_only_after_fixed_terminal=True)
    if plan.get('return_query_diversity'):result['return_query_diversity']=True
    if plan.get('return_error_replay'):result['return_error_replay']=True
    if plan.get('replay_exposure'):result['replay_exposure']=True
    if plan.get('replay_batch'):result['replay_batch']=True
    if plan.get('native_return_retention'):result['native_return_retention']=True
    if plan.get('replay_horizon'):result['replay_horizon']=True
    atomic_json(proposal,result);atomic_json(base/'SUCCESSOR_PREDECLARATION.json',result)
    return result


def gpu_release_receipt(inventory,clients):
    devices={int(line.split(',')[0]) for line in inventory.splitlines() if line.strip()}
    busy=[line for line in clients.splitlines() if line.strip()]
    return dict(released=devices==set(range(8)) and not busy,devices=sorted(devices),
        compute_clients=len(busy),capability_gate=False)


def formal_update_receipt(row,checkpoint):
    assert row.get('generation')==checkpoint.get('generation')==1
    assert row.get('finite') and row.get('nonzero_update')
    assert checkpoint.get('passed') and checkpoint.get('replicas_exact') and checkpoint.get('devices')==8
    for key in ['actor_update_l2','critic_update_l2']:
        values=row.get(key,[])
        assert len(values)==8 and all(math.isfinite(x) and x>0 for x in values)
    assert row.get('fresh_ticks')==checkpoint.get('fresh_ticks')==2097152
    return dict(formal_nonzero_verified=True,generation=1,devices=8,finite=True,
        actor_update_l2=row['actor_update_l2'],critic_update_l2=row['critic_update_l2'],
        replicas_exact=True,fresh_ticks=2097152,capability_gate=False)


def await_formal_update(c,base,worker):
    from scripts.population_run_state import atomic_json,sha
    checkpoint=base/'formal'/c.ARM/'gen000001'/'COMMITTED.json'
    while not checkpoint.exists():
        if worker.poll() is not None:raise RuntimeError('formal worker exited before committed nonzero update')
        time.sleep(5)
    with (checkpoint.parents[1]/'events.jsonl').open() as stream:row=json.loads(next(stream))
    marker=c.read(checkpoint);receipt=formal_update_receipt(row,marker)
    assert sha(checkpoint.parent/'policy.pt')==marker['policy_sha256']
    receipt.update(checkpoint=str(checkpoint.parent.relative_to(ROOT)),policy_sha256=marker['policy_sha256'])
    atomic_json(base/'FORMAL_NONZERO_CONFIRMED.json',receipt)
    return receipt


def await_gpu_release(c,base):
    from scripts.population_run_state import atomic_json
    warned=False
    while True:
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        clients=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
        receipt=gpu_release_receipt(inventory,clients)
        atomic_json(base/'GPU_RELEASE_RECEIPT.json',receipt)
        if receipt['released']:return receipt
        if not warned:
            c.notify('JE10固定run已终态，正在核验8卡compute进程释放；不终止其他任务进程。')
            warned=True
        time.sleep(5)


def install_successor_host_trace_fix(parent):
    """Apply only after predecessor terminal; no healthy worker source mutation."""
    terminal=parent.parent/'TRAINER_EXIT.json'
    # parent is .../formal/arm/gen000128, so TRAINER_EXIT lives at arm/.
    result=json.loads(terminal.read_text())
    assert result['generation']==128 and not result['atomic_stop']
    worker=ROOT/'scripts/je10_worker.py';source=worker.read_text()
    old="state_hash=hashlib.sha256(np.asarray(batch[0])[dev,tick,env_index].tobytes()+np.asarray(batch[1])[dev,tick,env_index].tobytes()).hexdigest()"
    new="state_hash=observation_state_hash(batch[0],batch[1],(int(dev),int(tick),int(env_index)))"
    if old in source:
        assert source.count(old)==1
        source=source.replace('            failures=[]\n','            from scripts.learner_trace_sampling import observation_state_hash\n            failures=[]\n').replace(old,new)
    else:assert new in source,'unexpected host trace implementation'
    if "row['dagger']['verifier_audit']" not in source:
        anchor="            row['signal_decomposition']=dict(control=float(trace[...,1].sum()),route=float(trace[...,2].sum()),"
        assert source.count(anchor)==1
        source=source.replace(anchor,"            from scripts.learner_trace_sampling import verifier_audit_counts\n            row['dagger']['verifier_audit']=verifier_audit_counts(ctx)\n"+anchor)
    if "row['executed_opponent_exposure']" not in source:
        anchor="            row['dagger']['verifier_audit']=verifier_audit_counts(ctx)"
        assert source.count(anchor)==1
        source=source.replace(anchor,anchor+"\n            from scripts.learner_trace_sampling import executed_opponent_exposure\n            row['executed_opponent_exposure']=executed_opponent_exposure(ctx)")
    worker.write_text(source)


def install_successor_exposure_trace(parent):
    result=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert result['generation']==128 and not result['atomic_stop']
    path=ROOT/'scripts/je09_stage_h5c_ops.py';source=path.read_text()
    action='        other=curriculum.assisted_actions(other,mm[oppidx],assist,kwalk)'
    captured=action+'\n        executed_source=jnp.where(assist,6,jnp.where(role>0,3+role,kinds))'
    anchor='            context=jnp.concatenate([context,trace],-1)'
    replacement="""            # Diagnostic provenance only: last override wins, no action/RNG change.
            provenance=jnp.stack([executed_source,jnp.full_like(seat,member),states.alive[rows,seat]],-1).astype(jnp.float32)
            context=jnp.concatenate([context,trace,provenance],-1)"""
    if anchor in source:
        assert source.count(anchor)==source.count(action)==1
        path.write_text(source.replace(anchor,replacement).replace(action,captured))
    else:assert replacement in source,'unexpected execution provenance implementation'


def supervise(args):
    # Factory remains sealed; administrative policy is a separate source.
    from scripts import je10_campaign as c
    from scripts.population_run_state import atomic_json,sha,lock
    base=c.BASE
    if args.command=='start':
        decision=json.loads(Path(args.decision).read_text())
        install_successor_host_trace_fix(ROOT/args.parent)
        install_successor_exposure_trace(ROOT/args.parent)
        certificate=None;continuous=None;objective=None;prestock=None;phase_replay=None;navigation=None;carrier_fields=None;normalized=None;return_weight=None;query_diversity=None;error_replay=None;replay_exposure=None;replay_batch=None;native_retention=None;horizon=None
        for name,expected in decision.get('proposal',{}).get('sources',{}).items():
            assert sha(ROOT/name)==expected,'prepared successor recipe source changed'
        if decision.get('native_one_step_certificate'):
            from scripts.native_certificate_recipe import install
            certificate=install(ROOT,ROOT/args.parent)
        assert not decision.get('carrier_phase_hysteresis'),'uninstalled recipe excluded from release'
        if decision.get('continuous_route_metric'):
            from scripts.continuous_metric_recipe import install as install_continuous
            continuous=install_continuous(ROOT,ROOT/args.parent)
        if decision.get('route_objective_consistency'):
            from scripts.route_objective_recipe import install as install_objective
            objective=install_objective(ROOT,ROOT/args.parent)
        if decision.get('prestock_preparation'):
            from scripts.prestock_recipe import install as install_prestock
            prestock=install_prestock(ROOT,ROOT/args.parent)
        if decision.get('phase_balanced_replay'):
            from scripts.phase_balanced_replay_recipe import install as install_phase_replay
            phase_replay=install_phase_replay(ROOT,ROOT/args.parent)
        if decision.get('navigation_replay'):
            from scripts.navigation_replay_recipe import install as install_navigation
            navigation=install_navigation(ROOT,ROOT/args.parent)
        if decision.get('carrier_route_fields'):
            from scripts.carrier_route_recipe import install as install_carrier_fields
            carrier_fields=install_carrier_fields(ROOT,ROOT/args.parent)
        if decision.get('normalized_advantage_diagnostic'):
            from scripts.ppo_advantage_diagnostic_recipe import install as install_normalized
            normalized=install_normalized(ROOT,ROOT/args.parent)
        if decision.get('return_dagger_weight'):
            from scripts.return_dagger_recipe import install as install_return_weight
            return_weight=install_return_weight(ROOT,ROOT/args.parent)
        # A previously loaded owner preserves nested proposals across the relay.
        if decision.get('return_query_diversity') or decision.get('proposal',{}).get('return_query_diversity'):
            from scripts.return_query_recipe import install as install_return_queries
            query_diversity=install_return_queries(ROOT,ROOT/args.parent)
        if decision.get('return_error_replay') or decision.get('proposal',{}).get('return_error_replay'):
            from scripts.return_error_recipe import install as install_return_error
            error_replay=install_return_error(ROOT,ROOT/args.parent)
        if decision.get('replay_exposure') or decision.get('proposal',{}).get('replay_exposure'):
            from scripts.replay_exposure_recipe import install as install_replay_exposure
            replay_exposure=install_replay_exposure(ROOT,ROOT/args.parent)
        if decision.get('replay_batch') or decision.get('proposal',{}).get('replay_batch'):
            from scripts.replay_batch_recipe import install as install_replay_batch
            replay_batch=install_replay_batch(ROOT,ROOT/args.parent)
            if error_replay:
                error_replay['current_history_and_phase_slot_counts_unchanged']=False
                error_replay['current_history_and_phase_proportions_unchanged']=True
                error_replay['slot_count_override']='See replay_batch: each half 32 to 64'
        if decision.get('native_return_retention') or decision.get('proposal',{}).get('native_return_retention'):
            from scripts.native_return_retention_recipe import install as install_native_retention
            native_retention=install_native_retention(ROOT,ROOT/args.parent)
        if decision.get('replay_horizon') or decision.get('proposal',{}).get('replay_horizon'):
            from scripts.replay_horizon_recipe import install as install_replay_horizon
            horizon=install_replay_horizon(ROOT,ROOT/args.parent)
        c.prepare(args.parent,args.round)
        plan=c.read(base/'plan.json');plan.update(combat_fraction=decision['combat_fraction'],bridge_fraction=decision['bridge_fraction'])
        plan['campaign']['change']+='; successor: '+decision['changed_factor']
        if certificate:plan['oracle_certificate']=certificate
        if continuous:plan['continuous_metric']=continuous
        if objective:plan['route_objective_alignment']=objective
        if prestock:
            plan['prestock_preparation']=prestock
            if not plan.get('clean_causal_init'):plan['state_schema']='je10-control-route-v38-prestock-v1'
            plan['dagger']['oracle']='Shared pre-stock corridor target, native-safe joint movement/ability set; ordinary movement labels preserve all legal abilities'
        if phase_replay:plan['phase_balanced_replay']=phase_replay
        if navigation:plan['navigation_replay']=navigation
        if carrier_fields:plan['carrier_route_fields']=carrier_fields
        if normalized:plan['normalized_advantage_diagnostic']=normalized
        if return_weight:plan['return_dagger_weight']=return_weight
        if query_diversity:plan['return_query_diversity']=query_diversity
        if error_replay:plan['return_error_replay']=error_replay
        if replay_exposure:plan['replay_exposure']=replay_exposure
        if replay_batch:plan['replay_batch']=replay_batch
        if native_retention:plan['native_return_retention']=native_retention
        if horizon:
            plan['replay_horizon']=horizon
            plan['dagger']['capacity_per_device']=horizon['capacity_per_device']
        if replay_batch:plan['dagger']['replay_per_microbatch']=128
        atomic_json(base/'plan.json',plan)
        declaration=c.read(base/'ROUND_DECLARATION.json');declaration.update(successor_recipe=decision,
            change=declaration['change']+'; successor: '+decision['changed_factor'],
            executed_scene_mixture=dict(combat=decision['combat_fraction'],bridge=decision['bridge_fraction'],natural_full=.35,other_native_routes=decision['route_mass']))
        declaration['actual_run_initialization']=dict(raw_parent='final Actor/Critic and full pool',
            optimizer='fresh Adam',environment_rng='fresh identity-derived seed',dagger_buffer='fresh FIFO',
            retention_reference='new parent Actor; frozen g0/stage1 unchanged',
            causal_scope='Sequential followup without same-parent control; recipe factor does not imply causal attribution.')
        # Historical template metadata is not this run's admission protocol.
        for stale in ['smoke_attempts','parent_preservation','parent_status']:
            declaration.pop(stale,None)
        declaration['round']=args.round
        declaration['seeds']={role:declaration['seeds'][role] for role in ['train','technical']}
        declaration['seed_derivation']={role:dict(identity=declaration['experiment_id']+':'+role,
            algorithm='SHA256 first4 bytes, big endian') for role in ['train','technical']}
        declaration['smoke_generations']=1
        declaration['smoke_policy']='Technical only: imports/focused TDD, finite nonzero update, NaN/OOM, save and fresh CPU restore. No ability/admission/promotion threshold.'
        declaration['curriculum']=dict(declaration['executed_scene_mixture'],
            scope='reset sampling weights; actual valid tick occupancy logged separately')
        declaration['full_scene_coverage']=dict(scene=8,fraction=.35,
            source='complete native full games; learner drives both seats')
        declaration['immutable']['retention_reference']='new run parent Actor; frozen g0/stage1 unchanged'
        declaration['dagger']['retention']='Executed reset mixture above; unchanged retention coefficients; reference is new parent Actor.'
        declaration['parent_selection']['reason']=('User-authorized rollback to r10 g128; r11 excluded for synthetic learner custody; preserve raw Actor/Critic and full pool, rebuild all learner state.' if plan.get('clean_causal_init') and args.round==12 else 'Predeclared sequential continuation from preceding final raw Actor/Critic and complete pool; published origin and full curves recorded in ledger.')
        if certificate:declaration['oracle_certificate']=certificate
        if continuous:declaration['continuous_metric']=continuous
        if objective:declaration['route_objective_alignment']=objective
        if prestock:
            declaration['prestock_preparation']=prestock
            declaration['dagger']['oracle']=plan['dagger']['oracle']
            declaration['dagger']['query_policy']='Existing bounded fragment slots; joint first deviation, one navigation/opening query each per preparation revision within a rollout episode chunk; ordinary fragment quotas unchanged'
            declaration['state_schema']=plan['state_schema']
        if phase_replay:declaration['phase_balanced_replay']=phase_replay
        if navigation:declaration['navigation_replay']=navigation
        if carrier_fields:declaration['carrier_route_fields']=carrier_fields
        if normalized:declaration['normalized_advantage_diagnostic']=normalized
        if return_weight:declaration['return_dagger_weight']=return_weight
        if query_diversity:
            declaration['return_query_diversity']=query_diversity
            declaration['dagger']['query_policy']='All baseline queries plus at most one later native-verified ordinary return mismatch at a different observed cell, gap>=8 ticks per rollout episode chunk; terminal resets, other quotas unchanged'
        if error_replay:declaration['return_error_replay']=error_replay
        if replay_exposure:declaration['replay_exposure']=replay_exposure
        if replay_batch:declaration['replay_batch']=replay_batch
        if native_retention:
            declaration['native_return_retention']=native_retention
            declaration['dagger']['retention']='Old teacher movement KL omitted only on certified safe native enemy-origin return rows; bomb/danger/combat and parent retention unchanged.'
        if horizon:
            declaration['replay_horizon']=horizon
            declaration['dagger']['capacity_per_device']=horizon['capacity_per_device']
        if replay_batch:declaration['dagger']['replay_per_microbatch']=128
        atomic_json(base/'ROUND_DECLARATION.json',declaration)
        seal=c.read(base/'SOURCE_SEAL.json');seal['plan_sha256']=sha(base/'plan.json')
        # Administrative relay and CPU evidence writers are independently sealed;
        # neither changes the worker's numerical recipe during a fixed run.
        administrative={name:seal['sources'].pop(name) for name in list(seal['sources'])
            if name in ['scripts/je10_supervise.py','scripts/je10_diagnostics.py',
                        'tests/test_je10_successor_recipe.py','tests/test_je10_relay_contract.py']}
        atomic_json(base/'ADMINISTRATIVE_SEAL.json',dict(sources=administrative,numerical_recipe=False))
        helpers=['scripts/learner_trace_sampling.py']
        if certificate:helpers+=['scripts/native_route_certificate.py']
        if continuous:helpers+=['scripts/continuous_route_metric.py','scripts/continuous_metric_recipe.py']
        if objective:helpers+=['scripts/route_objective_consistency.py','scripts/route_objective_recipe.py']
        if prestock:helpers+=['scripts/prestock_recipe.py','scripts/prestock_training_state.py',
            'scripts/prestock_route_preparation.py','scripts/prestock_verified_labels.py',
            'scripts/prestock_native_teacher.py','scripts/prestock_joint_payload.py',
            'scripts/prestock_joint_replay.py','scripts/prestock_joint_oracle.py',
            'scripts/prestock_fragment_queries.py','scripts/frontier_native_witness.py','scripts/route_frontier.py',
            'scripts/terminal_checkpoint_restore.py']
        if phase_replay:helpers+=['scripts/phase_balanced_joint_replay.py','scripts/phase_balanced_replay_recipe.py']
        if navigation:helpers+=['scripts/navigation_replay_recipe.py']
        if carrier_fields:helpers+=['scripts/carrier_route_fields.py','scripts/carrier_route_recipe.py']
        if normalized:helpers+=['scripts/ppo_advantage_diagnostics.py','scripts/ppo_advantage_diagnostic_recipe.py']
        if return_weight:helpers+=['scripts/return_dagger_weight.py','scripts/return_dagger_recipe.py']
        if query_diversity:helpers+=['scripts/return_query_diversity.py','scripts/return_query_recipe.py',
            'tests/test_return_query_diversity.py','tests/test_return_query_recipe.py',
            'tests/test_return_query_collector_integration.py','tests/test_return_query_audit.py']
        if error_replay:helpers+=['scripts/return_error_replay.py','scripts/return_error_recipe.py',
            'tests/test_return_error_replay.py','tests/test_return_error_recipe.py',
            'tests/test_return_error_collector_integration.py']
        if replay_exposure:helpers+=['scripts/replay_exposure.py','scripts/replay_exposure_recipe.py',
            'tests/test_replay_exposure.py','tests/test_replay_exposure_recipe.py']
        if replay_batch:helpers+=['scripts/replay_batch_recipe.py','tests/test_replay_batch_recipe.py',
            'tests/test_replay_batch_collector_integration.py']
        if native_retention:helpers+=['scripts/native_return_retention.py','scripts/native_return_retention_recipe.py',
            'tests/test_native_return_retention.py','tests/test_native_return_retention_recipe.py',
            'tests/test_native_return_retention_collector_integration.py']
        if horizon:helpers+=['scripts/replay_horizon_recipe.py','tests/test_replay_horizon_recipe.py',
            'tests/test_replay_horizon_collector_integration.py']
        for helper in helpers:
            seal['sources'][helper]=sha(ROOT/helper)
            target_source=base/'inputs/sealed_sources'/helper
            target_source.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/helper,target_source)
        atomic_json(base/'SOURCE_SEAL.json',seal)
        predecessor=ROOT/args.parent
        evaluation=c.read(predecessor.parents[2]/'EVALUATION_SEAL.json')
        evaluation.update(sources={name:sha(ROOT/name) for name in evaluation['sources']},
            training_source_changed=False,protocol_sha256=sha(base/'ROUND_DECLARATION.json'),
            reason='Independent CPU/JS authority provenance; fixed protocol inherited, never GPU admission')
        if prestock:evaluation['sources']['scripts/audit_joint_dagger.py']=sha(ROOT/'scripts/audit_joint_dagger.py')
        atomic_json(base/'EVALUATION_SEAL.json',evaluation)
        predeclare_continuity(base,plan,declaration)
        cache=predecessor.parents[2]/'cache/gpu';target=base/'cache/gpu'
        if not plan.get('clean_causal_init') and cache.exists() and not target.exists():target.symlink_to(os.path.relpath(cache,target.parent))
    owner=lock(base/'campaign.lock');c.check_seal()
    atomic_json(base/'CONTINUOUS_OWNER.json',dict(unique=True,kind='adaptive administrative relay',pid=os.getpid(),worker_recipe_immutable=True,capability_gate=False))
    if args.command in ['start','recover-technical']:
        if args.command=='recover-technical':
            decision=c.read(base/'ROUND_DECLARATION.json')['successor_recipe']
        smoke=base/'technical_smoke01'
        if args.command=='recover-technical':
            assert not (base/'TRAIN_LAUNCH.json').exists(),'formal recovery requires checkpoint resume'
            attempt=2
            while (base/f'technical_smoke{attempt:02d}').exists():attempt+=1
            smoke=base/f'technical_smoke{attempt:02d}'
        (smoke/'inputs').mkdir(parents=True)
        for name in ['initial.pkl','initial_pool.pkl','frozen.pkl','regions.pkl','retention_parent.pkl']:shutil.copyfile(base/'inputs'/name,smoke/'inputs'/name)
        shutil.copyfile(base/'regions.json',smoke/'regions.json')
        plan=c.read(base/'plan.json');plan.update(generations=1,checkpoints=[0,1],seed=c.read(base/'ROUND_DECLARATION.json')['seeds']['technical'])
        atomic_json(smoke/'plan.json',plan)
        proc=c.launch('worker',smoke,log_name='smoke.log')
        c.notify('JE10预声明后继已占用8卡执行最小技术smoke；'+decision['changed_factor']+'，无能力门禁。')
        if proc.wait()!=0:raise RuntimeError('successor numerical technical smoke failed')
        row=c.read(smoke/'formal'/c.ARM/'STATE.json')
        assert row['finite'] and row['nonzero_update']
        assert c.launch('restore',smoke,smoke/'formal'/c.ARM/'gen000001',cpu=True,log_name='smoke_restore.log').wait()==0
        atomic_json(base/'TECHNICAL_SMOKE_COMPLETE.json',dict(passed=True,finite=True,nonzero_update=True,devices=8,save_restore=True,capability_gate=False))
        train=c.launch('worker',base,log_name='formal.log');pid=train.pid
        atomic_json(base/'TRAIN_LAUNCH.json',dict(started=True,devices=list(range(8)),fixed_generations=128,fresh_ticks=268435456,capability_gate=False,namespace=c.NAMESPACE))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JE10_NAMESPACE=c.NAMESPACE)
        subprocess.Popen(['taskset','-c','80-95',sys.executable,'scripts/je10_evaluate.py'],cwd=ROOT,env=env,stdout=(base/'async_eval.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
        receipt=await_formal_update(c,base,train)
        c.notify(f'JE10后继正式非零更新已确认：{c.NAMESPACE}，g1/2,097,152 ticks，8副本一致、finite、Actor/Critic均真实非零；固定128代继续，CPU/JS独立跟随。')
    else:
        marker=base/'formal'/c.ARM/'WORKER.json'
        assert (base/'TRAIN_LAUNCH.json').exists() and marker.exists()
        pid=c.read(marker)['pid'];assert alive(pid),'cannot attach dead worker'
    while (train.poll() is None if args.command=='start' else alive(pid)):time.sleep(10)
    exit_file=base/'formal'/c.ARM/'TRAINER_EXIT.json'
    if not exit_file.exists():raise RuntimeError('worker exited without atomic terminal checkpoint')
    result=c.read(exit_file)
    assert result['generation']==128 and not result['atomic_stop'],'technical/user stop; never restart silently'
    release=await_gpu_release(c,base)
    summaries=[c.read(p) for p in sorted((base/'authority_je10').glob('g*/SUMMARY.json')) if (p.parent/'COMPLETE.json').exists()]
    proposal_path=base/'NEXT_RECIPE_PROPOSAL.json'
    proposal=c.read(proposal_path) if proposal_path.exists() else None
    recipe=choose_recipe(c.read(base/'plan.json'),summaries,proposal)
    atomic_json(base/'SUCCESSOR_RECIPE.json',recipe)
    declaration=c.read(base/'ROUND_DECLARATION.json');successor=declaration['terminal_relay']['successor']
    # Start the predeclared successor BEFORE terminal report/final eval waits.
    env=dict(os.environ,JE10_NAMESPACE=successor,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu');env.pop('JE10_REUSE_TECHNICAL_CHECKPOINT',None)
    proc=subprocess.Popen([sys.executable,'scripts/je10_supervise.py','start','--parent',f'{c.NAMESPACE}/formal/{c.ARM}/gen000128',
        '--round',str(declaration['round_number']+1),'--decision',str((base/'SUCCESSOR_RECIPE.json').relative_to(ROOT))],cwd=ROOT,env=env,
        stdout=(base/'successor.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
    terminal=dict(completed=True,**result,successor=successor,successor_owner_started=proc.poll() is None,successor_recipe=recipe,gpu_release_verified=release['released'],successor_formal_nonzero_verified=False)
    atomic_json(base/'RUN_TERMINAL.json',terminal)
    c.notify(f'JE10固定run终态：{c.NAMESPACE}已完成128代/268,435,456 ticks；8卡释放已核验，已立即交给预声明后继{successor}执行最小技术smoke，正式非零确认随后通知。')
    # Successor installation can legitimately change current source files.
    # Validate the predecessor's immutable snapshot and checkpoint on CPU.
    restore_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu')
    restore=subprocess.Popen(['taskset','-c','96-111',sys.executable,'scripts/terminal_checkpoint_restore.py',
        str(base.relative_to(ROOT)),f'{c.NAMESPACE}/formal/{c.ARM}/gen000128'],cwd=ROOT,env=restore_env,
        stdout=(base/'final_snapshot_restore.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
    confirmed=ROOT/successor/'FORMAL_NONZERO_CONFIRMED.json'
    while not confirmed.exists():
        if proc.poll() is not None:raise RuntimeError('successor owner failed before formal nonzero confirmation')
        time.sleep(5)
    terminal['successor_formal_nonzero_verified']=True
    terminal['successor_first_update']=c.read(confirmed)
    atomic_json(base/'RUN_TERMINAL.json',terminal)
    summary_line=f"\n固定run `{c.NAMESPACE}` 完成128代/268,435,456 ticks；compute={result['compute_seconds']:.3f}s、wall={result['wall_seconds']:.3f}s。已先启动预声明后继 `{successor}`；本次后继配方选择依据 `{recipe['reason']}`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。\n"
    from datetime import datetime
    from zoneinfo import ZoneInfo
    report_date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    for target in ['docs/training_experiment_ledger.md',f'reports/je10_training_{report_date}.md']:
        path=ROOT/target
        with path.open('a') as stream:stream.write(summary_line)
    assert restore.wait()==0,'final checkpoint restore failed; successor continues'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['attach','start','recover-technical']);parser.add_argument('--parent');parser.add_argument('--round',type=int,default=1);parser.add_argument('--decision');args=parser.parse_args()
    try:supervise(args)
    except Exception as error:
        from scripts import je10_campaign as c
        from scripts.population_run_state import atomic_json
        if c.BASE.exists():atomic_json(c.BASE/'CONTINUOUS_OWNER_FAILURE.json',dict(error=str(error),capability_gate=False))
        c.notify(f'JE10连续owner技术阻塞：{c.NAMESPACE}；{type(error).__name__}: {error}');raise


if __name__=='__main__':main()
