"""Read-only R16 consumer and unique bounded handoff, never old queues."""
import os
import sys
import time
import subprocess
from scripts.dagger_rl_r17_run import B, R, PRIOR, read, notify, UserStopped
from scripts.population_run_state import atomic_json, sha, lock

CPU_ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')


def stopped():
    return (B/'USER_STOP.json').exists() or (PRIOR/'USER_STOP.json').exists()


def consume_prior():
    required=['GPU_COMPLETE.json','OWNER_EXIT.json','COMPLETE_EVIDENCE_CONSUMED.json',
              'SUPPLEMENTAL_EVIDENCE_CONSUMED.json','FORMAL_TERMINAL_ACCEPTANCE.json',
              'FIXED_DEV_DIRECTION_AUDIT_v2.json','TRAIN_PROJECTION_AUDIT.json',
              'GOAL0_REPLAY_WITNESS_v2.json','REAL_GOAL_GUARD_CHECK.json',
              'GOAL_KIND_GUARD_PREPARATION.json','BANK_SOURCE_CLUSTER_AUDIT.json']
    for name in required:
        d=read(PRIOR/name)
        if name!='OWNER_EXIT.json': assert d['passed'],name
    assert read(PRIOR/'ATOMIC_ACCEPTANCE.json')['declared_checkpoints']==63
    native=read(PRIOR/'FIXED_DEV_DIRECTION_AUDIT_v2.json');assert native['completed']==19
    sampled=read(PRIOR/'TRAIN_PROJECTION_AUDIT.json')['rows']
    eligible=[x for x in sampled if x['route_adoptable']]
    zero=[x for x in eligible if x['kind']==0]
    guard=read(PRIOR/'REAL_GOAL_GUARD_CHECK.json')
    assert guard['kind_only'] and guard['valid_real_cargo_preserved_exact'] and guard['invalid_goal_removed']
    witness=read(PRIOR/'GOAL0_REPLAY_WITNESS_v2.json')
    assert witness['full_joint_label_match'] and sum(x['actual_replay_draws'] for x in witness['history_input_matches'])>0
    comp=read(PRIOR/'FINAL_COMPARISON.json');supp=read(PRIOR/'SUPPLEMENTAL_EVIDENCE_CONSUMED.json')
    assert supp['comparison_sha256']==sha(PRIOR/'FINAL_COMPARISON.json')
    summary={arm:dict(holdout_delivery=[x['holdout']['episode_counts']['delivery'] for x in rows],
        holdout_death=[x['holdout']['episode_counts']['death'] for x in rows],
        dev_tail=[x['tail_dev_delivery'] for x in rows],
        worst_holdout_delivery=min(x['holdout']['episode_counts']['delivery'] for x in rows),
        Hunter=[x['Hunter'] for x in rows],Tactical=[x['Tactical'] for x in rows])
        for arm,rows in comp['arms'].items()}
    audit=dict(passed=True,inputs={n:sha(PRIOR/n) for n in required+['FINAL_COMPARISON.json']},
        comparison=summary,parent=comp['parent'],sampled_route_rows=len(eligible),sampled_goal0_rows=len(zero),
        sampled_goal0_unique_states=len({x['state_hash'] for x in zero}),
        actual_goal0_replay_draws=sum(x['actual_replay_draws'] for x in witness['history_input_matches']),
        geometric_flagged=len([x for x in native['checkpoints'].values() if x.get('first')]),
        geometric_flagged_but_native_one_step_best=sum(x.get('factual_one_step_best',False) for x in native['checkpoints'].values()),
        actual_stage_exposure=supp['actual_stage_exposure'],fixed_fits=supp['fixed_fits'],
        fixed_trace_reward_links=supp['fixed_trace_links'],source_clustered_Bank=supp['source_clustered_Bank_contrasts'],
        conclusion='No promotion from degraded controls. R16 compares the original shared teacher recipe; '
                   'goal0 fallback was adopted, not proven the dominant cause. Test only kind>0; '
                   'whole-active guard rejected because it removes real-cargo danger escape. '
                   'Geometric first divergence can misrank native motion and is not actual Q/GAE.',
        limits='Sample endpoint counts are not whole-training contamination. Used map806/P0/allfar Bank. '
               'Own-label fit is not natural whole-chain ability; three seeds are a screen.',
        environment_steps_adopted=0,optimizer_steps=0)
    atomic_json(PRIOR/'FULL_HANDOFF_EVIDENCE.json',audit)
    return {k:audit[k] for k in ['comparison','parent','sampled_route_rows','sampled_goal0_rows',
                               'actual_goal0_replay_draws','conclusion','limits']}


def run():
    unique=lock(B/'handoff.lock')
    required=['GPU_COMPLETE.json','OWNER_EXIT.json','COMPLETE_EVIDENCE_CONSUMED.json',
              'SUPPLEMENTAL_EVIDENCE_CONSUMED.json']
    try:
        while True:
            if stopped(): raise UserStopped('user atomic stop overrides successor')
            for name in ['TECHNICAL_FAILURE.json','CPU_TECHNICAL_FAILURE.json']:
                if (PRIOR/name).exists(): raise RuntimeError('prior technical failure: '+name)
            subprocess.run(['node','scripts/dagger_rl_direction_trace_audit.js',str(PRIOR.relative_to(R))],
                           cwd=R,env=CPU_ENV,check=True,stdout=subprocess.DEVNULL)
            if all((PRIOR/n).exists() for n in required) and read(PRIOR/'FIXED_DEV_DIRECTION_AUDIT_v2.json')['completed']==19:
                break
            atomic_json(B/'HANDOFF_STATUS.json',dict(stage='R16_FIXED_BUDGET_AND_EVIDENCE',
                missing=[n for n in required if not (PRIOR/n).exists()],no_GPU_owner=True,unix=time.time()))
            time.sleep(10)
        from scripts.dagger_rl_r17_prepare import select
        select()
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).strip():
            if stopped(): raise UserStopped('user stop before resource acquisition')
            atomic_json(B/'HANDOFF_STATUS.json',dict(stage='RESOURCE_WAIT',no_preempt=True));time.sleep(5)
        if stopped(): raise UserStopped('user stop before launch')
        from scripts.dagger_rl_r17_run import authorize
        authorize()
        for name,h in read(B/'SOURCE_SEAL.json')['sources'].items():assert sha(R/name)==h,name
        # One dispatcher only; listener holds this lock through the full bounded experiment.
        lanes=[('scripts.dagger_rl_r17_run',[]),('scripts.dagger_rl_r17_cpu',[]),
               ('scripts.dagger_rl_r17_diagnostics',[]),('scripts.dagger_rl_r17_run',['complete']),
               ('scripts.dagger_rl_r17_monitor',[])]
        workers=[]
        for index,(module,args) in enumerate(lanes):
            with (B/f'lane_{index}.log').open('a') as f:
                workers.append(subprocess.Popen([sys.executable,'-m',module]+args,cwd=R,env=CPU_ENV,
                    stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
        atomic_json(B/'LAUNCH_RECEIPT.json',dict(passed=True,lanes=[m for m,_ in lanes],
            protocol_sha256=sha(B/'PROTOCOL.json'),unique_bounded_owner=True,old_continuous_resume=False))
        notify('R16完整证据冻结；R17仅kind目标资格处理已接续，三run复用原配对控制，先技术smoke后正式，旧队列不恢复。')
        while any(w.poll() is None for w in workers):
            if any(w.poll() not in [None,0] for w in workers):
                raise RuntimeError('bounded lane failed; inspect independent lane logs')
            time.sleep(5)
        atomic_json(B/'HANDOFF_COMPLETE.json',dict(passed=True,old_continuous_resume=False))
    except UserStopped as e:
        atomic_json(B/'HANDOFF_PAUSED.json',dict(safe_atomic=True,reason=str(e),old_continuous_resume=False))
    except Exception as e:
        atomic_json(B/'HANDOFF_FAILURE.json',dict(reason=str(e).replace(str(R),'<WORKTREE>')))
        notify('R17持久接续技术故障，证据保留，不重复owner、不恢复旧队列。');raise


if __name__=='__main__':run()
