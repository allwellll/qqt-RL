from scripts.population_run_state import atomic_json
from scripts.bomb_threat_catchup_acceptance import continuation_evidence


def test_catchup_accepts_both_real_receipt_abis(tmp_path):
    assert continuation_evidence(tmp_path) is None
    atomic_json(tmp_path/'SOURCE_RESTORE.json',{'passed':True})
    atomic_json(tmp_path/'TECHNICAL_CONTINUATION.json',{'passed':True,'prewarm':'real'})
    assert continuation_evidence(tmp_path)['abi']=='legacy-prewarm-v1'
    (tmp_path/'SOURCE_RESTORE.json').unlink()
    restored={k:True for k in ('passed','fresh_process','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact')}
    restored['checkpoint']='run/u000002'
    atomic_json(tmp_path/'RESTORE.json',restored)
    smoke=dict(passed=True,actor_critic_nonzero=True,checkpoint={'checkpoint':'run/u000002','update':2})
    atomic_json(tmp_path/'SMOKE.json',smoke)
    assert continuation_evidence(tmp_path)['formal_after_update']==2
    restored['Adam_exact']=False;atomic_json(tmp_path/'RESTORE.json',restored)
    assert continuation_evidence(tmp_path) is None
    restored['Adam_exact']=True;restored['checkpoint']='wrong';atomic_json(tmp_path/'RESTORE.json',restored)
    assert continuation_evidence(tmp_path) is None


def test_fixed_metrics_denominators_and_missing_are_explicit():
    from scripts.r3_sp_fixed_history_owner_20261031 import summary
    rows=[dict(seat=0,winner=-1,causal_kills=[2,1],deaths=[0,1],placed=[7,3],
               r3={'contacts':[dict(type=2,player=0,owner=0),dict(type=2,player=0,owner=1),dict(type=1,player=0,owner=0)]}),
          dict(seat=0,winner=0,causal_kills=[1,0],deaths=[0,0],placed=[5,4],r3={'contacts':[]})]
    value=summary(rows,0)
    assert value['win_rate']==.5 and value['placed_bombs_per_game']==6
    assert value['own_glue_contacts']==1 and value['own_glue_contacts_per_game']==.5
    assert value['draw_causal_kill_difference_per_draw']==1
    del rows[0]['placed'];del rows[0]['r3']
    value=summary(rows,0)
    assert value['placed_bombs_per_game'] is None and value['own_glue_contacts'] is None
    assert value['metric_coverage']==dict(placed_games=1,contact_games=1,games=2)


def test_fixed_cadence_forty_preserves_old_requests(tmp_path,monkeypatch):
    from scripts import r3_selfplay_eval_20261031 as bot
    from scripts import r3_sp_fixed_history_owner_20261031 as fixed
    monkeypatch.setattr(bot,'OUT',tmp_path);monkeypatch.setattr(fixed,'OUT',tmp_path)
    parent=dict(checkpoint='parent',policy_sha256='parent',update=3040)
    atomic_json(tmp_path/'REGISTRATION.json',dict(parent_source=parent))
    for update in (3080,3120,3160):
        atomic_json(tmp_path/f'formal/r3_sp55_c000000/eval_queue/u{update:06d}.json',dict(parent,checkpoint=str(update),update=update))
    assert {r['source']['update'] for r in fixed.requests()}=={3040,3080,3120,3160}


def test_u3127_resume_requires_complete_fresh_restore_and_keeps_budget(tmp_path,monkeypatch):
    from scripts import r3_selfplay_owner_20261031 as owner
    from scripts.r3_sp_reclaim_1820_20261031 import consume_resume
    from scripts.population_run_state import sha
    folder=tmp_path/'formal/run';cp=folder/'u003127';cp.mkdir(parents=True)
    (cp/'state.pkl').write_bytes(b'full Adam env aux RNG ring');(cp/'policy.pt').write_bytes(b'actor')
    atomic_json(cp/'COMMITTED.json',dict(update=3127,state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt')))
    atomic_json(folder/'NEXT_STEP_u003127.json',dict(digest='next'))
    atomic_json(folder/'LAUNCH.json',dict(phase='FORMAL'))
    atomic_json(folder/'RESTORE.json',dict(old=True))
    atomic_json(tmp_path/'RESOURCE_PAUSE.json',dict(old_pause=True))
    atomic_json(tmp_path/'REGISTRATION.json',dict(entry=dict(run='run',gpu=3)))
    atomic_json(tmp_path/'RECLAIM_RESUME.json',dict(status='AUTHORIZED',checkpoint=str(cp)))
    monkeypatch.setattr(owner,'processes',lambda *a:iter(()))
    monkeypatch.setattr(owner,'compute_app_count',lambda *a:0)
    monkeypatch.setattr(owner,'notify',lambda *a,**k:None)
    def launch(entry,phase):atomic_json(folder/'LAUNCH.json',dict(phase=phase))
    monkeypatch.setattr(owner,'launch',launch)
    assert consume_resume(tmp_path)
    assert (tmp_path/'RESOURCE_PAUSE.json').exists()
    restored={k:True for k in ('passed','fresh_process','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact')}
    atomic_json(folder/'RESTORE.json',dict(restored,checkpoint=str(cp)))
    assert not consume_resume(tmp_path)
    assert not (tmp_path/'RESOURCE_PAUSE.json').exists()
    request=owner.read(tmp_path/'RECLAIM_RESUME.json')
    assert request['remaining_updates']==553 and request['terminal_update']==3680
    assert (tmp_path/'attempts/reclaim1820/PRE_RESUME_RESOURCE_PAUSE.json').exists()


def test_explicit_1820_handoff_does_not_wait_missing_holdout(tmp_path,monkeypatch):
    from scripts import r3_sp_second_seed_20261031 as scheduler
    from scripts import r3_selfplay_owner_20261031 as owner
    from scripts.population_run_state import sha
    monkeypatch.setattr(scheduler,'OUT',tmp_path/'sp');monkeypatch.setattr(scheduler,'NS',tmp_path/'sp/seed02')
    monkeypatch.setattr(scheduler,'R3',tmp_path/'r3');monkeypatch.setattr(scheduler,'EVAL',tmp_path/'r3/eval')
    monkeypatch.setattr(scheduler,'freeze',lambda:dict(required_updates=[3840,3920,4000],baseline_highs={},seed=2026323002))
    monkeypatch.setattr(owner,'paused',lambda:False);monkeypatch.setattr(owner,'notify',lambda *a,**k:None)
    monkeypatch.setattr(owner,'processes',lambda *a:iter(()));monkeypatch.setattr(owner,'compute_app_count',lambda gpu:1)
    monkeypatch.setattr(scheduler.subprocess,'check_output',lambda *a,**k:'2, 80000\n')
    atomic_json(scheduler.OUT/'RECLAIM_1820_AUTHORIZATION.json',dict(override_wait_u4000_holdout=True))
    folder=scheduler.R3/'formal/control';cp=folder/'u004002';cp.mkdir(parents=True)
    (cp/'state.pkl').write_bytes(b'full');(cp/'policy.pt').write_bytes(b'actor')
    atomic_json(folder/'COMPLETE.json',dict(update=4002,checkpoint=str(cp),state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt')))
    atomic_json(folder/'NEXT_STEP_u004002.json',{})
    p=scheduler.R3/'protocols/control.json';atomic_json(p,dict(chunk_start_update=3362,chunk_updates=640))
    atomic_json(scheduler.R3/'R3_SLOTS.json',{'a':dict(run='control',protocol=str(p),gpu=2),'b':dict(run='control',protocol=str(p),gpu=2)})
    scheduler.advance()
    result=scheduler.read(scheduler.OUT/'SECOND_SEED_CONDITIONAL_1610.json')
    assert result['status']=='WAIT_UNKNOWN_COMPUTE_APP_NO_SIGNAL'


def test_authorized_reference_baseline_applies_once_and_strikes_persist(tmp_path,monkeypatch):
    from scripts import r3_selfplay_owner_20261031 as owner
    monkeypatch.setattr(owner,'OUT',tmp_path);monkeypatch.setattr(owner,'notify',lambda *a,**k:None)
    key='runs/bun_native_v2_20261030:b'
    atomic_json(tmp_path/'RESOURCE_GUARD.json',dict(baseline={key:dict(run='same',update=80,env_per_second=20.)},strikes={},last_update={}))
    atomic_json(tmp_path/'AMENDMENT05_RECLAIM_ACTIVE_REFERENCES.json',dict(reference_baselines={key:dict(run='same',update=100,env_per_second=100.)}))
    for update,strikes in [(105,1),(110,2),(115,3)]:
        monkeypatch.setattr(owner,'reference_snapshot',lambda:{key:dict(run='same',update=update,env_per_second=90.)})
        owner.resource_guard()
        assert owner.read(tmp_path/'RESOURCE_GUARD.json')['strikes'][key]==strikes
    assert owner.read(tmp_path/'RESOURCE_PAUSE.json')['references']==[key]


def test_first_formal_acceptance_waits_for_next_step_token(tmp_path,monkeypatch):
    from scripts import r3_selfplay_owner_20261031 as owner
    from scripts.r3_sp_reclaim_1820_20261031 import consume_resume
    from scripts.population_run_state import sha
    monkeypatch.setattr(owner,'notify',lambda *a,**k:None)
    folder=tmp_path/'formal/run';before=folder/'u003127';cp=folder/'u003128';cp.mkdir(parents=True)
    atomic_json(before/'COMMITTED.json',dict(environment_steps=1000000,optimizer_steps=100))
    atomic_json(tmp_path/'RECLAIM_RESUME.json',dict(status='CONSUMED',checkpoint=str(before)))
    atomic_json(tmp_path/'REGISTRATION.json',dict(entry={'run':'run'}))
    (cp/'state.pkl').write_bytes(b'new state');(cp/'policy.pt').write_bytes(b'new policy')
    atomic_json(cp/'COMMITTED.json',dict(environment_steps=1262144,optimizer_steps=132,
        state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'),metrics=dict(
            finite=True,update_l2=dict(actor=.1,critic=.2),actual_environment_steps_this_update=262144,cycle_seconds=20.)))
    assert not consume_resume(tmp_path)
    assert not (tmp_path/'RECLAIM_FIRST_FORMAL_ACCEPTANCE.json').exists()
    atomic_json(folder/'NEXT_STEP_u003128.json',dict(digest='complete'))
    assert not consume_resume(tmp_path)
    assert owner.read(tmp_path/'RECLAIM_FIRST_FORMAL_ACCEPTANCE.json')['next_step_token_verified']


def test_cold_baseline_correction_does_not_reset_existing_strikes(tmp_path,monkeypatch):
    from scripts import r3_selfplay_owner_20261031 as owner
    monkeypatch.setattr(owner,'OUT',tmp_path);monkeypatch.setattr(owner,'notify',lambda *a,**k:None)
    key='runs/bomb_threat_v1:5';baseline=dict(run='same',update=100,env_per_second=50.)
    atomic_json(tmp_path/'RESOURCE_GUARD.json',dict(baseline={key:baseline},strikes={key:2},last_update={key:110}))
    atomic_json(tmp_path/'AMENDMENT06_COLD_REFERENCE_EXCLUSION.json',dict(baseline_overrides={key:dict(baseline,env_per_second=100.)}))
    monkeypatch.setattr(owner,'reference_snapshot',lambda:{key:dict(run='same',update=115,env_per_second=80.)})
    owner.resource_guard()
    state=owner.read(tmp_path/'RESOURCE_GUARD.json')
    assert state['comparisons'][key]['ratio']==.8 and state['strikes'][key]==3
    assert owner.read(tmp_path/'RESOURCE_PAUSE.json')['references']==[key]
