from scripts.r3_sp_second_seed_20261031 import scores,eligible_arms


def doc(p0,p1):
    return {'metrics':{'holdout':{m:{'P0':{'games':64,'win_minus_loss':p0},'P1':{'games':64,'win_minus_loss':p1}}
                                    for m in ('greedy','categorical')}}}


def test_new_high_blocks_yield_and_separate_modes_retained():
    baseline={'a':{'P0':100,'P1':90},'b':{'P0':100,'P1':90}}
    observations={'a':[doc(50,45),doc(49,44)],'b':[doc(51,44)]}
    assert scores(observations['a'][0])=={'P0':100,'P1':90}
    assert eligible_arms(baseline,observations)==['a']
    observations['a'].append(doc(49,46))
    assert eligible_arms(baseline,observations)==[]


def test_second_seed_worker_only_changes_namespace():
    from scripts.r3_selfplay_train_20261031 import worker_source as first
    from scripts.r3_sp_seed2_train_20261031 import worker_source as second,NAMESPACE
    assert second().replace(f"BASE=ROOT/'{NAMESPACE}'","BASE=ROOT/'runs/r3_selfplay_ramp_20261031'")==first()
    compile(second(),'<SP2>','exec')


def test_missing_evaluations_cannot_release_a_worker(tmp_path,monkeypatch):
    from scripts import r3_sp_second_seed_20261031 as scheduler
    from scripts import r3_selfplay_owner_20261031 as owner
    monkeypatch.setattr(scheduler,'OUT',tmp_path)
    monkeypatch.setattr(scheduler,'NS',tmp_path/'seed02')
    monkeypatch.setattr(scheduler,'EVAL',tmp_path/'evaluation')
    monkeypatch.setattr(scheduler,'freeze',lambda:{'required_updates':[3840,3920,4000]})
    monkeypatch.setattr(owner,'paused',lambda:False)
    scheduler.advance()
    assert scheduler.read(tmp_path/'SECOND_SEED_CONDITIONAL_1610.json')['status']=='WAIT_U4000_COMPLETE_HOLDOUT'
    assert not (tmp_path/'seed02/REGISTRATION.json').exists()


def test_second_seed_metadata_matches_effective_seed_without_mutating_first():
    from scripts.r3_sp_second_seed_20261031 import second_seed_metadata
    original={'seed':2026323001,'resource':{'gpu':3,'training_cpu_affinity':'128-135'},'budget_updates':640}
    result=second_seed_metadata(original,2026323002,'runs/test/seed02',{'gpu':2,'cpu_affinity':'176-179'})
    assert original['seed']==2026323001 and original['resource']['gpu']==3
    assert result['seed']==2026323002 and result['resource']['gpu']==2 and result['budget_updates']==640
