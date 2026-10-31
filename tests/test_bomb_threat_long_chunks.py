from scripts import bomb_threat_schedule as S


def test_chunk_and_checkpoint_cadence_are_independent():
    assert S.chunk_updates({})==320
    assert S.chunk_updates({'chunk_updates':640})==640
    assert S.should_evaluate(720) and not S.should_evaluate(721)
    assert S.should_evaluate(1680) and not S.should_evaluate(1682)


def test_prewarm_fingerprint_excludes_checkpoint_and_chunk_not_recipe():
    a=dict(common={'envs_per_device':2048},arms={'a':{'lr':.000075}},source_sha256={'worker':'hash'},
        spawn_manifest='spawns.json',rule_version='stock-window-v1',reward_version='bomb-threat-v1',slot=0)
    b=dict(a,arms={'b':{'lr':.000075}},sources={'s':{'checkpoint':'later'}},chunk=99)
    assert S.prewarm_fingerprint(a)==S.prewarm_fingerprint(b)
    b['arms']={'b':{'lr':.0001}}
    assert S.prewarm_fingerprint(a)!=S.prewarm_fingerprint(b)


def test_warmup_acceptance_requires_both_nonzero_update_and_fresh_full_restore():
    smoke={'passed':True,'checkpoint':{'metrics':{'finite':True,'update_l2':{'actor':.1,'critic':.2}}}}
    restore={'passed':True,'fresh_process':True,'full_env_aux_RNG_exact':True,'next_step_exact':True,'Adam_exact':True,'ring_exact':True}
    assert S.valid_prewarm(smoke,restore)
    assert not S.valid_prewarm(smoke,dict(restore,next_step_exact=False))
    smoke['checkpoint']['metrics']['update_l2']['critic']=0
    assert not S.valid_prewarm(smoke,restore)
