from scripts.campaign_eval_resources import limits, cpu_group, regression

def test_caps_and_disjoint_cpus():
    cfg={'limits':{'gpu':1,'cpu':3,'evaluation':4,'analysis':2,'js':1}}
    assert limits(cfg)['evaluation']==4
    groups=[set(cpu_group(list(range(192)),r,s)) for r,n in [('cpu',3),('evaluation',4),('analysis',2),('js',1)] for s in range(n)]
    assert sum(map(len,groups))<=70
    for i,a in enumerate(groups):
        for b in groups[i+1:]:assert not a&b
    assert not set(range(12))&set.union(*groups)

def test_gpu_regression_requires_same_run_warm_samples():
    assert regression(100,[100]*20) is False
    assert regression(100,[94]*20) is True
    assert regression(100,[1]*3) is None

def test_js_merge_rejects_duplicate_or_missing_seed():
    from scripts.campaign_js_parallel import validate_batch
    spec={'seed_start':100,'models':{'m':{'model_json_sha256':'h'}}}
    value={'passed':True,'model_json_sha256':'h','results':{'natural_full2400':{'episodes':[{'seed':100},{'seed':101}]}}}
    assert len(validate_batch(value,spec,'m',0,2))==2
    value['results']['natural_full2400']['episodes'][1]['seed']=100
    import pytest
    with pytest.raises(AssertionError):validate_batch(value,spec,'m',0,2)


def test_scale_and_v2_throughput_abi():
    from scripts.campaign_eval_resources import row_rate
    assert row_rate(dict(env_per_second=3000))==3000
    assert row_rate(dict(actual_environment_steps_this_update=16384,cycle_seconds=8))==2048

def test_funnel_environment_binds_before_frozen_module_import(tmp_path,monkeypatch):
    import json,os
    from scripts.campaign_funnel_bootstrap import bootstrap
    for key in ('FULLMATCH_OUTPUT','FULLMATCH_GAMES_PER_MODE','FULLMATCH_SEED_START','FULLMATCH_MODELS','CUDA_VISIBLE_DEVICES','JAX_PLATFORMS','JAX_PLATFORM_NAME'):
        monkeypatch.setenv(key,'unset')
    p=tmp_path/'protocol.json';p.write_text(json.dumps(dict(namespace='runs/funnel',conditions={'parent':dict(games_per_mode=304,seed_start=123)})))
    receipt=bootstrap(str(p),'parent')
    assert receipt['games_per_mode']==304
    assert os.environ['FULLMATCH_GAMES_PER_MODE']=='304'
    assert os.environ['FULLMATCH_SEED_START']=='123'
    assert os.environ['FULLMATCH_OUTPUT'].endswith('/runs/funnel/parent')
    assert os.environ['CUDA_VISIBLE_DEVICES']==''

def test_limits_reject_partition_overflow():
    import pytest
    with pytest.raises(ValueError):limits({'limits':{'evaluation':5}})
    with pytest.raises(ValueError):cpu_group(list(range(192)),'cpu',3)
    groups=[cpu_group(list(range(192)),r,s) for r,n in [('cpu',3),('evaluation',4),('analysis',2),('js',1)] for s in range(n)]
    assert sum(map(len,groups))+48+12<=88

def test_adoption_preserves_existing_slot():
    from scripts.campaign_eval_resources import scheduling_slot
    active={'evaluation:3':None}
    assert scheduling_slot('evaluation',active,preferred=1)==1
    assert scheduling_slot('evaluation',active,preferred=3)==0

def test_rate_uses_continuous_active_revision_and_excludes_pause():
    from scripts.campaign_eval_monitor import warm_rate
    def sample(t,n,suspended=False,revision='a'):
        return {'unix':t,'scheduler_revision':revision,'workers':[dict(job='j',completed_games=n,suspended=suspended,cpu_affinity=[1,2])]}
    history=[sample(0,8),sample(120,24),sample(180,24,True),sample(240,24),sample(360,48)]
    assert warm_rate(history,'j')==12
    assert warm_rate(history+[sample(420,56,revision='b')],'j') is None

def test_missing_js_truncation_is_not_false():
    from scripts.campaign_js_parallel import truncation_coverage
    assert truncation_coverage([{}, {'artificial_truncation':False},{'artificial_truncation':True}])==dict(available=2,missing=1,flagged=1)

def test_gradual_ramp_requires_twenty_healthy_updates():
    from scripts.campaign_eval_resources import ramp_ready
    assert not ramp_ready(100,[100]*19)
    assert not ramp_ready(100,[97]*20)
    assert ramp_ready(100,[99]*20)

def test_passive_monitor_accepts_null_analysis_budget(tmp_path):
    import json
    from scripts.campaign_eval_monitor import tick
    base=tmp_path/'campaign';base.mkdir()
    (base/'EVAL_PARALLEL_CONFIG.json').write_text(json.dumps({'limits':{'gpu':1,'cpu':3,'evaluation':4,'analysis':2,'js':1}}))
    (base/'EVAL_PARALLEL_BASELINE.json').write_text(json.dumps({'active':[],'zombies':[]}))
    (tmp_path/'progress.json').write_text(json.dumps({'completed_games':0}))
    job=dict(id='analysis',resource='analysis',status='REGISTERED',budget=None,args=[],complete='progress.json')
    tick({'jobs':[job]}, {},tmp_path,base)
    value=json.loads((base/'EVAL_PARALLEL_ACCEPTANCE.json').read_text())
    assert value['worker_counts']['analysis']==dict(present=0,working=0,suspended=0)
    assert value['eta_by_resource'][-1]['budget_unknown_jobs']==['analysis']

def test_merge_rejects_seed_and_trace_corruption(tmp_path):
    import json
    import pytest
    from scripts.campaign_eval_merge import verify
    from scripts.population_run_state import sha
    (tmp_path/'state.pkl').write_bytes(b'state')
    (tmp_path/'trace.bin').write_bytes(b'raw trace')
    out=tmp_path/'out'/'m';(out/'batches').mkdir(parents=True)
    (out/'MANIFEST.json').write_text('{}');(out/'SUMMARY.json').write_text('{}')
    protocol=dict(namespace='out',games_per_condition=1,dev_seed_start=10,checkpoints={'m':{'state':'state.pkl'}})
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (out/'COMPLETE.json').write_text(json.dumps(dict(passed=True,total_games=2,summary_sha256=sha(out/'SUMMARY.json'),manifest_sha256=sha(out/'MANIFEST.json'))))
    row=dict(episode_index=0,seed=10,provenance=dict(state_sha256=sha(tmp_path/'state.pkl'),trace='trace.bin',trace_sha256=sha(tmp_path/'trace.bin')),artificial_truncation=False,termination='native_capture')
    for mode in ('categorical','greedy'):
        (out/'batches'/f'{mode}_00000.json').write_text(json.dumps(dict(passed=True,source_state_sha256=sha(tmp_path/'state.pkl'),episodes=[row])))
    assert verify('protocol.json','m','native',tmp_path)['total_games']==2
    (tmp_path/'trace.bin').write_bytes(b'corrupt')
    with pytest.raises(AssertionError):verify('protocol.json','m','native',tmp_path)
    (tmp_path/'trace.bin').write_bytes(b'raw trace')
    path=out/'batches/categorical_00000.json';value=json.loads(path.read_text());value['episodes'][0]['seed']=11;path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):verify('protocol.json','m','native',tmp_path)
