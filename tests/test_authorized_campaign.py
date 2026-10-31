from scripts import authorized_experiment_campaign as c


def test_gpu_idle_after_final_budget_does_not_imply_missing_work():
    jobs=[dict(id='g',resource='gpu',status='COMPLETE'),
          dict(id='bank',resource='analysis',status='REGISTERED')]
    assert c.gpu_idle_reason(jobs)=='NO_REGISTERED_GPU_SUCCESSOR'
    jobs[0]['status']='REGISTERED'
    assert c.gpu_idle_reason(jobs)=='GPU_PENDING_DEPENDENCIES_OR_TECHNICAL_WORK'


def test_progress_age_does_not_invent_updates(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    job=dict(id='x',complete='COMPLETE.json',progress='PROGRESS.json',started_unix=10.)
    result=c.progress_observation(job,210.)
    assert result['progress_age_seconds']==200.
    assert result['completed_iterations'] is None and not result['progress_exists']
    assert result['process_presence_is_not_update_evidence']


def test_request_claim_race_preserves_worker_and_owner():
    jobs=[dict(id='x',status='RUNNING',env={'frozen':'unchanged'})]
    receipt=c.replace_pending_job(jobs,dict(id='x',status='REGISTERED',env={'extra':'logging'}))
    assert receipt['consumed'] and not receipt['passed']
    assert jobs[0]['status']=='RUNNING' and jobs[0]['env']=={'frozen':'unchanged'}
    jobs[0]['status']='REGISTERED'
    assert c.replace_pending_job(jobs,dict(id='x',env={'extra':'logging'}))['passed']


def test_retry_only_explicitly_failed_job_never_completed_work():
    jobs=[dict(id='x',status='FAILED')]
    assert c.replace_pending_job(jobs,dict(id='x',status='REGISTERED'),retry_failed=True)['passed']
    jobs[0]['status']='COMPLETE'
    assert not c.replace_pending_job(jobs,dict(id='x',status='REGISTERED'),retry_failed=True)['passed']


def test_dependency_and_pause_control_actual_eligible_queue():
    jobs=[dict(id='p',resource='gpu',dependencies=[],status='COMPLETE'),dict(id='x',resource='gpu',dependencies=['p'],status='REGISTERED')]
    assert [j['id'] for j in c.eligible(jobs,'gpu',paused=False)]==['x']
    assert c.eligible(jobs,'gpu',paused=True)==[]
    jobs[0]['status']='FAILED'
    assert c.eligible(jobs,'gpu',paused=False)==[]


def test_completed_job_is_not_repeated_and_pending_design_not_executable():
    jobs=[dict(id='a',resource='cpu',dependencies=[],status='COMPLETE'),dict(id='b',resource='cpu',dependencies=[],status='AWAITING_PROTOCOL')]
    assert c.eligible(jobs,'cpu',paused=False)==[]


def test_gpu_release_requires_all_cards_zero_and_no_app():
    assert c.resources_free([0]*8,[])
    assert not c.resources_free([0]*7,[])
    assert not c.resources_free([0]*7+[1],[])
    assert not c.resources_free([0]*8,['unrelated'])


def test_resume_reconciles_only_real_completion_markers(tmp_path):
    assert not c.marker_passes(tmp_path/'COMPLETE.json')
    (tmp_path/'COMPLETE.json').write_text('{"passed": false}')
    assert not c.marker_passes(tmp_path/'COMPLETE.json')
    (tmp_path/'COMPLETE.json').write_text('{"passed": true}')
    assert c.marker_passes(tmp_path/'COMPLETE.json')


def test_progress_disappears_after_exists_does_not_crash_owner(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    path=tmp_path/'PROGRESS.json';path.write_text('{"iterations": 12}')
    def disappeared(_): raise FileNotFoundError()
    monkeypatch.setattr(c,'read',disappeared)
    result=c.progress_observation(dict(id='x',complete='COMPLETE.json',progress='PROGRESS.json'),100.)
    assert result['completed_iterations'] is None
    assert result['progress_read_error']=='FileNotFoundError'


def test_pending_science_requires_explicit_protocol_registration():
    jobs=[dict(id='x',status='AWAITING_PROTOCOL')]
    assert not c.replace_pending_job(jobs,dict(id='x',status='REGISTERED'))['passed']
    assert c.replace_pending_job(jobs,dict(id='x',status='REGISTERED'),register_protocol=True)['passed']


def test_checkpoint_completion_requires_actual_matching_payload(tmp_path,monkeypatch):
    import json
    from scripts.population_run_state import sha
    monkeypatch.setattr(c,'ROOT',tmp_path)
    cp=tmp_path/'cp';cp.mkdir();(cp/'state.pkl').write_bytes(b'complete')
    (cp/'policy.pt').write_bytes(b'actor')
    done=dict(passed=True,checkpoint='cp',state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'))
    (tmp_path/'COMPLETE.json').write_text(json.dumps(done))
    assert c.job_completion_passes(dict(complete='COMPLETE.json'))
    (cp/'state.pkl').write_bytes(b'corrupt')
    assert not c.job_completion_passes(dict(complete='COMPLETE.json'))


def test_accept_delegated_receipt_verifies_hashes_and_never_claims_worker(tmp_path,monkeypatch):
    import json
    from scripts.population_run_state import sha
    monkeypatch.setattr(c,'ROOT',tmp_path)
    (tmp_path/'protocol.json').write_text('{}');(tmp_path/'side.json').write_text('{"status":"COMPLETE"}')
    receipt=dict(passed=True,task='side',side_complete_path='side.json',side_complete_sha256=sha(tmp_path/'side.json'),input_hashes=[dict(path='protocol.json',expected=sha(tmp_path/'protocol.json'),passed=True)],scientific_acceptance=False)
    path=tmp_path/'receipt.json';path.write_text(json.dumps(receipt))
    jobs=[dict(id='side',status='DELEGATED_SIDE_CX',complete='old.json',resource='analysis')]
    assert c.accept_delegated_result(jobs,'side','receipt.json',sha(path))['passed']
    assert jobs[0]['status']=='COMPLETE' and jobs[0]['complete']=='receipt.json'
    assert jobs[0]['scientific_acceptance'] is False
    jobs[0]['status']='DELEGATED_SIDE_CX';(tmp_path/'protocol.json').write_text('{"changed":true}')
    assert not c.accept_delegated_result(jobs,'side','receipt.json',sha(path))['passed']
    assert jobs[0]['status']=='DELEGATED_SIDE_CX'
    jobs[0]['status']='RUNNING'
    assert not c.accept_delegated_result(jobs,'side','receipt.json',sha(path))['passed']
