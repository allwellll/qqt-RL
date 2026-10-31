import json
import pytest
from scripts import authorized_experiment_campaign as owner
from scripts.campaign_checkpoint_eval import validate_source
from scripts.population_run_state import sha


def test_atomic_checkpoint_must_exist_before_actual_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, 'ROOT', tmp_path)
    job = dict(id='eval', resource='evaluation', status='REGISTERED', dependencies=[],
               ready_files=['cp/COMMITTED.json', 'request.json'])
    assert owner.eligible([job], 'evaluation', paused=False) == []
    (tmp_path/'cp').mkdir(); (tmp_path/'cp/COMMITTED.json').write_text('{}')
    assert owner.eligible([job], 'evaluation', paused=False) == []
    (tmp_path/'request.json').write_text('{}')
    assert owner.eligible([job], 'evaluation', paused=False) == [job]
    assert owner.eligible([job], 'evaluation', paused=True) == []


def test_smoke_and_stale_queue_hash_are_rejected(tmp_path):
    cp = tmp_path/'cp'; cp.mkdir()
    (cp/'state.pkl').write_bytes(b'state'); (cp/'policy.pt').write_bytes(b'policy')
    marker = dict(state_sha256=sha(cp/'state.pkl'), policy_sha256=sha(cp/'policy.pt'))
    (cp/'COMMITTED.json').write_text(json.dumps(marker))
    spec = dict(checkpoint='cp', queue='queue.json')
    queue = dict(**marker, checkpoint='cp', stage='formal')
    (tmp_path/'queue.json').write_text(json.dumps(queue))
    assert validate_source(spec, tmp_path) == marker
    queue['stage'] = 'smoke'
    (tmp_path/'queue.json').write_text(json.dumps(queue))
    with pytest.raises(ValueError, match='formal'): validate_source(spec, tmp_path)
    queue.update(stage='formal', state_sha256='stale')
    (tmp_path/'queue.json').write_text(json.dumps(queue))
    with pytest.raises(ValueError, match='hash mismatch'): validate_source(spec, tmp_path)


def test_owner_does_not_claim_stale_smoke_request(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, 'ROOT', tmp_path)
    job = dict(id='eval', resource='evaluation', status='REGISTERED', dependencies=[],
               ready_json={'queue.json': dict(stage='formal', checkpoint='formal/cp')})
    assert owner.eligible([job], 'evaluation', paused=False) == []
    (tmp_path/'queue.json').write_text(json.dumps(dict(stage='smoke', checkpoint='smoke/cp')))
    assert owner.eligible([job], 'evaluation', paused=False) == []
    (tmp_path/'queue.json').write_text(json.dumps(dict(stage='formal', checkpoint='formal/cp')))
    assert owner.eligible([job], 'evaluation', paused=False) == [job]
