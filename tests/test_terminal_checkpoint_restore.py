"""Terminal restore uses the immutable saved sources after successor install."""
import hashlib
import json
import pytest
from scripts.terminal_checkpoint_restore import verify_saved_sources


def test_saved_source_proof_survives_successor_change_but_rejects_snapshot_tamper(tmp_path):
    base=tmp_path/'run';saved=base/'inputs/sealed_sources/scripts/worker.py'
    saved.parent.mkdir(parents=True);saved.write_text('version = 1\n')
    (base/'plan.json').write_text('{}\n')
    seal={'sources':{'scripts/worker.py':hashlib.sha256(saved.read_bytes()).hexdigest()},
          'plan_sha256':hashlib.sha256((base/'plan.json').read_bytes()).hexdigest()}
    (base/'SOURCE_SEAL.json').write_text(json.dumps(seal))
    current=tmp_path/'scripts/worker.py';current.parent.mkdir();current.write_text('version = 2\n')
    assert verify_saved_sources(base)['saved_source_count']==1
    saved.write_text('version = 3\n')
    with pytest.raises(AssertionError):verify_saved_sources(base)


def test_saved_source_proof_rejects_changed_plan(tmp_path):
    (tmp_path/'plan.json').write_text('{}')
    (tmp_path/'SOURCE_SEAL.json').write_text(json.dumps({'sources':{},'plan_sha256':'wrong'}))
    with pytest.raises(AssertionError):verify_saved_sources(tmp_path)


def test_checkpoint_schema_is_matched_to_sealed_run_plan():
    from scripts.terminal_checkpoint_restore import verify_state_schema
    import pytest
    for schema in ['je10-control-route-v38','je10-control-route-v38-prestock-v1']:
        assert verify_state_schema({'state_schema':schema},{'state_schema':schema})==schema
    with pytest.raises(AssertionError):
        verify_state_schema({'state_schema':'je10-control-route-v38-prestock-v1'},
                            {'state_schema':'je10-control-route-v38'})


def test_clean_causal_state_schema_restores_only_against_matching_plan():
    from scripts.terminal_checkpoint_restore import verify_state_schema
    schema='je10-control-route-v38-clean-causal-init-v1'
    assert verify_state_schema({'state_schema':schema},{'state_schema':schema})==schema
