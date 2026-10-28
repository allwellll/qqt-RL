from scripts import combat_mix_owner as O


def test_failed_adopted_smoke_is_blocked_and_not_reclaimed(tmp_path,monkeypatch):
    monkeypatch.setattr(O,'BASE',tmp_path)
    out=tmp_path/'formal/legacy_mix_s1';out.mkdir(parents=True)
    (out/'SMOKE.json').write_text('{"passed":false,"status":"BLOCKED_THROUGHPUT"}')
    got=O.technical_failure('legacy_mix_s1')
    assert got['status']=='BLOCKED_THROUGHPUT'
    assert got['source']=='persisted_smoke'


def test_arm_hold_does_not_block_unrelated_healthy_arms(tmp_path,monkeypatch):
    monkeypatch.setattr(O,'BASE',tmp_path)
    (tmp_path/'ARM_TECHNICAL_HOLD.json').write_text('{"legacy_mix":{"status":"HOLD_PERFORMANCE"}}')
    assert O.technical_failure('legacy_mix_s2')['status']=='HOLD_PERFORMANCE'
    assert O.technical_failure('opp_mix_curriculum_s2') is None


def test_gpu_reservation_includes_unallocated_old_worker(tmp_path):
    d=tmp_path/'123';d.mkdir()
    (d/'cmdline').write_bytes(b'python\0-m\0scripts.campaign_scale_train\0old.json\0')
    (d/'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=6\0JAXBOMB_DEVICE_OVERRIDE=6\0')
    (d/'stat').write_text('123 (python) S 0')
    assert 6 in O.worker_reserved_devices(tmp_path)


def test_exclusive_recheck_exception_does_not_release_other_seeds(tmp_path,monkeypatch):
    monkeypatch.setattr(O,'BASE',tmp_path)
    (tmp_path/'ARM_TECHNICAL_HOLD.json').write_text('{"legacy_mix":{"status":"HOLD","only_seed":1}}')
    assert O.technical_failure('legacy_mix_s1') is None
    assert O.technical_failure('legacy_mix_s2')['status']=='HOLD'


def test_versioned_run_namespace_is_consumed_by_real_owner(tmp_path,monkeypatch):
    monkeypatch.setattr(O,'BASE',tmp_path);monkeypatch.setattr(O,'ROOT',tmp_path)
    (tmp_path/'protocols').mkdir()
    (tmp_path/'protocols/legacy_mix_s1.json').write_text('{"namespace":"native_v2"}')
    assert O.run_output('legacy_mix_s1')==tmp_path/'native_v2/legacy_mix_s1'
