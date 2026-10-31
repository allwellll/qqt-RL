from pathlib import Path
from scripts.bomb_threat_reboot_worker import adapted_source
from scripts.bomb_threat_reboot_recovery import dispatch
from scripts.population_run_state import atomic_json


def test_recovery_preserves_training_semantics_and_cumulative_budget():
    for module in ['scripts.bomb_threat_r1_progress_train', 'scripts.bomb_threat_iter2_diagnostics_train', 'scripts.bomb_threat_a3sr_resume_train']:
        source=adapted_source(module,'smoke',Path('runs/recovery_test'))
        assert 'end=min(total,start+2) if args.smoke else total' in source
        assert 'for index in range(start+1,end+1)' in source
        assert "pickle.load((cp/'state.pkl').open('rb'))" in source
        compile(source,module,'exec')


def test_missing_probe_bootstrap_never_overwrites_or_trains():
    s=adapted_source('scripts.bomb_threat_long_train','bootstrap',Path('runs/recovery_test'))
    assert "assert not token_path.exists()" in s
    assert 'training_updates=0,pre_reboot_nextstep_parity_available=False' in s
    assert s.index('token_path=')<s.index('for index in range(start+1,stop+1)')
    assert 'retained[-1]' in s
    compile(s,'test','exec')


def test_dispatch_does_not_touch_unlisted_or_already_released_slot(tmp_path,monkeypatch):
    import scripts.bomb_threat_reboot_recovery as R
    monkeypatch.setattr(R,'OUT',tmp_path)
    assert not dispatch(0,{'run':'other'}, {}, {}, {0}, '0')
    atomic_json(tmp_path/'PLAN.json',{'slots':{'0':{'run':'expected'}}})
    assert not dispatch(0,{'run':'other'}, {}, {}, {0}, '0')
    (tmp_path/'slot00').mkdir()
    atomic_json(tmp_path/'slot00/STATUS.json',{'status':'READY_FOR_FORMAL'})
    assert not dispatch(0,{'run':'expected'}, {}, {}, {0}, '0')
