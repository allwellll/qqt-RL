"""The generated formal loop must capture post-bind callbacks."""
from pathlib import Path


def test_formal_binds_guard_before_dynamic_function_globals_are_copied():
    source=Path('scripts/dagger_rl_r17_train.py').read_text()
    assert "source.replace(anchor, 'bind()\\n    formal_function(L, bind=bind')" in source


def test_kind_wrapper_installs_both_student_streams_and_commit_telemetry():
    source=Path('scripts/dagger_rl_r17_train.py').read_text()
    assert "L.C.collector = lambda steps: factory" in source
    assert "L.auxiliary_collector = auxiliary" in source
    assert "L.commit = commit_with_goal_counts" in source
    assert "metrics['goal_guard_telemetry']" in source
