"""Boundaries that must survive moving the live training closure onto main."""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from scripts import audit_combat_training_path as audit
from scripts import cc_sil_updater_v3 as sil
from scripts import model_launch
from scripts.cx_ftn2_train_v1 import transform_worker


def test_sil_requires_full_future_window_without_control_or_death():
    live = jnp.ones((45, 2), dtype=bool).at[20, 1].set(False)
    safe = np.asarray(sil.survived(live))
    assert safe[:5, 0].all()
    assert not safe[5:, 0].any()  # A truncated rollout cannot certify survival.
    assert not safe[:, 1].any()


def test_ftn2_worker_transform_preserves_bounded_budget_and_restore_gate():
    source = Path('scripts/combat_specialist_g_league_train.py').read_text()
    patched = transform_worker(source)
    compile(patched, '<ftn2-worker>', 'exec')
    assert "target=min(target,args.until)" in patched
    assert "'SEGMENT_COMPLETE'" in patched
    assert "diagnostic_steps_not_counted=steps" in patched
    with pytest.raises(AssertionError):
        transform_worker(source.replace("    args = parser.parse_args(); protocol_path", "missing_anchor"))


def test_generic_launcher_uses_current_interpreter_and_rejects_another_tree(monkeypatch, tmp_path):
    monkeypatch.delenv('PYTHON', raising=False)
    monkeypatch.delenv('QQT_PROJECT_ROOT', raising=False)
    import sys
    assert model_launch.command('training', 'local_config/example.json')[0] == sys.executable
    monkeypatch.setenv('QQT_PROJECT_ROOT', str(tmp_path))
    with pytest.raises(ValueError):
        model_launch.public_project()


def test_source_closure_has_no_missing_node_dependencies_or_opaque_entries():
    manifest = json.loads(Path('docs/combat_league_v2_dependency_manifest_20261106.json').read_text())
    result = audit.audit(manifest['roots'])
    assert result == manifest
