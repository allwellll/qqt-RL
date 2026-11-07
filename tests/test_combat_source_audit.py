"""Static closure checks never import training modules or initialize an accelerator."""
import json
from pathlib import Path

import pytest

from scripts import audit_combat_training_path as audit


def test_current_source_manifest_matches_exact_graph_and_sha():
    manifest = json.loads(Path(audit.DEFAULT_MANIFEST).read_text())
    assert audit.audit(manifest['roots']) == manifest
    assert 'web/sim-bun-native-v4-facing.js' in manifest['files']


def test_only_explicit_guarded_optional_bot_may_be_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'web').mkdir()
    sim = tmp_path / 'web/sim-bun-native-v4-facing.js'
    sim.write_text("try { require('./time_astar_ai.js'); } catch (_) {}")
    assert audit.dependencies('web/sim-bun-native-v4-facing.js') == []
    sim.write_text("require('./unavailable_physics.js')")
    with pytest.raises(FileNotFoundError):
        audit.dependencies('web/sim-bun-native-v4-facing.js')
    (tmp_path / 'web/other.js').write_text("require('./time_astar_ai.js')")
    with pytest.raises(FileNotFoundError):
        audit.dependencies('web/other.js')
