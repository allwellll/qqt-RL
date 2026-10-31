"""Candidate source screen is isolated and preserves the frozen training math."""
import ast
import importlib
from pathlib import Path
import numpy as np
import pytest


def test_prepare_is_not_authority_to_train(tmp_path, monkeypatch):
    run = importlib.import_module('scripts.dagger_rl_r5_run')
    monkeypatch.setattr(run, 'B', tmp_path)
    with pytest.raises(AssertionError, match='scientific selection'):
        run.authorize()


def test_collector_only_attaches_matching_passive_sources(monkeypatch):
    mod = importlib.import_module('scripts.dagger_rl_r5_train')
    batch = tuple(np.arange(6).reshape(2, 3) + k for k in range(11))
    origins = np.array([[-1, 0, -1], [2, -1, 3]], np.int32)
    diag = (1, 2, 3, 4, 5, 6, origins)
    monkeypatch.setattr(mod, 'BASE_COLLECTOR', lambda steps: lambda *a: ('end', batch, diag))
    end, actual, diagnostics = mod.collector(8)('unused')
    assert end == 'end' and diagnostics is diag
    for a, b in zip(actual[:11], batch):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(actual[11], origins)


def test_entrypoints_cannot_restore_r4_smoke_or_launch_legacy_queue():
    source = Path('scripts/dagger_rl_r5_train.py').read_text()
    assert "'scripts.dagger_rl_r5_train','restore'" in source
    assert "'scripts.dagger_rl_r4_train','restore'" not in source
    for path in ['scripts/dagger_rl_r5_run.py', 'scripts/dagger_rl_r5_cpu.py']:
        tree = ast.parse(Path(path).read_text())
        literals = [n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert not any('scripts.dagger_rl_r4_train' == v for v in literals)
        assert not any('scripts/dagger_rl_r4_cpu.py' == v for v in literals)
    assert 'scripts/dagger_rl_r5_cpu.py' in [
        n.value for n in ast.walk(ast.parse(Path('scripts/dagger_rl_r5_cpu.py').read_text()))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)]
