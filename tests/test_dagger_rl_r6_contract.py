"""The isolated opponent experiment must preserve legacy pause and learner math."""
import ast
import importlib
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
import pytest


def test_unregistered_or_stopped_owner_cannot_launch(tmp_path, monkeypatch):
    mod = importlib.import_module('scripts.dagger_rl_r6_run')
    monkeypatch.setattr(mod, 'B', tmp_path)
    with pytest.raises(AssertionError, match='scientific selection'):
        mod.authorize()


def test_opponent_assignment_survives_transition_without_entering_reward(monkeypatch):
    mod = importlib.import_module('scripts.dagger_rl_r6_train')
    monkeypatch.setattr(mod, 'BASE_SETUP', lambda: ('items', 'parent'))
    monkeypatch.setattr(mod, 'INSTALLED', False)
    keys = jax.random.split(jax.random.PRNGKey(9), 4)
    monkeypatch.setattr(mod.C.T, 'fresh', lambda *a: ('state', {'existing': j.arange(4)}))
    def transition(states, aux, actions, keys, arm):
        assert set(aux) == {'existing'}
        return states, aux, 'reward', 'done', 'info', 'parts'
    monkeypatch.setattr(mod.C.T, 'transition', transition)
    monkeypatch.setattr(mod.C.E, 'choose', mod.C.E.choose)
    assert mod.setup() == ('items', 'parent')
    st, ax = mod.C.T.fresh(keys, None, None)
    assert ax['opponent_candidate'].shape == (4,)
    assert not ax['opponent_enabled'].any()
    ax = dict(ax, opponent_enabled=j.ones(4, bool))
    after, na, reward, done, info, parts = mod.C.T.transition(st, ax, None, keys)
    assert (after, reward, done, info, parts) == ('state', 'reward', 'done', 'info', 'parts')
    for key in ax:
        np.testing.assert_array_equal(na[key], ax[key])


def test_only_auxiliary_treatment_flag_differs_at_paired_initialization(monkeypatch):
    mod = importlib.import_module('scripts.dagger_rl_r6_train')
    def initial(*args):
        return ('state', {'opponent_candidate': j.array([[0, 1]], bool),
                          'opponent_enabled': j.zeros((1,2), bool)}, 'keys', 'bank', 'aux')
    monkeypatch.setattr(mod, 'BASE_INITIAL', initial)
    monkeypatch.setattr(mod, 'ACTIVE_MIX', False)
    a = mod.initial(1, None)
    monkeypatch.setattr(mod, 'ACTIVE_MIX', True)
    b = mod.initial(1, None)
    assert a[0] == b[0] and a[2:] == b[2:]
    np.testing.assert_array_equal(a[1]['opponent_candidate'], b[1]['opponent_candidate'])
    assert not a[1]['opponent_enabled'].any() and b[1]['opponent_enabled'].all()


def test_memory_source_treatment_is_disabled_in_both_arms(monkeypatch):
    mod = importlib.import_module('scripts.dagger_rl_r6_train')
    observed = []
    def aggregate(accepted, sources, history, **kwargs):
        observed.append(kwargs)
        return 'history', tuple(np.zeros((8192,)) for _ in range(6)), {}
    monkeypatch.setattr(mod, 'aggregate_with_sources', aggregate)
    batch = [np.zeros((8, 1, 1024)) for _ in range(12)]
    batch[8] = np.zeros((8, 1, 1024, 5, 3), bool)
    mod.aggregate(tuple(batch), None, 1, 1)
    assert observed[0]['strategy'] == 'reservoir'
    assert observed[0]['balance_items'] is False


def test_namespaces_and_restore_entrypoints_are_isolated():
    for name in ['run', 'cpu', 'train', 'report', 'prepare', 'bank']:
        source = Path(f'scripts/dagger_rl_r6_{name}.py').read_text()
        ast.parse(source)
        if name == 'train':
            assert "'scripts.dagger_rl_r6_train','restore'" in source
            assert "fa = dict(fa, opponent_enabled=ax['opponent_enabled'])" in source
        if name == 'cpu':
            assert "'scripts/dagger_rl_r6_cpu.py'" in source
    from scripts import dagger_rl_r6_train as mod, dagger_rl_r4_train as frozen
    assert mod.updater is frozen.updater
    from scripts.dagger_rl_r6_run import B
    assert B.name == 'r6_opponent_candidate'
