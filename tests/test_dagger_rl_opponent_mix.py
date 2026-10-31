"""Single-variable frozen-opponent mixture; no learner/teacher intervention."""
import importlib
import jax
import jax.numpy as j
import numpy as np


def module():
    return importlib.import_module('scripts.dagger_rl_opponent_mix')


def test_episode_assignment_does_not_consume_or_modify_environment_keys():
    mod = module()
    keys = jax.random.split(jax.random.PRNGKey(31), 4096)
    saved = np.asarray(keys).copy()
    assignment = mod.assign(keys)
    np.testing.assert_array_equal(assignment, mod.assign(keys))
    np.testing.assert_array_equal(keys, saved)
    assert 1800 < int(assignment.sum()) < 2300


def test_parent_actions_are_sampled_legally_and_disable_abilities_at_terminal():
    mod = module()
    n = 256
    ml = j.zeros((n, 5)); al = j.zeros((n, 3))
    mm = j.tile(j.array([1, 0, 1, 0, 0], bool), (n, 1))
    bm = j.ones((n, 3), bool)
    terminal = j.arange(n) < n//2
    actions = np.asarray(mod.sample_parent(ml, al, mm, bm,
                                         jax.random.split(jax.random.PRNGKey(7), n),
                                         terminal))
    assert set(actions[:, 0]) == {0, 2}
    assert set(actions[n//2:, 1]) == {0, 1, 2}
    np.testing.assert_array_equal(actions[:n//2, 1], 0)


def test_only_the_selected_opponent_is_replaced():
    mod = module()
    original = j.array([[[0, 0], [4, 1]], [[2, 2], [3, 0]]])
    parent = j.array([[1, 2], [0, 1]])
    selected = j.array([False, True])
    actual = np.asarray(mod.replace_enemy(original, parent, selected))
    np.testing.assert_array_equal(actual[:, 0], original[:, 0])
    np.testing.assert_array_equal(actual[0, 1], original[0, 1])
    np.testing.assert_array_equal(actual[1, 1], parent[1])
    np.testing.assert_array_equal(mod.replace_enemy(original, parent,
                                                   j.zeros(2, bool)), original)


def test_parent_view_uses_enemy_rows_and_own_masks(monkeypatch):
    mod = module()
    class State:
        t = j.arange(2)
    class Train:
        both_perspectives = staticmethod(lambda s: j.arange(4)[:, None])
        both_states = staticmethod(lambda s: j.arange(4)[:, None] + 10)
        both_masks = staticmethod(lambda s: (j.arange(20).reshape(4, 5),
                                            j.arange(12).reshape(4, 3)))
    obs, gv, mm, bm = mod.enemy_inputs(Train, State())
    np.testing.assert_array_equal(obs[:, 0], [2, 3])
    np.testing.assert_array_equal(gv[:, 0], [12, 13])
    np.testing.assert_array_equal(mm, j.arange(20).reshape(4, 5)[2:])
    np.testing.assert_array_equal(bm, j.arange(12).reshape(4, 3)[2:])
