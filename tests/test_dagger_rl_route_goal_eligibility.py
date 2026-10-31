"""Inactive successor guard; legacy mode reproduces the sealed omission."""
import os
from collections import namedtuple
import jax.numpy as j
import numpy as np
import pytest
from scripts import combat_item_engine as T
from scripts import je10_route_oracle as oracle


@pytest.mark.parametrize('active,kind,carrying,enemy_alive,enemy_trapped,expected', [
    (False, 0, False, False, 0, False),
    (False, 2, False, False, 0, True),
    (True, 2, False, False, 0, True),
    (False, 3, True, True, 0, True),
    (True, 3, True, True, 0, True),
    (True, 2, False, True, 0, False),
    (True, 0, False, False, 0, False),
    (True, 1, False, False, 0, True),
    (True, 4, False, False, 0, True),
    (True, 2, False, True, 5, True),
])
def test_teacher_requires_goal_kind_and_preserves_temporal_safety_eligibility(
        monkeypatch, active, kind, carrying, enemy_alive, enemy_trapped, expected):
    State = namedtuple('State', ['t', 'bun_carried', 'alive', 'trapped'])
    state = State(j.zeros(1, j.int32), j.array([[1 if carrying else -1, -1]]),
                  j.array([[True, enemy_alive]]), j.array([[0, enemy_trapped]]))
    aux = dict(terminal=j.array([False]),
               carry_items=dict(ledger=dict(custody=j.array([[carrying, False]]))))
    inputs = lambda s: (j.zeros((1, 24, 13, 15)), j.zeros((1, 24)),
                         j.ones((1, 5), bool), j.ones((1, 3), bool))
    raw = j.zeros((1, 12)).at[0, 0].set(1).at[0, 5].set(active).at[0, 6].set(kind)
    labels = lambda *args: raw
    monkeypatch.setattr(T.E, 'inputs', inputs)
    monkeypatch.setattr(oracle, 'labels', labels)
    Safety = namedtuple('Safety', ['survivable'])
    monkeypatch.setattr(T.bun_safety, 'analyze_actions',
        lambda s: Safety(survivable=j.ones((2, 5, 3), bool)))
    teacher = T.route_teacher
    if os.environ.get('DAGGER_GOAL_TEST_LEGACY') != '1':
        from scripts.dagger_rl_route_goal_eligibility import guarded_teacher
        teacher = guarded_teacher(teacher, inputs, labels)
    actual = np.asarray(teacher(state, aux))
    assert bool(actual.any()) == expected
    if expected:
        assert actual[0, 0].all()
        assert not actual[0, 1:].any()


def test_disabled_guard_is_the_exact_original_callable():
    from scripts.dagger_rl_route_goal_eligibility import guarded_teacher
    original = lambda states, aux: object()
    assert guarded_teacher(original, None, None, enabled=False) is original
