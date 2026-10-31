"""Inactive successor preparation: require an existing, valid route goal kind.

No live teacher is patched on import. The original labels, safety candidates,
rewards, targets and item oracle are unchanged. Bind only in a new protocol.
"""
import jax.numpy as j


def guarded_teacher(original, inputs, labels, *, enabled=True):
    if not enabled:
        return original

    def teacher(states, aux):
        obs, _, legal, _ = inputs(states)
        seat = j.zeros(states.t.shape, j.int32)
        # The legacy active bit also rejects danger in the current cell. The
        # live adapter intentionally replaces that heuristic with temporal
        # joint safety, including real-cargo escape, so preserve its decisions.
        # kind0 means no stock/drop/home goal and must never supply a label.
        has_goal = labels(states, aux, legal, seat, obs[:, 5])[:, 6] > 0
        return original(states, aux) & has_goal[:, None, None]

    return teacher
