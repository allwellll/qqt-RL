"""Inactive next-round candidate; importing it changes no training behavior.

Retain a frozen parent's joint distribution only in observable empty combat.
No reward, action target, curriculum, model input or evaluation override.
"""
import jax
import jax.numpy as j


def combat_rows(obs, global_state):
    """No cargo, native window, real enemy-origin loose target or inactivity."""
    return jax.lax.stop_gradient(
        (global_state[:, 9] == 1) & (global_state[:, 10] == 1)
        & (global_state[:, 13] == 0)
        & ~j.any(obs[:, 17] > 0, axis=(-2, -1)))


def reference_kl(move, ability, parent_move, parent_ability,
                 legal_move, legal_ability, eligible):
    """Parent-to-student factored joint KL on the same legal public inputs.

    The parent never receives gradients. Masked actions and off-phase rows
    have exactly zero raw auxiliary gradient. Empty eligibility is finite zero.
    """
    terms = []
    for current, reference, legal in [(move, parent_move, legal_move),
                                      (ability, parent_ability, legal_ability)]:
        lp = jax.nn.log_softmax(j.where(legal, current, -1e9), -1)
        lq = jax.lax.stop_gradient(
            jax.nn.log_softmax(j.where(legal, reference, -1e9), -1))
        terms.append(j.sum(j.exp(lq) * (lq - lp), -1))
    gate = jax.lax.stop_gradient(eligible)
    return j.sum(j.where(gate, terms[0] + terms[1], 0.)) / j.maximum(gate.sum(), 1)
