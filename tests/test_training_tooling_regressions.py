"""Regression test for critic legal-Q reporting."""
import jax.numpy as jnp

from scripts.eval_critic_decided_cases import legal_best_q


def test_legal_best_q_ignores_illegal_actions():
    q = jnp.asarray([[1.0, 99.0, 3.0, 4.0, 5.0, 6.0]])
    move_mask = jnp.asarray([[[True, False, True]]])
    ability_mask = jnp.asarray([[[True, False]]])
    # Joint flattening order: move-major, ability-minor. Legal are indices 0 and 4.
    got = legal_best_q(q, move_mask, ability_mask)
    assert float(got[0]) == 5.0
