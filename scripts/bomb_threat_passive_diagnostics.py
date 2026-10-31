"""Detached policy statistics; no RNG, action, reward or optimizer mutation."""
import jax
import jax.numpy as j


def policy_sums(move_logp, ability_logp, live=None):
    entropy = -(j.sum(j.exp(move_logp)*move_logp, -1) +
                j.sum(j.exp(ability_logp)*ability_logp, -1))
    weights = j.ones_like(entropy) if live is None else live.astype(entropy.dtype)
    return jax.tree.map(jax.lax.stop_gradient, dict(
        policy_entropy_sum=j.sum(entropy*weights),
        bomb_probability_sum=j.sum(j.exp(ability_logp[:, 1])*weights),
        bomb_argmax_count=j.sum((j.argmax(ability_logp, -1)==1)*weights),
        policy_diagnostic_ticks=j.sum(weights)))


POLICY_DIAGNOSTICS_VERSION = 'masked-live-collector-v1'
