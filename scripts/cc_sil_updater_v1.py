"""Inherited mixed PPO updater plus advantage-weighted self-imitation on sampled placements.

The extra term is -coef * mean(relu(A_norm) * log pi(ability=1)) over rows where the
policy itself sampled a legal placement and the on-policy normalized advantage is
positive. It is not a reward: placements that led to worse-than-expected outcomes
(self-defeat, empty or stalling bubbles) have A<=0 and get no imitation weight, and
the PPO term still pushes them down. No action is replaced online.
"""
import jax
import jax.numpy as j
from scripts import r4_k4_collector as common

BOMB = 1


def objective(coef):
    def ppo_objective(params, batch):
        from scripts.r4_k4_observation import actor_critic_forward
        from jax_bomb.jax_train import hl_gauss_value_loss
        obs, global_state, actions, old_logp, advantage, returns, move_mask, ability_mask = batch[:8]
        move, ability, _, value_logits = actor_critic_forward(params, obs, global_state)
        lm = jax.nn.log_softmax(j.where(move_mask, move, -1e9))
        la = jax.nn.log_softmax(j.where(ability_mask, ability, -1e9))
        rows = j.arange(len(old_logp))
        selected = lm[rows, actions[:, 0]] + la[rows, actions[:, 1]]
        ratio = j.exp(selected - old_logp)
        policy = j.maximum(-advantage * ratio, -advantage * j.clip(ratio, .9, 1.1)).mean()
        value = hl_gauss_value_loss(value_logits, returns)
        entropy = -(j.sum(j.exp(lm) * lm, -1) + j.sum(j.exp(la) * la, -1)).mean()
        weight = j.where((actions[:, 1] == BOMB) & ability_mask[:, BOMB].astype(bool), jax.nn.relu(advantage), 0.)
        sil = -(jax.lax.stop_gradient(weight) * la[:, BOMB]).sum() / len(old_logp)
        loss = policy + .5 * value - .005 * entropy + coef * sil
        return loss, j.array([policy, value, entropy, (old_logp - selected).mean(), (j.abs(ratio - 1.) > .1).mean()])
    return ppo_objective


def updater(coef):
    """Same scan/Adam/normalization as r4_k4_mixed.updater, only the objective differs."""
    from types import FunctionType, ModuleType
    from scripts import r4_k4_mixed as mixed
    custom = ModuleType('cc_sil_common'); custom.__dict__.update(common.__dict__)
    custom.ppo_objective = objective(coef)
    def build(minibatch, optimizer):
        fn = FunctionType(mixed.updater.__code__, dict(mixed.updater.__globals__, common=custom), mixed.updater.__name__)
        return fn(minibatch, optimizer)
    return build
