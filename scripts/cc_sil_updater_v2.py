"""Placement self-imitation restricted to placements the learner survives.

Same PPO math, normalization, scan and Adam as r4_k4_mixed.updater. The imitation
weight relu(A_norm) is kept only when the learner stays live for the next
SURVIVE ticks of the same time-major rollout; rows without a full window (the last
SURVIVE ticks) are never imitated. Early positive credit from a placement that
later kills its own player therefore cannot be imitated. Not a reward; no online
action replacement.
"""
import jax
import jax.numpy as j
from scripts import r4_k4_collector as common
from scripts.cc_sil_updater_v1 import BOMB

SURVIVE = 40


def survived(live):
    """live [T,N] bool -> [T,N]: live at t+1..t+SURVIVE (False when the window leaves the rollout)."""
    T = live.shape[0]
    dead = (~live).astype(j.int32)
    c = j.concatenate([j.zeros_like(dead[:1]), j.cumsum(dead, 0)], 0)  # c[t] = deaths in [0,t)
    t = j.arange(T)
    hi = j.minimum(t + 1 + SURVIVE, T)
    deaths = c[hi] - c[t + 1]
    return (deaths == 0) & ((t + SURVIVE) < T)[:, None]


def objective(coef):
    def ppo_objective(params, batch):
        from scripts.r4_k4_observation import actor_critic_forward
        from jax_bomb.jax_train import hl_gauss_value_loss
        obs, global_state, actions, old_logp, advantage, returns, move_mask, ability_mask, safe = batch[:9]
        move, ability, _, value_logits = actor_critic_forward(params, obs, global_state)
        lm = jax.nn.log_softmax(j.where(move_mask, move, -1e9))
        la = jax.nn.log_softmax(j.where(ability_mask, ability, -1e9))
        rows = j.arange(len(old_logp))
        selected = lm[rows, actions[:, 0]] + la[rows, actions[:, 1]]
        ratio = j.exp(selected - old_logp)
        policy = j.maximum(-advantage * ratio, -advantage * j.clip(ratio, .9, 1.1)).mean()
        value = hl_gauss_value_loss(value_logits, returns)
        entropy = -(j.sum(j.exp(lm) * lm, -1) + j.sum(j.exp(la) * la, -1)).mean()
        use = (actions[:, 1] == BOMB) & ability_mask[:, BOMB].astype(bool) & safe
        weight = j.where(use, jax.nn.relu(advantage), 0.)
        sil = -(jax.lax.stop_gradient(weight) * la[:, BOMB]).sum() / len(old_logp)
        loss = policy + .5 * value - .005 * entropy + coef * sil
        return loss, j.array([policy, value, entropy, (old_logp - selected).mean(), (j.abs(ratio - 1.) > .1).mean()])
    return ppo_objective


def updater(coef):
    import optax
    fn = objective(coef)
    def build(minibatch, optimizer):
        def update(params, adam, batch, key):
            safe = survived(batch[9].astype(bool))
            batch = tuple(v.reshape((-1,) + v.shape[2:]) for v in tuple(batch) + (safe,))
            order = jax.random.permutation(key, len(batch[4])).reshape(-1, minibatch)
            def body(carry, indices):
                p, a = carry; mb = tuple(v[indices] for v in batch)
                advantage, normalization = common.normalize_by_mode(mb[4], mb[8], mb[9])
                mb = mb[:4] + (advantage,) + mb[5:8] + (mb[10],)
                (loss, metrics), gradient = jax.value_and_grad(fn, has_aux=True)(p, mb)
                changes = {}; next_adam = {}
                for branch in ('actor', 'critic'):
                    changes[branch], next_adam[branch] = optimizer.update(gradient[branch], a[branch], p[branch])
                next_params = optax.apply_updates(p, changes)
                values = j.concatenate([loss[None], metrics, optax.global_norm(gradient)[None],
                    optax.global_norm(gradient['actor']['heads']['wb'])[None],
                    normalization['mode_samples'], normalization['mode_mean_abs_adv']])
                return (next_params, next_adam), values
            return jax.lax.scan(body, (params, adam), order)
        return jax.jit(update)
    return build
