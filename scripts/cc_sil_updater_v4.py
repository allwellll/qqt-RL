"""v1/v3 placement self-imitation plus a KL anchor to a frozen healthy actor (v4).

loss = PPO + .5 value - .005 entropy + sil_coef * SIL + kl_coef * KL(anchor || pi)
KL covers move and ability distributions on live rows. The anchor is the G_frozen
u018262 actor: the fixed 88-game evaluation found every later point of the same
recipe (original lineage and CC arms) drifting away from it. With trap_filter the
SIL weight uses the v3 live+untrapped 40-tick window, otherwise v1 (no filter).
"""
import jax
import jax.numpy as j
from scripts import r4_k4_collector as common
from scripts.cc_sil_updater_v1 import BOMB
from scripts.cc_sil_updater_v2 import survived


def objective(sil_coef, kl_coef):
    def ppo_objective(params, batch, anchor):
        from scripts.r4_k4_observation import actor_critic_forward, forward
        from jax_bomb.jax_train import hl_gauss_value_loss
        obs, global_state, actions, old_logp, advantage, returns, move_mask, ability_mask, safe, live = batch[:10]
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
        am, aa, _, _ = forward(anchor, obs, global_state)
        sm = jax.lax.stop_gradient(jax.nn.log_softmax(j.where(move_mask, am, -1e9)))
        sa = jax.lax.stop_gradient(jax.nn.log_softmax(j.where(ability_mask, aa, -1e9)))
        kl = (j.exp(sm) * (sm - lm)).sum(-1) + (j.exp(sa) * (sa - la)).sum(-1)
        kl = j.where(live, kl, 0.).sum() / j.maximum(live.sum(), 1)
        loss = policy + .5 * value - .005 * entropy + sil_coef * sil + kl_coef * kl
        return loss, j.array([policy, value, entropy, (old_logp - selected).mean(), (j.abs(ratio - 1.) > .1).mean()])
    return ppo_objective


def updater(sil_coef, kl_coef, trap_filter, anchor):
    """Returns build(minibatch, optimizer) -> update(params, adam, batch, key); anchor passed as a jit argument."""
    import optax
    fn = objective(sil_coef, kl_coef)
    def build(minibatch, optimizer):
        def update(params, adam, batch, key, anchor):
            live = batch[9].astype(bool)
            if trap_filter:
                trapped = ~batch[6][..., :4].astype(bool).any(-1) & ~batch[7][..., BOMB].astype(bool)
                safe = survived(live & ~trapped)
            else:
                safe = j.ones_like(live)
            batch = tuple(v.reshape((-1,) + v.shape[2:]) for v in tuple(batch) + (safe,))
            order = jax.random.permutation(key, len(batch[4])).reshape(-1, minibatch)
            def body(carry, indices):
                p, a = carry; mb = tuple(v[indices] for v in batch)
                advantage, normalization = common.normalize_by_mode(mb[4], mb[8], mb[9])
                mb = mb[:4] + (advantage,) + mb[5:8] + (mb[10], mb[9].astype(bool))
                (loss, metrics), gradient = jax.value_and_grad(fn, has_aux=True)(p, mb, anchor)
                changes = {}; next_adam = {}
                for branch in ('actor', 'critic'):
                    changes[branch], next_adam[branch] = optimizer.update(gradient[branch], a[branch], p[branch])
                next_params = optax.apply_updates(p, changes)
                values = j.concatenate([loss[None], metrics, optax.global_norm(gradient)[None],
                    optax.global_norm(gradient['actor']['heads']['wb'])[None],
                    normalization['mode_samples'], normalization['mode_mean_abs_adv']])
                return (next_params, next_adam), values
            return jax.lax.scan(body, (params, adam), order)
        compiled = jax.jit(update)
        return lambda params, adam, batch, key: compiled(params, adam, batch, key, anchor)
    return build
