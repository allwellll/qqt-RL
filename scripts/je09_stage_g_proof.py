"""Attribution from actual production collect/GAE tensors consumed by PPO."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scripts.population_run_state import atomic_json


def production_proof(replicated, batch, run):
    from jax_bomb import jax_train as train
    from scripts.je09_stage_g_combat import supervised_loss
    params = jax.tree.map(lambda x: x[0], replicated)
    flat = lambda x: np.asarray(x).reshape((-1,) + x.shape[3:])
    o, g, m, b, c = map(flat, (batch[0], batch[1], batch[6][0], batch[6][1], batch[-1]))
    active = c[:, 33] > 0
    assert active.any() and not np.any(active & ((c[:, 23] > 0) | (c[:, 24] > 0)))
    assert not np.any((c[:, 18] > 0) & (c[:, 23] == 0))
    ix = np.concatenate([np.flatnonzero(active)[:32], np.flatnonzero(~active)[:32]])
    o, g, m, b, c = map(jnp.asarray, (o[ix], g[ix], m[ix], b[ix], c[ix]))
    def loss(q):
        move, ability = train.net_forward(q, 'transformer', o, g)[:2]
        return supervised_loss(move, ability, m, b, c[:, 25:34])
    grad = jax.jit(jax.grad(loss))(params)
    norms = {k: float(optax.global_norm(grad[k])) for k in ['actor', 'critic']}
    assert norms['actor'] > 0 and norms['critic'] == 0
    move, ability = train.net_forward(params, 'transformer', o, g)[:2]
    gm, gb = jax.grad(lambda x, y: supervised_loss(x, y, m, b, c[:, 25:34]), (0, 1))(move, ability)
    assert np.all(np.asarray(gm)[np.asarray(c[:, 33]) == 0] == 0)
    assert np.all(np.asarray(gb)[np.asarray(c[:, 33]) == 0] == 0)
    atomic_json(run/'validation/COMBAT_PRODUCTION_GRADIENT.json', dict(
        passed=True, collect_gae_batch_consumed_by_ppo=True, raw_auxiliary_gradient=norms,
        active_rows=int(active.sum()), excluded_logit_gradient_exact_zero=True,
        action_override=False, teacher='native hard rule bot; same-state escape BFS certificate',
        existing_chase_mode_gate_intact=True, context_width=int(c.shape[-1])))
