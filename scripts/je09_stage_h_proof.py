"""Attribute the added event reward through actual production GAE/PPO data."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scripts.population_run_state import atomic_json


def production_proof(replicated, batch, run):
    from jax_bomb import jax_train as train
    from scripts.je09_stage_h_items import METRICS
    from scripts.je09_stage_g_proof import production_proof as combat_proof
    combat_proof(replicated, batch, run)
    raw = np.asarray(batch[-1]); extra = raw[..., 34+METRICS.index('item_net')]
    metrics = {k: float(raw[..., 34+i].sum()) for i, k in enumerate(METRICS)}
    # Future rewards legitimately influence earlier actions through GAE. The
    # ledger, not an artificial per-row gradient mask, enforces event validity.
    delta = np.zeros_like(extra); carry = np.zeros_like(extra[:, 0])
    # collect context contains factual episode terminal below the item fields.
    terminal = raw[..., 47] > 0
    for tick in range(extra.shape[1]-1, -1, -1):
        carry = extra[:, tick]+.995*.95*(~terminal[:, tick])*carry
        delta[:, tick] = carry
    active = np.flatnonzero(np.abs(delta.reshape(-1)) > 1e-8)
    report = dict(collect_gae_batch_consumed_by_ppo=True, event_metrics=metrics,
                  actual_nonzero_event_rows=int(np.count_nonzero(extra)),
                  action_override=False, source='native current transitions; no reward injection',
                  gae_credit_note='delayed outcome may credit earlier placement/approach actions')
    if not len(active):
        report.update(passed=True, gradient_observed=False,
                      limitation='No paid item outcome in this actual batch; not nonzero reward-gradient evidence')
        atomic_json(run/'validation/ITEM_PRODUCTION_OBSERVATION.json', report)
        return
    flatten = lambda x: np.asarray(x).reshape((-1,)+x.shape[3:])
    obs, gv, acts, mm, bm = map(flatten, (batch[0], batch[1], batch[2], batch[6][0], batch[6][1]))
    actual_adv = np.asarray(batch[4]).reshape(-1); target = np.asarray(batch[5]).reshape(-1)
    removed = actual_adv-delta.reshape(-1)
    normalized = lambda x: (x-x.mean())/(x.std()+1e-8)
    adv_difference = normalized(actual_adv)-normalized(removed)
    ix = active[:32]; params = jax.tree.map(lambda x: x[0], replicated)
    o, g, a, m, b = map(jnp.asarray, (obs[ix], gv[ix], acts[ix], mm[ix], bm[ix]))
    diff = jnp.asarray(adv_difference[ix]); full = jnp.asarray(target[ix]); without = full-jnp.asarray(delta.reshape(-1)[ix])
    def effect(q):
        move, ability, _, vl = train.net_forward(q, 'transformer', o, g)
        lm = jax.nn.log_softmax(jnp.where(m, move, -1e9)); lb = jax.nn.log_softmax(jnp.where(b, ability, -1e9))
        lp = lm[jnp.arange(len(ix)), a[:, 0]]+lb[jnp.arange(len(ix)), a[:, 1]]
        value_delta = train.hl_gauss_value_loss(vl, full)-train.hl_gauss_value_loss(vl, without)
        return -(diff*lp).mean()+.5*value_delta
    grads = jax.jit(jax.grad(effect))(params)
    norms = {k: float(optax.global_norm(grads[k])) for k in ['actor', 'critic']}
    assert np.isfinite(list(norms.values())).all() and norms['actor'] > 0
    report.update(passed=True, gradient_observed=True, event_reward_gae_ppo_gradient=norms,
                  critic_note='reward changes value targets; nonzero Critic gradient is expected, unlike Actor-only auxiliary KL')
    atomic_json(run/'validation/ITEM_PRODUCTION_GRADIENT.json', report)
