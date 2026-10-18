"""Current-state native combat imitation, training only; never override actions."""
import jax
import jax.numpy as jnp
from jax_bomb import bun_jax_bots as bots

COEFFICIENT = .05
TEMPERATURE = 1.


def combat_gate(states, seat, terminal):
    rows = jnp.arange(len(seat)); foe = 1-seat
    return jax.lax.stop_gradient(
        states.alive[rows, seat] & states.alive[rows, foe]
        & (states.trapped[rows, seat] == 0) & (states.trapped[rows, foe] == 0)
        & (states.bun_carried[rows, seat] < 0) & ~terminal)


def labels(states, seat, mm, bm, terminal, key):
    """Use existing native hard combat bot and its escape BFS safety certificate.

    No bun/course objective, future state, or teacher action enters env.step.
    Unavoidable/no-safe-action fallback rows have no added supervision.
    """
    table = bots.tier_table()
    params = jax.tree.map(lambda x: jnp.broadcast_to(x[4], (len(seat),)), table)
    actions, debug = jax.vmap(lambda s, p, m, b, k, t:
                             bots.bot_action(s, p, m, b, k, t, debug=True))(
        states, seat, mm, bm, jax.random.split(key, len(seat)), params)
    rows = jnp.arange(len(seat)); move, ability = actions.T
    # Variant1 includes our proposed bomb; variant3 also tests an enemy bomb.
    variant = jnp.where(ability == 1, 1, 0)
    safe = debug['surv'][rows, variant, move]
    active = combat_gate(states, seat, terminal) & safe & mm[rows, move] & bm[rows, ability]
    targets = jnp.concatenate([jax.nn.one_hot(move, 5), jax.nn.one_hot(ability, 3),
                               active[:, None]], -1)
    return jax.lax.stop_gradient(targets)


def supervised_loss(move, ability, mm, bm, targets, coefficient=COEFFICIENT,
                    temperature=TEMPERATURE):
    mt, bt = targets[:, :5], targets[:, 5:8]
    legal = (jnp.sum(mt * mm, -1) == jnp.sum(mt, -1)) & (jnp.sum(bt * bm, -1) == jnp.sum(bt, -1))
    active = (targets[:, 8] > 0) & legal & (mt.sum(-1) > 0) & (bt.sum(-1) > 0)
    ml = jax.nn.log_softmax(jnp.where(mm, move/temperature, -1e9))
    bl = jax.nn.log_softmax(jnp.where(bm, ability/temperature, -1e9))
    ce = -(mt * ml).sum(-1)/jnp.maximum(mt.sum(-1), 1) - (bt * bl).sum(-1)/jnp.maximum(bt.sum(-1), 1)
    return coefficient * jnp.where(active, ce, 0).sum()/jnp.maximum(active.sum(), 1)
