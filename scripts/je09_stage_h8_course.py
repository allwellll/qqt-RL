"""H8 native weak-courier FSM and target allocation."""
import jax
import jax.numpy as jnp
from scripts.je09_return_expert import action_set


TARGET_PROBABILITIES = jnp.array([0.0, 0.70, 0.05, 0.05, 0.20], jnp.float32)


def level(generation):
    return jnp.minimum(3, (jnp.asarray(generation) - 1) // 8)


def weak_fraction(generation):
    values = jnp.array([.5, .35, .2, .1], jnp.float32)
    return values[level(generation)]


def targeted_roles(scene, role_uniform, target_uniform, generation):
    weak = weak_fraction(generation)
    selected = (scene == 8) & (role_uniform < weak)
    target = jnp.sum(target_uniform[..., None] >= jnp.cumsum(TARGET_PROBABILITIES), axis=-1)
    target = jnp.clip(target, 1, 4).astype(jnp.int32)
    courier = (target == 1) | (target == 3)
    role = jnp.where(selected, jnp.where(courier, 1, 2), 0).astype(jnp.int32)
    return role, jnp.where(selected, target, 0).astype(jnp.int32)


def actions(states, aux, opponent, proposal, legal, ability, key, role, generation, danger):
    """Courier state is native empty/carrying state; no state is injected."""
    from jax_bomb import je07_bun_env as env
    from scripts import je09_bun_scenes as scenes

    n = opponent.size
    rows = jnp.arange(n)
    tier = level(generation)
    carrying = states.bun_carried[rows, opponent] >= 0
    goal_team = jnp.where(role == 1, jnp.where(carrying, opponent, 1-opponent), opponent)
    variant = scenes.OPEN_SCENE[states.scene].astype(jnp.int32)
    distance = scenes.DISTANCE[variant, goal_team]
    target, reachable = action_set(states.pos[rows, opponent], distance, legal,
        jnp.zeros_like(danger), states.wall | states.brick | states.core.pushable,
        states.fuse)
    directed = jnp.argmax(target, -1)
    cell = jnp.clip(states.pos[rows, opponent].astype(jnp.int32),
                    jnp.array([0, 0]), jnp.array([12, 14]))
    neighbors = jnp.clip(cell[:, None] + jnp.array(
        [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]]),
        jnp.array([0, 0]), jnp.array([12, 14]))
    static = distance[rows[:, None], neighbors[..., 0], neighbors[..., 1]]
    fallback = jnp.argmin(jnp.where(legal, static, 1e8), -1)
    move = jnp.where(reachable, directed, fallback)
    base_mask = jnp.stack([env._base_mask(0), env._base_mask(1)])
    at_base = base_mask[opponent, cell[:, 0], cell[:, 1]]
    random = jax.random.uniform(key, (n, 3))
    carry_duty = jnp.array([.2, .3, .4, .55], jnp.float32)[tier]
    guard_duty = jnp.array([0., .1, .25, .5], jnp.float32)[tier]
    duty = jnp.where(role == 1, jnp.where(carrying, carry_duty, 1.),
                     jnp.where(at_base, guard_duty, 1.))
    move = jnp.where(random[:, 0] < duty, move, 4)
    hazard = jnp.any(danger > 0, (-2, -1)) | jnp.any(states.fuse > 0, (-2, -1))
    escape = hazard & (random[:, 1] < jnp.array([0., .25, .5, 1.])[tier])
    move = jnp.where(escape, proposal[:, 0], move)
    bombcount = ((states.core.owner == opponent[:, None, None]) & (states.fuse > 0)).sum((-2, -1))
    capacity = jnp.array([1, 1, 2, 2])[tier]
    period = jnp.array([64, 32, 16, 8])[tier]
    use = (proposal[:, 1] == 1) & ability[:, 1] & (bombcount < capacity) & ((states.t + rows) % period == 0)
    result = jnp.stack([jnp.where(legal[rows, move], move, 4), use.astype(jnp.int32)], -1)
    return jnp.where((role > 0)[:, None], result, proposal)
