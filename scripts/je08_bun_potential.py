"""Exact weighted grid distance caches and signed, capped phase ledgers."""
import heapq

import jax
import jax.numpy as jnp
import numpy as np

INF = 32767
DROP_COEFFICIENT = .05
CARRY_COEFFICIENT = .0625
DROP_CAP = 2.
CARRY_CAP = 4.


def cpu_shortest(wall, brick, goals):
    wall = np.asarray(wall, bool)
    brick = np.asarray(brick, bool)
    goals = np.asarray(goals, bool)
    h, w = wall.shape
    dist = np.full((h, w), INF, np.int32)
    queue = []
    for r, c in np.argwhere(goals & ~wall):
        dist[r, c] = 0
        heapq.heappush(queue, (0, int(r), int(c)))
    while queue:
        d, r, c = heapq.heappop(queue)
        if d != dist[r, c]:
            continue
        cost = 1 + 3 * int(brick[r, c])
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not wall[rr, cc] and d + cost < dist[rr, cc]:
                dist[rr, cc] = d + cost
                heapq.heappush(queue, (d + cost, rr, cc))
    return dist


def shortest(wall, brick, goals):
    """Reverse weighted Bellman relaxation, exact after at most V-1 edges.

    Goal entry costs 1+3*brick, matching precompute_regions. This is a
    planning distance through breakable brick, not permission to walk through
    it. Walls remain unreachable. Called only on a dirty cached topology.
    Leading goal axes are allowed (four maps per environment).
    """
    h, w = wall.shape[-2:]
    start = jnp.where(goals & ~wall, 0, INF).astype(jnp.int32)

    def condition(carry):
        return (carry[0] < h * w) & carry[2]

    def sweep(carry):
        i, distance, _ = carry
        entry = jnp.minimum(INF, distance + 1 + 3 * brick.astype(jnp.int32))
        padding = [(0, 0)] * (distance.ndim - 2) + [(1, 1), (1, 1)]
        padded = jnp.pad(entry, padding, constant_values=INF)
        near = jnp.minimum(
            jnp.minimum(padded[..., 2:, 1:-1], padded[..., :-2, 1:-1]),
            jnp.minimum(padded[..., 1:-1, 2:], padded[..., 1:-1, :-2]),
        )
        new = jnp.where(wall, INF, jnp.minimum(distance, near))
        return i + 1, new, jnp.any(new != distance)

    return jax.lax.while_loop(
        condition, sweep, (jnp.int32(0), start, jnp.array(True))
    )[1]


def phase_state():
    return dict(
        active=jnp.array(False), identity=jnp.int32(-1), retired=jnp.int32(-1),
        gross=jnp.float32(0), balance=jnp.float32(0), pending=jnp.array(False),
    )


def phase_reward(e, s, coefficient, cap, allow_restart=False):
    s = jax.tree.map(jnp.asarray, s)
    changed = e['identity'] != s['identity']
    eligible = e['eligible'] & ~e['bad'] & ((e['identity'] != s['retired']) | allow_restart)
    start = eligible & (~s['active'] | changed)
    end = s['active'] & (~eligible | changed)
    gross = jnp.where(changed & start, 0., s['gross'])
    measurable = s['active'] & eligible & ~changed & ~e['topology_changed'] & ~e['boundary']
    measurable &= (e['before'] < INF) & (e['after'] < INF) & (e['before'] >= 0) & (e['after'] >= 0)
    raw = jnp.where(measurable, coefficient * (e['before'] - e['after']), 0.)
    positive = jnp.minimum(jnp.maximum(raw, 0), jnp.maximum(0, cap - gross))
    negative = jnp.minimum(raw, 0)
    delta = positive + negative
    balance = s['balance'] + delta
    clawback = jnp.where(e['bad'], -jnp.maximum(balance, 0), 0.)
    pending = (s['pending'] | e['capture']) & ~e['bad']
    settle = pending & e['settled']
    retired = jnp.where(end | e['bad'], s['identity'], s['retired'])
    new = dict(
        active=eligible,
        identity=jnp.where(start, e['identity'], s['identity']),
        retired=retired,
        gross=gross + positive,
        balance=jnp.where(e['bad'] | settle, 0., balance),
        pending=pending & ~settle,
    )
    metrics = dict(
        positive=positive, negative=negative, net=delta, clawback=clawback,
        stationary=measurable & (e['before'] == e['after']), phase_start=start,
        phase_end=end, cap_hit=measurable & (raw > positive),
        topology_rebase=e['topology_changed'] & eligible,
    )
    return delta + clawback, new, metrics


def state(n):
    return {
        name: jax.tree.map(lambda x: jnp.broadcast_to(x, (n, 2) + x.shape), phase_state())
        for name in ['drop', 'carry']
    }


def routing_state(n, h=13, w=15):
    return dict(
        ready=jnp.zeros(n, bool), wall=jnp.zeros((n, h, w), bool),
        brick=jnp.zeros((n, h, w), bool), target=jnp.full((n, 2, 2), -1, jnp.int32),
        nonce=jnp.zeros((n, 2), jnp.int32), distance=jnp.full((n, 4, h, w), INF, jnp.int32),
    )


def refresh(states, next_states, info, routing, drop, goals):
    """Cache exact distances; no search on a clean rollout tick."""
    n = states.pos.shape[0]
    h, w = states.wall.shape[-2:]
    cells = routing['target']
    nonce = routing['nonce']
    for origin in range(2):
        happened = info['drop'] & (info['drop_origin'] == origin)
        who = jnp.argmax(happened, -1)
        row = jnp.arange(n)
        cells = cells.at[:, origin].set(
            jnp.where(happened.any(-1)[:, None], info['drop_cells'][row, who], cells[:, origin])
        )
        nonce = nonce.at[:, origin].set(drop['nonce'][:, origin])
    dirty = (
        ~routing['ready']
        | jnp.any(next_states.wall != routing['wall'], (-2, -1))
        | jnp.any(next_states.brick != routing['brick'], (-2, -1))
        | jnp.any(cells != routing['target'], (-2, -1))
    )
    target = jnp.clip(cells[:, ::-1], jnp.array([0, 0]), jnp.array([h - 1, w - 1]))
    drop_goals = jnp.zeros((n, 2, h, w), bool).at[
        jnp.arange(n)[:, None], jnp.arange(2)[None], target[..., 0], target[..., 1]
    ].set(jnp.all(cells[:, ::-1] >= 0, -1))
    allgoals = jnp.concatenate([drop_goals, jnp.broadcast_to(goals, (n, 2, h, w))], 1)

    def rebuild(_):
        result = shortest(next_states.wall[:, None], next_states.brick[:, None], allgoals)
        return jnp.where(dirty[:, None, None, None], result, routing['distance'])

    distance = jax.lax.cond(jnp.any(dirty), rebuild, lambda _: routing['distance'], operand=None)
    return dict(
        ready=jnp.ones(n, bool), wall=next_states.wall, brick=next_states.brick,
        target=cells, nonce=nonce, distance=distance,
    ), dirty
