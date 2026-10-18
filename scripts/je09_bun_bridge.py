"""JE09 real drop-to-capture bridge contract and checkpointed reward ledger."""
from dataclasses import dataclass


BRIDGE_MANIFEST = {
    "schema": "je09.real-drop-bridge/v1",
    "assist": False,
    "start": "empty_learner_real_enemy_carrier",
    "events": ["attributed_drop", "first_pickup", "pickup_return", "progress25",
                "progress50", "home_half", "capture"],
    "seeds": {"train": 2026101509, "diagnostic": 2026101519, "evaluation": 2026101529},
    "holdout_seed": 2026101539,
    "games": 16,
    "stage": "bridge",
}

EVENT_REWARD = {
    "attributed_drop": 0.0,
    "first_pickup": 8.0,
    "pickup_return": 1.0,
    "progress25": 1.0,
    "progress50": 1.5,
    "home_half": 2.0,
    "capture": 24.0,
}
MILESTONES = tuple(BRIDGE_MANIFEST["events"])


def validate_bridge_start(start):
    if start.get("learner_carried") != -1:
        raise ValueError("bridge learner must start empty-handed")
    if start.get("enemy_carried") != 1:
        raise ValueError("bridge enemy must start with its own token")
    if not 2 <= int(start.get("drop_distance", 0)) <= 8:
        raise ValueError("bridge drop must be 2..8 topology steps away")
    if not start.get("legal_return_route") or start.get("enemy_can_open_route"):
        raise ValueError("bridge route must be reachable without enemy route opening")


@dataclass
class BridgeLedger:
    paid: set
    balance: float = 0.0
    capture: bool = False

    def apply(self, event):
        if event not in EVENT_REWARD or event in self.paid:
            return 0.0
        if event != "attributed_drop" and "attributed_drop" not in self.paid:
            return 0.0
        value = EVENT_REWARD[event]
        self.paid.add(event)
        self.balance += value
        if event == "capture":
            self.capture = True
        return value

    def fail(self, reason):
        if reason not in {"drop", "death", "self", "trade", "timeout"}:
            raise ValueError("unknown bridge failure")
        clawback = -self.balance
        self.balance = 0.0
        self.capture = False
        return clawback


def bridge_reward_ledger():
    return BridgeLedger(set())


def bridge_funnel(rows):
    names = ["eligible_drop", "toward_drop", "arrival", "first_pickup", "pickup_return",
             "progress25", "progress50", "home_half", "capture", "death", "self", "trade"]
    result = {name: 0 for name in names}
    for row in rows:
        for name in names:
            if name == "toward_drop":
                result[name] += int(row.get(name, 0) > 0)
            else:
                result[name] += int(bool(row.get(name, False)))
    result["eligible_drop"] = len(rows)
    return result


def jax_bridge_state(n):
    """Checkpointable per-environment phase ledger used by the JE09 worker."""
    import jax.numpy as jnp
    return dict(
        active=jnp.zeros((n, 2), bool),
        phase=jnp.zeros((n, 2), jnp.int8),
        balance=jnp.zeros((n, 2), jnp.float32),
        paid=jnp.zeros((n, 2, 7), bool),
        pending=jnp.zeros((n, 2), bool),
        start_distance=jnp.zeros((n, 2), jnp.float32),
        start_home=jnp.zeros((n, 2), bool),
        ticks=jnp.zeros((n, 2), jnp.int32),
    )


def jax_bridge_reward(info, state, window=None, scale=1.):
    """One-shot bridge milestones consumed by production PPO transitions."""
    import jax.numpy as jnp
    # Native ledgers already own pickup8 and settled capture24. This ledger
    # adds only post-pickup switching/milestones to a real attributed chain.
    dropped = info['drop'] & ~state['paid'][..., 0]
    active = state['active'] | dropped
    picked = active & info['pickup_loose'] & ~state['paid'][..., 1]
    distance = info['after_distance'].astype(jnp.float32)
    start = jnp.where(picked, distance, state['start_distance'])
    carrying = active & state['paid'][..., 1] & info['carried']
    measurable = carrying & ~info['topology_changed'] & (distance < 32767) & (start > 0)
    ticks = jnp.where(picked, 0, state['ticks'] + carrying.astype(jnp.int32))
    switch = measurable & (ticks <= 120) & (distance < start)
    start_home = jnp.where(picked, info["home_half"], state["start_home"])
    within = jnp.ones_like(measurable) if window is None else ticks <= window
    home = measurable & info["home_half"]
    if window is not None:
        home &= ~start_home & (distance < start)
    events = jnp.stack([
        dropped, picked, switch & within, measurable & within & (distance <= .75 * start),
        measurable & within & (distance <= .5 * start), home & within,
        active & state['paid'][..., 1] & info['capture'],
    ], -1)
    fresh = events & ~state['paid']
    values = jnp.asarray([0., 0., 1., 1., 1.5, 2., 0.], jnp.float32)
    reward = scale * (fresh * values).sum(-1)
    bad = info['death'] | info['self'] | info['trade'] | info['timeout'] | info['lost']
    pending = (state['pending'] | events[..., 6]) & ~bad
    balance = state['balance'] + reward
    claw = jnp.where(bad, -balance, 0.)
    settled = pending & info['settled']
    new = dict(state, active=active & ~bad & ~settled, paid=state['paid'] | fresh,
               start_distance=start, start_home=start_home, ticks=ticks, pending=pending & ~settled,
               balance=jnp.where(bad | settled, 0., balance))
    return reward + claw, new, dict(bridge_reward=reward, bridge_clawback=claw, bridge_events=fresh)
