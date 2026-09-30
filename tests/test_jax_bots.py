import os

os.environ.setdefault("JAXBOMB_RULE", "bun")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_bomb import bun_env
from jax_bomb import bun_jax_bots as B
from jax_bomb import levels

GAMES = 64


@pytest.fixture(autouse=True)
def danger_arena():
    bun_env.prepare()
    bun_env.configure_training("danger_arena=1", 1)
    yield
    bun_env.configure_training("full=1", bun_env.MAX_HP)
    levels.clear()


def _run_vs_idle(tier, ticks=300):
    kinds = jnp.full((GAMES,), B.TIER_NAMES.index(tier), jnp.int32)
    seats = jnp.ones((GAMES,), jnp.int32)
    idle = jnp.tile(jnp.asarray([[4, 0]], jnp.int32), (GAMES, 1))

    def tick(carry, _):
        states, key = carry
        key, k_bot, k_step = jax.random.split(key, 3)
        mm, bm = jax.vmap(bun_env.legal_mask)(states)
        acts = B.rule_bot_actions(states, seats, mm[:, 1], bm[:, 1], k_bot, kinds)
        legal = mm[jnp.arange(GAMES), 1, acts[:, 0]] & bm[jnp.arange(GAMES), 1, acts[:, 1]]
        new_states, _, info = jax.vmap(
            lambda s, a, k: bun_env.step(s, a, k, auto_reset=False, return_info=True)
        )(states, jnp.stack([idle, acts], axis=1), jax.random.split(k_step, GAMES))
        f = lambda n: info[n].astype(jnp.float32)
        ev = jnp.stack([
            jnp.maximum(f("surviving_causal_kill")[:, 1], f("surviving_physical_kill")[:, 1]),
            f("own_bomb_defeat")[:, 1], f("bomb_placed")[:, 1],
            (~legal).astype(jnp.float32)], axis=-1).sum(0)
        return (new_states, key), ev

    @jax.jit
    def run(key):
        key, ik = jax.random.split(key)
        _, ev = jax.lax.scan(tick, (bun_env.init_batch(ik, GAMES), key), None,
                             length=ticks)
        return ev.sum(0) / GAMES

    kill, self_kill, bombs, illegal = np.asarray(run(jax.random.PRNGKey(7)))
    return dict(kill=kill, self=self_kill, bombs=bombs, illegal=illegal)


def test_tier_table_is_consistent():
    table = B.tier_table()
    assert B.TIER_NAMES == tuple(B.TIERS)
    for field in B.TierParams._fields:
        assert np.asarray(getattr(table, field)).shape == (len(B.TIER_NAMES),)
    weights = B.parse_tier_names("dodge=1,hunter=3", B.TIER_NAMES)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[B.TIER_NAMES.index("hunter")] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        B.parse_tier_names("nope=1", B.TIER_NAMES)


@pytest.mark.parametrize("tier", ["dodge_easy", "dodge"])
def test_dodge_tiers_never_bomb(tier):
    out = _run_vs_idle(tier, ticks=200)
    assert out["illegal"] == 0
    assert out["bombs"] == 0
    assert out["self"] == 0


@pytest.mark.parametrize("tier", ["bomber_easy", "hunter", "hunter_hard"])
def test_attacking_tiers_kill_idle_without_suiciding(tier):
    out = _run_vs_idle(tier)
    assert out["illegal"] == 0
    assert out["kill"] >= 0.8
    assert out["self"] <= 0.1


def test_collect_rollout_with_jax_bot_pool():
    from jax_bomb import jax_train as T
    if not T.IS_BUN:
        pytest.skip("jax_train imported under non-bun rule")
    n, steps = 32, 8
    states = bun_env.init_batch(jax.random.PRNGKey(0), n)
    params = T.init_net(jax.random.PRNGKey(1), "mlp", bun_env.N_OBS_CH,
                        bun_env.H, bun_env.W, hidden=32)
    weights = B.parse_tier_names("dodge=1,hunter=1", T.JAX_BOT_NAMES)
    out = T.collect_rollout(params, "mlp", states, jax.random.PRNGKey(2), steps,
                            flee_bot_ratio=0.5, jax_bot_pool=jnp.asarray(weights),
                            jax_bot_enabled=np.asarray(weights) > 0,
                            return_bot_stats=True)
    stats = np.asarray(out[4])
    assert stats.shape == (len(T.JAX_BOT_NAMES), 4)
    assert stats[:, 0].sum() == pytest.approx(n // 2 * steps)
    enabled = np.asarray(weights) > 0
    assert stats[~enabled].sum() == 0


def test_curriculum_adapts_toward_informative_tiers():
    from jax_bomb import jax_train as T
    if not T.IS_BUN:
        pytest.skip("jax_train imported under non-bun rule")
    base = np.zeros(len(T.JAX_BOT_NAMES))
    base[:3] = 1.0
    cur = T.JaxBotCurriculum(base, adaptive=True, ema=0.0)
    stats = np.zeros((len(base), 4))
    stats[:3, 0] = 300
    stats[0, 1:3] = [10, 0]    # mastered
    stats[1, 1:3] = [5, 5]     # balanced
    stats[2, 1:3] = [0, 10]    # hopeless
    cur.update(stats)
    w = cur.weights()
    assert w[1] > w[0] and w[1] > w[2]
    assert w[0] > 0 and w[2] > 0
    assert w[3:].sum() == 0
