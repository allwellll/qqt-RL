"""Native carry-item event accounting; no per-tick item income."""
import jax
import jax.numpy as jnp
import numpy as np


def event():
    return dict(carry=jnp.array([[True, False]]), acquired=jnp.array([[False, False]]),
                bad=jnp.zeros((1, 2), bool), capture=jnp.zeros((1, 2), bool),
                glue_hit=jnp.zeros((1, 2), bool), banana_start=jnp.zeros((1, 2), bool),
                sliding=jnp.zeros((1, 2), bool), distance=jnp.array([[12, 12]]),
                safe=jnp.ones((1, 2), bool), topology_changed=jnp.zeros((1, 2), bool))


def test_no_initialized_carry_credit_then_real_acquisition_and_one_edge():
    from scripts.je09_stage_h_items import ledger_state, settle
    s = ledger_state(1); e = event(); e['glue_hit'] = e['glue_hit'].at[0, 0].set(True)
    r, s, _ = settle(e, s); np.testing.assert_array_equal(r, 0)
    e['acquired'] = e['acquired'].at[0, 0].set(True)
    r, s, _ = settle(e, s); np.testing.assert_array_equal(r, 0)
    e['acquired'] = jnp.zeros((1, 2), bool)
    r, s, _ = settle(e, s); assert float(r[0, 0]) == .5
    e['glue_hit'] = jnp.zeros((1, 2), bool)
    r, s, _ = settle(e, s); np.testing.assert_array_equal(r, 0)


def test_banana_full_slide_net_distance_and_loss_exclusions():
    from scripts.je09_stage_h_items import ledger_state, settle
    e = event(); e['acquired'] = e['acquired'].at[0, 0].set(True)
    _, s, _ = settle(e, ledger_state(1)); e['acquired'] = jnp.zeros((1, 2), bool)
    e['banana_start'] = e['banana_start'].at[0, 0].set(True)
    e['sliding'] = e['sliding'].at[0, 0].set(True)
    r, s, _ = settle(e, s); np.testing.assert_array_equal(r, 0)
    e['banana_start'] = jnp.zeros((1, 2), bool); e['distance'] = e['distance'].at[0, 0].set(8)
    r, s, _ = settle(e, s); np.testing.assert_array_equal(r, 0)
    e['sliding'] = jnp.zeros((1, 2), bool)
    r, ended, m = settle(e, s); assert np.isclose(float(r[0, 0]), .4)
    assert float(m['banana_before'][0, 0]) == 12 and float(m['banana_after'][0, 0]) == 8
    r, _, _ = settle(e, ended); np.testing.assert_array_equal(r, 0)
    for key in ['bad', 'topology_changed']:
        bad = dict(e); bad[key] = jnp.ones((1, 2), bool)
        r, _, _ = settle(bad, s); assert not bool((r > 0).any())
    bad = dict(e, safe=jnp.zeros((1, 2), bool))
    r, _, _ = settle(bad, s); assert not bool((r > 0).any())
    far = dict(e, distance=jnp.full((1, 2), 15))
    r, _, _ = settle(far, s); np.testing.assert_array_equal(r, 0)


def test_cap_clawback_no_refund_and_capture_settlement():
    from scripts.je09_stage_h_items import ledger_state, settle
    s = ledger_state(1); s['custody'] = s['custody'].at[0, 0].set(True)
    e = event(); e['glue_hit'] = e['glue_hit'].at[0, 0].set(True)
    total = 0
    for _ in range(8):
        r, s, _ = settle(e, s); total += float(r[0, 0])
    assert total == 2
    e['bad'] = e['bad'].at[0, 0].set(True)
    r, s, _ = settle(e, s); assert float(r[0, 0]) == -2
    assert float(s['gross'][0, 0]) == 2
    e = event(); e['capture'] = e['capture'].at[0, 0].set(True)
    _, s, _ = settle(e, s); assert float(s['balance'][0, 0]) == 0


def test_glue_identity_requires_recorded_own_placement_and_free_live_enemy():
    from scripts.je09_stage_h_items import attributable_glue
    a = [jnp.array([x]) for x in [True, True, True, True, True, True]]
    assert bool(attributable_glue(*a)[0])
    for i in range(len(a)):
        bad = list(a); bad[i] = jnp.array([False])
        assert not bool(attributable_glue(*bad)[0])


def test_native_real_pickup_placement_and_enemy_glue_trigger():
    from jax_bomb import je07_bun_env as env
    from scripts import je09_stage_h_items as item
    env.prepare()
    key = jax.random.PRNGKey(2703000000)
    s = env.init_batch(key, 1)
    core = s.core._replace(pos=jnp.array([[[5.5, 5.5], [5.5, 7.5]]]),
                           wall=jnp.zeros_like(s.wall), brick=jnp.zeros_like(s.brick),
                           alive=jnp.ones_like(s.alive), invuln=jnp.zeros_like(s.core.invuln))
    s = s._replace(core=core, trapped=jnp.zeros_like(s.trapped),
                   bun_carried=jnp.full_like(s.bun_carried, -1),
                   bun_loose=jnp.zeros_like(s.bun_loose).at[0, 5, 5, 1].set(1),
                   item_slots=jnp.zeros_like(s.item_slots).at[0, 0, 0].set(2),
                   item_counts=jnp.zeros_like(s.item_counts).at[0, 0, 0].set(1))
    goals = jnp.zeros((2, env.H, env.W), bool).at[0, 5, 0].set(True).at[1, 5, 14].set(True)
    ledger = item.state(1)
    @jax.jit
    def step(s, a, ledger):
        after, _, info = jax.vmap(lambda st, ac: env.step(st, ac, key, False, True))(s, a)
        reward, new, metrics = item.transition(s, after, info, a, ledger, goals, jnp.zeros(1, bool))
        return after, reward, new, metrics
    s, r, ledger, m = step(s, jnp.array([[[4, 0], [4, 0]]]), ledger)
    assert int(s.bun_carried[0, 0]) == 1 and bool(ledger['ledger']['custody'][0, 0])
    s, r, ledger, m = step(s, jnp.array([[[2, 2], [4, 0]]]), ledger)
    assert float(m['glue_placed'][0, 0]) == 1 and float(m['glue_qualified_placement'][0, 0]) == 1
    assert float(r[0, 0]) == 0
    hits = 0; paid = 0
    for tick in range(15):
        a = jnp.array([[[2, 0], [2 if tick < 7 else 4, 0]]])
        s, r, ledger, m = step(s, a, ledger)
        hits += float(m['glue_attributed_hit'][0, 0]); paid += float(r[0, 0])
    assert hits == 1 and np.isclose(paid, .5)
    assert int(ledger['glue_token'].sum()) == 0
