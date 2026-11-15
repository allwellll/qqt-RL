"""Admission checks plus explicit CPU-only native simulation/PPO regressions."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.train_dcu_jax_carry import arguments, plan, SCHEMA


@pytest.mark.parametrize('options', [
    ['--devices', '0,0'], ['--envs', '3'], ['--ticks', '0'],
    ['--minibatch', '3'], ['--hours', 'nan'], ['--hours', '6'],
    ['--interceptor-fraction', 'nan'], ['--interceptor-fraction', '1.1'],
    ['--critic-warmup', '-1'], ['--deadline-utc', '2026-11-15T12:00:00'],
])
def test_invalid_plan(options):
    with pytest.raises(SystemExit):
        arguments(['plan', *options])


def test_read_only_plan_and_pre_jax_stop():
    code = '''import sys
from scripts.train_dcu_jax_carry import main
main(['plan'])
assert 'jax' not in sys.modules
try:
    main(['train', '--out', 'runs/carry-must-not-exist', '--acknowledge-training'])
except RuntimeError as error:
    assert 'Persistent user stop' in str(error)
else:
    raise AssertionError('Persistent stop was bypassed')
assert 'jax' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not Path('runs/carry-must-not-exist').exists()
    row = plan(arguments(['plan']))
    assert row['opponents_frozen'] and not row['online_selfplay']
    assert row['opponent_fractions'] == dict(interceptor=.75, SP_web=.25)


@pytest.fixture(scope='module')
def cpu():
    if os.environ.get('JAX_PLATFORMS') != 'cpu':
        pytest.skip('Set JAX_PLATFORMS=cpu explicitly; these tests must not use DCUs')
    import jax
    import jax.numpy as j
    import numpy as np
    from qqt_rl.training import jax_carry_env as env
    env.prepare()
    assert jax.default_backend() == 'cpu'
    return jax, j, np, env, jax.jit(env.advance)


def clean_world(cpu, positions):
    jax, j, _, env, _ = cpu
    w = env.fresh(jax.random.PRNGKey(12))
    c = w.state.core
    c = c._replace(pos=j.array(positions, j.float32), wall=j.zeros_like(c.wall),
                   brick=j.zeros_like(c.brick), bush=j.zeros_like(c.bush),
                   pushable=j.zeros_like(c.pushable), crate=j.zeros_like(c.crate))
    return w._replace(state=w.state._replace(core=c))


def bun_total(cpu, w):
    _, _, np, _, _ = cpu
    s = w.state
    return int(np.asarray(s.bun_stored).sum() + np.asarray(s.bun_loose).sum()
               + (np.asarray(s.bun_carried) >= 0).sum())


def tick(cpu, w, permissions):
    jax, j, _, _, advance = cpu
    result = advance(w, j.array([[4, 0], [4, 0]]), jax.random.PRNGKey(3), j.array(permissions))
    jax.block_until_ready(result)
    assert bun_total(cpu, result[0]) == bun_total(cpu, w)
    return result


def test_full_map_and_normal_observations(cpu):
    jax, j, np, env, _ = cpu
    w = env.fresh(jax.random.PRNGKey(12))
    assert w.state.brick.sum() > 0 and bun_total(cpu, w) == 2
    worlds = jax.tree.map(lambda x: j.stack([x, x]), w)
    obs, gv, *_ = env.inputs(worlds, j.array([0, 1]))
    assert obs.shape == (2, 25, 13, 15) and gv.shape == (2, 38)
    for seat in (0, 1):
        danger = env.native.C._danger_map(w.state.fuse, w.state.wall,
                                         w.state.core.bomb_blast, w.state.brick)
        np.testing.assert_array_equal(obs[seat, :24], env.E.make_obs(w.state, seat, danger))
        np.testing.assert_array_equal(gv[seat, :24], env.E.global_vec(w.state, seat))
    assert obs[:, 14:16].sum() == 36


@pytest.mark.parametrize('seat', [0, 1])
def test_interceptor_cannot_steal_or_pickup_but_courier_can(cpu, seat):
    _, j, _, env, _ = cpu
    positions = [[10.5, 1.5], [10.5, 13.5]]
    positions[seat] = (env.E._BUN_BASES[1-seat] + j.array([1.5, 1.5])).tolist()
    w = clean_world(cpu, positions)
    permissions = [True, True]
    permissions[seat] = False
    no, _, _, _, info = tick(cpu, w, permissions)
    assert not info['steal'][seat] and no.state.bun_carried[seat] == -1
    yes, _, _, _, info = tick(cpu, w, [True, True])
    assert info['steal'][seat] and yes.state.bun_carried[seat] == 1-seat
    positions[seat] = [7.5, 7.5]
    w = clean_world(cpu, positions)
    s = w.state
    s = s._replace(bun_stored=s.bun_stored.at[1-seat, 1-seat].set(0),
                   bun_loose=s.bun_loose.at[7, 7, 1-seat].set(1))
    w = w._replace(state=s)
    no, _, _, _, info = tick(cpu, w, permissions)
    assert not info['pickup_loose'][seat] and no.state.bun_loose[7, 7, 1-seat] == 1
    yes, _, _, _, info = tick(cpu, w, [True, True])
    assert info['pickup_loose'][seat] and yes.state.bun_carried[seat] == 1-seat


def test_delivery_and_contact_transfer_conserve_stock(cpu):
    _, j, _, env, _ = cpu
    home = (env.E._BUN_BASES[0] + j.array([1.5, 1.5])).tolist()
    w = clean_world(cpu, [home, [10.5, 13.5]])
    s = w.state
    w = w._replace(state=s._replace(bun_carried=s.bun_carried.at[0].set(1),
                                    bun_stored=s.bun_stored.at[1, 1].set(0)))
    no, _, done, _, info = tick(cpu, w, [False, True])
    assert not done and not info['capture'][0] and no.state.bun_carried[0] == 1
    yes, reward, done, _, info = tick(cpu, w, [True, True])
    assert done and info['capture'][0] and yes.state.bun_carried[0] == -1
    assert reward[0] > 0
    w = clean_world(cpu, [[7.5, 7.5], [7.5, 8.5]])
    s = w.state
    w = w._replace(state=s._replace(trapped=s.trapped.at[1].set(30),
        bun_carried=s.bun_carried.at[1].set(0), bun_stored=s.bun_stored.at[0, 0].set(0)))
    no, reward, _, _, info = tick(cpu, w, [False, True])
    assert info['death'][1] and not info['native_transfer'][0]
    assert no.state.bun_carried[0] == -1 and no.state.bun_loose[:, :, 0].sum() == 1
    assert reward[0] == 5  # real contact kill, no delivery/trap/position bonus
    yes, _, _, _, info = tick(cpu, w, [True, True])
    assert info['native_transfer'][0] and yes.state.bun_carried[0] == 0


def test_cargo_uses_native_slow_speed(cpu):
    jax, j, np, env, advance = cpu
    w = clean_world(cpu, [[7.5, 3.5], [7.5, 11.5]])
    actions = j.array([[3, 0], [4, 0]])
    key, permissions = jax.random.PRNGKey(3), j.array([True, True])
    normal = advance(w, actions, key, permissions)[0]
    s = w.state
    carrying = w._replace(state=s._replace(bun_carried=s.bun_carried.at[0].set(1),
                                          bun_stored=s.bun_stored.at[1, 1].set(0)))
    slow = advance(carrying, actions, key, permissions)[0]
    distance = np.linalg.norm(np.asarray(slow.state.pos[0]-s.pos[0]))
    np.testing.assert_allclose(distance, 72/(40*env.native.C.TICK_HZ), atol=1e-5)
    assert distance < np.linalg.norm(np.asarray(normal.state.pos[0]-s.pos[0]))
    assert bun_total(cpu, slow) == 2


def test_no_kill_no_interceptor_reward_and_independent_resets(cpu):
    jax, j, np, env, _ = cpu
    from qqt_rl.training.jax_carry_collect import reset_done, allocation
    w = clean_world(cpu, [[7.5, 3.5], [7.5, 11.5]])
    _, reward, _, _, _ = tick(cpu, w, [False, True])
    assert reward[0] == 0
    worlds = jax.tree.map(lambda x: j.stack([x, x]), w)
    worlds = worlds._replace(state=worlds.state._replace(
        core=worlds.state.core._replace(t=j.array([2400, 17]))))
    reset = jax.jit(reset_done)(worlds, j.array([True, False]), jax.random.PRNGKey(5))
    assert reset.state.t.tolist() == [0, 17]
    for x, y in zip(jax.tree.leaves(reset), jax.tree.leaves(worlds)):
        np.testing.assert_array_equal(x[1], y[1])
    for stage in ('interceptor', 'courier'):
        seats, kinds, permissions = allocation(128, stage)
        for kind in (0, 1):
            assert ((kinds == kind) & (seats == 0)).sum() == ((kinds == kind) & (seats == 1)).sum()
        assert (~permissions).sum() == (128 if stage == 'interceptor' else 96)


def test_carry_checkpoint_prunes_only_its_schema(tmp_path):
    from qqt_rl.training.checkpoints import CheckpointStore
    store = CheckpointStore(tmp_path, keep=3, min_free_gib=0, schema=SCHEMA)
    for index in range(5):
        store.publish(index, dict(actor=[index], critic=[index], optimizer='never save'), dict(schema=SCHEMA))
    assert sorted(p.name for p in tmp_path.glob('u*')) == ['u000000', 'u000003', 'u000004']
    assert json.loads((tmp_path/'LATEST.json').read_text())['optimizer_saved'] is False


def test_read_only_metrics_summary(tmp_path):
    from scripts.summarize_dcu_carry import summarize
    names = ['kills', 'deaths', 'self_deaths', 'carrier_kills', 'steals',
             'captures', 'drops', 'bombs', 'games', 'wins', 'draws', 'return_buns']
    a = [20, 4, 0, 7, 0, 0, 0, 30, 10, 0, 6, 0]
    b = [4, 20, 0, 0, 8, 4, 7, 20, 10, 4, 6, 0]
    (tmp_path/'PLAN.json').write_text(json.dumps(dict(stage='interceptor')))
    row = dict(update=3, environment_steps=1000, environment_steps_per_second=10,
               metric_names=names, cumulative_metrics=[[[0]*12, [0]*12], [a, b]])
    (tmp_path/'updates.jsonl').write_text(json.dumps(row)+'\n'+ '{"unfinished":')
    result = summarize(tmp_path)
    assert result['groups'][0]['opponent_delivery_rate'] == .4
    assert result['groups'][0]['learner_carrier_kills'] == 7


@pytest.mark.parametrize('stage', ['interceptor', 'courier'])
def test_device_collector_gae_and_learner_only_update(cpu, stage):
    jax, j, np, _, _ = cpu
    from jax_bomb.jax_net import init_transformer
    from scripts.r4_k4_observation import actor_critic_forward
    from qqt_rl.training.jax_carry_collect import build as collector
    from qqt_rl.training.kill_ppo import build
    # Tiny random network validates the actual rollout/optimizer interface;
    # no historical weights are trained and no accelerator can be used.
    actor = init_transformer(jax.random.PRNGKey(1), 25, 13, 15,
                             embed=32, depth=1, ff_factor=1, state_dim=38)
    critic = init_transformer(jax.random.PRNGKey(2), 25, 13, 15,
                              embed=32, depth=1, ff_factor=1, state_dim=38)
    assert actor['heads']['wb'][0].shape[-1] == 3, 'Use JAXBOMB_RULE=bun'
    params = dict(actor=actor, critic=critic)
    devices = [jax.devices('cpu')[0]]
    rep = lambda x: jax.device_put_replicated(x, devices)
    bank = rep(jax.tree.map(lambda a: j.stack([a, a]), actor))
    snapshot = jax.tree.map(np.asarray, bank)
    init_world, collect = collector(devices, 8, 2, stage=stage)
    keys = jax.random.split(jax.random.PRNGKey(3), 1)
    world = init_world(keys)
    # A real terminal transition alongside continuing games.
    world = world._replace(state=world.state._replace(core=world.state.core._replace(
        t=world.state.t.at[0, 0].set(2399))))
    world, keys, batch, metrics, health = collect(rep(params), world, bank, keys)
    jax.block_until_ready(batch)
    assert batch['obs'].shape == (1, 16, 25, 13, 15)
    assert world.state.t[0, 0] == 1 and world.state.t[0, 1] == 2
    assert metrics.shape == (1, 2, 2, 12)
    bad = {k: int((~np.isfinite(np.asarray(v))).sum()) for k, v in batch.items()
           if not np.isfinite(np.asarray(v)).all()}
    assert not bad, bad
    assert not any(np.asarray(v).any() for v in health.values())
    from qqt_rl.training.jax_carry_collect import consistency_check
    error = consistency_check(devices, 16)(rep(params), batch)
    assert (np.asarray(error)[:, 0] < .03).all()
    assert (np.asarray(error)[:, 1] < .1).all()
    init_opt, _, update = build(actor_critic_forward, devices, 16)
    (trained, _), loss = update(rep(params), rep(init_opt(params)), rep(actor),
                                batch, keys, np.array([False]))
    jax.block_until_ready(loss)
    assert np.isfinite(np.asarray(loss)).all()
    for branch in ('actor', 'critic'):
        assert any(not np.array_equal(np.asarray(a[0]), np.asarray(b))
                   for a, b in zip(jax.tree.leaves(trained[branch]), jax.tree.leaves(params[branch])))
    for a, b in zip(jax.tree.leaves(bank), jax.tree.leaves(snapshot)):
        np.testing.assert_array_equal(a, b)
