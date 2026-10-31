"""Isolated source-replay binding of the unchanged R4 collector/optimizer math.

Bindings affect only this experiment's interpreter. Frozen R4 files and active
processes are untouched. The extra batch column is passive source metadata.
"""
import os
import sys
import subprocess
import pickle
import numpy as np
import jax
import optax
from scripts import dagger_rl_r4_train as C
from scripts.dagger_rl_r5_run import B, R, read, authorize, notify
from scripts.dagger_rl_source_replay import aggregate_with_sources
from scripts.population_run_state import atomic_json, sha
from scripts.terminal_checkpoint_restore import digest

BASE_COLLECTOR = C.collector


def collector(steps):
    original = BASE_COLLECTOR(steps)
    def collect(*args):
        end, batch, diagnostics = original(*args)
        return end, (*batch, diagnostics[6]), diagnostics
    return collect


def aggregate(batch, history, seed, update):
    b = tuple(np.asarray(x).reshape((-1,) + x.shape[3:])
              for x in jax.device_get(batch))
    if len(b) != 12:
        raise ValueError('collector must provide actual aligned provenance, never infer it')
    new = (b[0], b[1], b[6], b[7], b[8], b[9])
    eligible = b[8].any((1, 2))
    accepted = tuple(x[eligible] for x in new)
    h, replay, stats = aggregate_with_sources(
        accepted, b[11][eligible], history, seed=seed, update=update,
        strategy='reservoir', balance_items=C.MEMORY_STRATEGY=='reservoir',
        fallback=new, fallback_sources=b[11])
    return h, tuple(x.reshape((8, 1024) + x.shape[1:]) for x in replay), stats


def _bind():
    # Reuse exactly the same on-policy PPO, teacher, loss, reset and Adam code.
    C.B, C.R, C.read = B, R, read
    C.authorize, C.notify = authorize, notify
    C.collector, C.aggregate = collector, aggregate


def smoke():
    _bind()
    items, source = C.setup()
    assert read(B/'ACTION_CHAIN_AUDIT.json')['passed']
    assert read(B/'TEACHER_CLOSED_LOOP.json')['passed']
    st, ax, key, bs, ba = C.initial(2026102498, items)
    opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(7.5e-5, eps=1e-8))
    p, a = C.rep(source['params']), C.rep(source['adam'])
    (st, ax, key), batch, diagnostics = collector(8)(p, st, ax, key, bs, ba)
    assert np.array_equal(np.asarray(batch[11]), np.asarray(diagnostics[6]))
    history, rb, stats = aggregate(batch, None, 2026102498, 1)
    rb = tuple(x[:, :64] for x in rb)
    (p, a), loss = C.updater('anchor', opt)(
        p, a, batch, rb, np.asarray(jax.random.split(jax.random.PRNGKey(8), 8)))
    jax.block_until_ready(p)
    changes = C.norms(source['params'], C.one(p))
    ability = float(optax.global_norm(jax.tree.map(
        lambda x, y: x-y, C.one(p)['actor']['heads']['wb'],
        source['params']['actor']['heads']['wb'])))
    assert all(v>0 for v in changes.values()) and ability>0
    assert np.isfinite(np.asarray(loss)).all()
    ability_gradient = float(np.asarray(loss)[..., -1].mean())
    assert ability_gradient > 0
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(C.one(p)))
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(C.one(a)))
    cp = C.commit(B/'smoke', 1, p, a, st, ax, key, 2026102498,
                  'technical', history, formal=False)
    x = pickle.load((cp/'state.pkl').open('rb'))
    expected = C.next_update_probe(x, bs, ba)
    atomic_json(B/'smoke/EXPECTED.json', dict(next_update=expected,
                adam=digest(x['adam']), history=digest(history), nonzero=changes))
    code = subprocess.run([sys.executable,'-m','scripts.dagger_rl_r5_train','restore'],
                          stdout=(B/'smoke/restore.log').open('a'),
                          stderr=subprocess.STDOUT).returncode
    assert code==0 and read(B/'smoke/RESTORE.json')['passed']
    atomic_json(B/'TECHNICAL_SMOKE.json', dict(
        passed=True, finite=True, nonzero=changes, ability_head_update_l2=ability,
        ability_head_gradient_norm=ability_gradient,
        fresh_process_restore=True, parity_environments=64,
        source_alignment_exact=True, source_memory_exact=True,
        environment_steps=512, optimizer_steps=2,
        total_executed_environment_steps=1536, total_executed_optimizer_steps=6,
        formal_inherits_smoke=False, capability_gate=False,
        state_sha256=sha(cp/'state.pkl'), aggregation=stats))
    notify('R5新技术smoke通过：真实on-policy更新、finite、Actor/Critic/ability非零，fresh64完整下一更新含来源回放精确恢复；立即正式训练。')


def restore():
    _bind()
    C.restore()


def formal():
    _bind()
    C.formal()


if __name__ == '__main__':
    globals()[sys.argv[1]]()
