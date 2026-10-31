"""CPU integration of actual native ABI, reward ledgers and fresh restore.

This is a G2 check, not a GPU smoke or a capacity/performance result.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                  JAX_PLATFORM_NAME='cpu', OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1')
import argparse
import json
import pickle
import subprocess
import sys
import time
import numpy as np
from scripts.population_run_state import atomic_json, sha
from scripts.rules_v2_migration import BASE, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    from scripts.rules_v2_runtime import install, assert_native_state
    env = install()
    import jax
    import jax.numpy as j
    from scripts import combat_item_engine as T
    from scripts import carrier_mechanism_engine as E
    from scripts import rules_v2_critic as critic
    from scripts.terminal_checkpoint_restore import digest
    T.configure()
    path = BASE / 'G2_native_state.pkl'
    reference = BASE / 'G2_NATIVE_REFERENCE.json'
    if args.verify:
        expected = json.loads(reference.read_text())
        assert sha(path) == expected['state_sha256']
        with path.open('rb') as stream:
            data = pickle.load(stream)
    else:
        export = json.loads((BASE / 'PARENT_EXPORT.json').read_text())
        assert sha(export['output']) == export['sha256']
        with open(export['output'], 'rb') as stream:
            data = pickle.load(stream)
        keys = jax.random.split(jax.random.PRNGKey(2026110151), 2)
        states, aux = jax.jit(E.initial, static_argnums=1)(keys, 0)
        # Carry the same passive item-chain aux leaf as the frozen training
        # worker. It is telemetry, not part of the policy observation.
        from scripts import combat_item_events
        native_fresh = T.fresh
        def fresh(k, s, a):
            core_aux = {name:value for name,value in a.items() if name!='tactics'}
            st, ax = native_fresh(k, s, core_aux, bank_probability=0.)
            return st, dict(ax, tactics=combat_item_events.state(len(k)))
        T.fresh = jax.jit(fresh)
        # Technical natural reset check only; no old Bank is migrated or used.
        states, aux = jax.jit(T.fresh)(keys, states, aux)
        assert not np.any(np.asarray(states.bun_carried) >= 0)
        data.update(states=states, aux=aux, keys=keys)
        save(path, jax.device_get(data))
    states, aux, keys = data['states'], data['aux'], data['keys']
    assert_native_state(states)
    choose = jax.jit(E.choose)
    nk, ke, actions, _ = choose(data['params'], states, aux, keys)
    next_step = jax.jit(T.transition)(states, aux, actions, ke)
    jax.block_until_ready(next_step)
    after, next_aux, reward, done, info, parts = next_step
    assert_native_state(after)
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(
        jax.device_get((reward, parts, critic.ledger_features(aux),
                        critic.ledger_features(next_aux)))))
    result = dict(next_step_digest=digest((nk,ke,actions,next_step)),
                  Adam_digest=digest(data['adam']), params_digest=digest(data['params']))
    if args.verify:
        for name, value in result.items():
            assert value == expected[name], name
        atomic_json(BASE / 'G2_NATIVE_RESTORE.json', dict(
            status='PASS', fresh_process=True, full_env_aux_reward_RNG_exact=True,
            Adam_exact=True, **result))
        return
    bad = states._replace(native_items=states.native_items._replace(
        overflow=j.ones_like(states.native_items.overflow)))
    try:
        assert_native_state(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Native queue overflow was silently accepted')
    # Actual collector resets externally (auto_reset=False, T.fresh +
    # select_tree), preserving its scene-tag dtype and rebuilding every leaf.
    reset_keys = jax.vmap(lambda k:jax.random.fold_in(k,0xF3))(nk)
    fresh, fresh_aux = jax.jit(T.fresh)(reset_keys, states, aux)
    collector_reset = jax.jit(E.select_tree)(j.ones(states.t.shape,bool),fresh,after)
    collector_aux = jax.jit(E.select_tree)(j.ones(states.t.shape,bool),fresh_aux,next_aux)
    assert_native_state(collector_reset)
    assert np.all(np.asarray(collector_reset.t)==0)
    assert collector_reset.scene.dtype == states.scene.dtype
    assert not np.any(np.asarray(collector_aux['terminal']))
    # The low-level direct reset expects the native scene dtype. This cast is
    # restricted to this API test; no rule or production collector is changed.
    terminal = states._replace(scene=states.scene.astype(j.int8),core=states.core._replace(
        t=j.full_like(states.t, env.MAX_STEPS)))
    reset, finished, _ = jax.jit(jax.vmap(lambda s,a,k:env.step(
        s,a,k,auto_reset=True,return_info=True)))(terminal,actions,ke)
    jax.block_until_ready(reset)
    assert_native_state(reset)
    assert np.all(np.asarray(finished)) and np.all(np.asarray(reset.t)==0)
    assert np.all(np.asarray(reset.spawn_protection)==0)
    atomic_json(reference, dict(**result, state_sha256=sha(path)))
    subprocess.run([sys.executable,'-m','scripts.rules_v2_native_acceptance','--verify'],check=True)
    atomic_json(BASE / 'G2_NATIVE_ACCEPTANCE.json', dict(
        status='PASS', backend='cpu', rule_version='rules_v2_native_parity',
        actual_native_states=2, fresh_natural_init=True, old_midgame_loaded=False,
        native_ABI_no_None=True, reward_ledger_finite=True,
        auto_reset_all_native_fields=True, overflow_explicitly_rejected=True,
        actual_collector_reset='auto_reset=False then T.fresh/select_tree',
        collector_reset_scene_dtype_preserved=True,
        direct_reset_test_only_scene_cast='int32 to native int8; no physics change',
        restore=str(BASE/'G2_NATIVE_RESTORE.json'), state_sha256=sha(path),
        no_GPU_smoke_claim=True, unix=time.time()))


if __name__ == '__main__':
    main()
