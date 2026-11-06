"""Two fresh CPU consumers must produce the same native next transition."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
                  OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', JAXBOMB_RULE='bun')
import json
import pickle
import subprocess
import sys
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from scripts.population_run_state import atomic_json, sha
from scripts.terminal_checkpoint_restore import digest
from scripts import combat_specialist_g_league_train as worker
from scripts import combat_regions_mixed_v2 as mixed

B = mixed.B


def consume(label):
    worker.setup()
    cp = B / 'inputs/cpu_parent'
    cm = json.loads((cp / 'COMMITTED.json').read_text())
    assert sha(cp / 'state.pkl') == cm['state_sha256']
    with (cp / 'state.pkl').open('rb') as stream:
        payload = pickle.load(stream)
    complete_digest = digest(payload)
    params_digest = digest((payload['params'], payload['adam'], payload['optimizer_key'], payload['league_bank']))
    states, aux, keys = jax.tree.map(lambda x: j.asarray(x[:2]),
        (payload['states'], payload['aux'], payload['keys']))
    aux = mixed.migrate_aux(aux)
    nextkeys, envkeys = j.swapaxes(jax.vmap(lambda k: jax.random.split(k, 2))(keys), 0, 1)
    bun = mixed.bun_backend()
    actions = j.zeros((2, 2, 2), j.int32).at[:, :, 0].set(4)
    result = jax.jit(lambda st, ax, kk: bun.transition(st, ax, actions, kk, 0.))(states, aux, envkeys)
    jax.block_until_ready(result)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(result))
    atomic_json(B / ('CPU_RESTORE_' + label + '.json'), dict(status='PASS', fresh_process=True,
        complete_source_digest=complete_digest, Actor_Critic_Adam_optimizer_RNG_league_digest=params_digest,
        next_native_env_aux_rng_digest=digest((result, nextkeys)), finite=True,
        source_state_sha256=cm['state_sha256'], samples=2,
        scope='complete source deserialization + migrated two-env native next transition; full next PPO checked separately on GPU'))


def main():
    children = []
    for label, cores in (('A', '140-141'), ('B', '142-143')):
        stream = (B / ('CPU_RESTORE_' + label + '.log')).open('ab')
        children.append((subprocess.Popen(['taskset', '-c', cores, sys.executable, '-m',
            'scripts.combat_regions_restore_cpu_v2', label], stdout=stream, stderr=subprocess.STDOUT), stream))
    for child, stream in children:
        assert child.wait() == 0
        stream.close()
    a = json.loads((B / 'CPU_RESTORE_A.json').read_text())
    b = json.loads((B / 'CPU_RESTORE_B.json').read_text())
    assert a == b
    atomic_json(B / 'CPU_FRESH_RESTORE.json', dict(a, status='PASS_TWO_FRESH_PROCESSES',
        worker_receipts=['CPU_RESTORE_A.json', 'CPU_RESTORE_B.json']))
    print((B / 'CPU_FRESH_RESTORE.json').read_text())


if __name__ == '__main__':
    if len(sys.argv) > 1:
        consume(sys.argv[1])
    else:
        main()
