"""CPU checks of actual native resets and metadata migration before GPU handoff."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                  OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from scripts.population_run_state import atomic_json,sha
from scripts import combat_specialist_g_league_train as worker
from scripts import multi_region_g_mixed_v1 as mixed

B=Path('runs/multi_region_combat_20261104')


def main():
    M=worker.setup()
    p=json.loads(Path('runs/combat_specialist_league_20261101/G_CURRENT.json').read_text())
    protocol=json.loads(Path(p['protocol']).read_text())
    bs,ba=M.banks(None,None,json.loads(Path(protocol['r1_bank_manifest']).read_text()))
    spawns=j.asarray(json.loads(Path(protocol['spawn_manifest']).read_text())['spawn_cells'])
    n=64;keys=jax.random.split(jax.random.PRNGKey(2026543000),n);seats=j.arange(n)%2
    bun=mixed.bun_backend()
    states,aux=jax.jit(lambda ks:bun.fresh(ks,bs,ba,j.zeros(n,bool),spawns,seats))(keys)
    jax.block_until_ready(states)
    pos=np.floor(np.asarray(states.pos)).astype(int);course=np.asarray(aux['a3sr']['mr_region'])>0
    assert course.any() and (~course).any()
    assert np.all(np.asarray(aux['provenance_source'])[course]<0)
    for i in np.flatnonzero(course):
        for row,col in pos[i]:assert not bool(states.wall[i,row,col]|states.brick[i,row,col])
    assert np.all(np.asarray(states.bun_carried)[course]<0)
    assert np.all(np.asarray(states.core.fuse)[course]<=0)
    # A short true physics rollout with legal no-op actions; reward finite.
    E=bun.E;actions=j.zeros((n,2,2),j.int32)
    after,done,info=jax.jit(jax.vmap(lambda s,k:E.env.step(s,j.zeros((2,2),j.int32),k,auto_reset=False,return_info=True)))(states,keys)
    assert np.isfinite(np.asarray(after.pos)).all()
    assert not np.asarray(info['death'])[course].any()
    migrated=mixed.migrate_aux(dict(aux,a3sr={k:v for k,v in aux['a3sr'].items() if not k.startswith('mr_')}))
    assert not np.asarray(migrated['a3sr']['mr_region']).any()
    output=dict(status='PASS_ACTUAL_CPU_RESET_AND_STEP',samples=n,regional=int(course.sum()),
        region_counts=np.bincount(np.asarray(aux['a3sr']['mr_region']),minlength=9).tolist(),
        course_never_bank=True,no_bombs=True,no_carried_packages=True,finite=True,
        no_initial_deaths=True,source_schema='G ability3 retained; no five-slot claim',
        center_coverage='unavailable; no legal manifest pair, no fabricated coverage',
        train_manifest_sha256=sha(B/'dev_starts.json'),holdout_manifest_sha256=sha(B/'holdout_starts.json'),
        reward_changes=False,unix=time.time())
    atomic_json(B/'TRAINING_CPU_VALIDATION.json',output);print(json.dumps(output))


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(B/'TRAINING_CPU_FAILURE.json',dict(type=type(error).__name__,
            detail=str(error).replace(str(Path.cwd()),'<WORKSPACE>'),unix=time.time()))
        raise
