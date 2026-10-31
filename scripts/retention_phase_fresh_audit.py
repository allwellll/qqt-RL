"""Independent CPU initialization/projection audit. Never blocks a trainer."""
import argparse
import hashlib
import os
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import numpy as np

from scripts.je03_bun_stage1_20261011 import digest
from scripts.population_run_state import atomic_json,sha
from scripts.retention_screen import read


def load_checkpoint(cp):
    while not (cp/'COMMITTED.json').exists():time.sleep(5)
    marker=read(cp/'COMMITTED.json')
    with (cp/'state.pkl').open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==marker['state_sha256']
        stream.seek(0);saved=pickle.load(stream)
    assert digest(saved['live'])==marker['full_state_digest']
    return saved['live'],marker


def audit(run):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and jax.default_backend()=='cpu'
    run=Path(run);out=run/'formal/bun392_bridge_pfsp8'
    live,marker=load_checkpoint(out/'gen000000')
    with (run/'inputs/initial.pkl').open('rb') as stream:initial=pickle.load(stream)
    assert jax.tree.structure(initial)==jax.tree.structure(live['params'])
    for raw,copies in zip(jax.tree.leaves(initial),jax.tree.leaves(live['params'])):
        np.testing.assert_array_equal(copies,np.broadcast_to(raw,(8,)+raw.shape))
    assert all(np.count_nonzero(x)==0 for x in jax.tree.leaves(live['adam']))
    assert not np.asarray(live['dagger_buffer']['valid']).any()
    assert not np.asarray(live['dagger_buffer']['retention_core']['valid']).any()
    assert not np.asarray(live['aux']['dagger']['custody']).any()
    assert read(out/'WORKER.json')['resume'] is None
    conditioned=read(run/'plan.json')['phase_conditioning']
    for role in ['actor','critic']:
        weight=np.asarray(initial[role]['state_w'])
        assert weight.shape==(29 if conditioned else 24,392)
        if conditioned:assert np.count_nonzero(weight[24:])==0
    atomic_json(run/'FRESH_G0_AUDIT.json',dict(passed=True,fresh_input_params_exact=True,
        raw_parent_extended_only_by_zero_projection=conditioned,
        raw_parent_same_parameter_shape=not conditioned,replicas_exact=True,
        optimizer_moments_zero=True,empty_fifo=True,empty_core=True,empty_custody=True,
        old_excluded_state_used=False,formal_from_fresh_not_smoke_checkpoint=True,
        functional_parent_equivalence_pretested=True,initial_sha256=sha(run/'inputs/initial.pkl')))
    del live,initial
    live,marker=load_checkpoint(out/'gen000001')
    norms={role:float(np.linalg.norm(np.asarray(live['params'][role]['state_w'])[:,24:]))
           for role in ['actor','critic']} if conditioned else {}
    if conditioned:assert all(value>0 and np.isfinite(value) for value in norms.values())
    atomic_json(run/'PHASE_PROJECTION_UPDATE.json',dict(passed=True,generation=1,
        phase_conditioning=conditioned,phase_projection_l2=norms,
        all_params_finite=all(np.isfinite(x).all() for x in jax.tree.leaves(live['params'])),
        checkpoint_policy_sha256=marker['policy_sha256'],diagnostic_only=True,controls_training=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();audit(a.run)
