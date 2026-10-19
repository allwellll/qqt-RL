"""CPU-only terminal checkpoint validation against its saved source snapshot.

A successor may install its own sources after the previous worker exits. This
does not relax the running worker seal: terminal validation checks the saved
source bytes and complete checkpoint, without importing training operations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.population_run_state import atomic_json,sha


def verify_saved_sources(base):
    base=Path(base);seal=json.loads((base/'SOURCE_SEAL.json').read_text())
    assert sha(base/'plan.json')==seal['plan_sha256'],'saved plan mismatch'
    for name,expected in seal['sources'].items():
        path=Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert sha(base/'inputs/sealed_sources'/path)==expected,'saved source mismatch: '+name
    return dict(saved_source_count=len(seal['sources']),saved_source_snapshot_verified=True,
                plan_sha256=seal['plan_sha256'])


def digest(tree):
    # The checkpoint protocol hashes tree structure, shapes, dtypes and bytes.
    import jax
    import numpy as np
    value=hashlib.sha256();leaves,structure=jax.tree.flatten(tree)
    value.update(str(structure).encode())
    for leaf in leaves:
        array=np.asarray(leaf)
        value.update(str(array.shape).encode());value.update(str(array.dtype).encode());value.update(array.tobytes())
    return value.hexdigest()


def verify_state_schema(checkpoint,plan):
    schema=checkpoint['state_schema']
    assert schema==plan['state_schema'],'checkpoint schema differs from sealed run plan'
    assert schema in ['je10-control-route-v38','je10-control-route-v38-prestock-v1','je10-control-route-v38-clean-causal-init-v1']
    return schema


def restore(base,checkpoint):
    import jax
    import jax.numpy as jnp
    import numpy as np
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and jax.default_backend()=='cpu'
    base=Path(base);checkpoint=Path(checkpoint)
    source_proof=verify_saved_sources(base)
    proof=json.loads((checkpoint/'COMMITTED.json').read_text())
    assert proof['passed'] and proof['replicas_exact'] and proof['devices']==8
    assert proof['shape']==[8,1024,256,4096]
    verify_state_schema(proof,json.loads((base/'plan.json').read_text()))
    assert proof['plan_sha256']==source_proof['plan_sha256']
    assert sha(checkpoint/'state.pkl')==proof['state_sha256']
    assert sha(checkpoint/'policy.pt')==proof['policy_sha256']
    with (checkpoint/'state.pkl').open('rb') as stream:saved=pickle.load(stream)
    assert saved['shape']==proof['shape'] and saved['generation']==proof['generation']
    assert digest(saved['live'])==proof['full_state_digest']
    required={'params','adam','states','aux','key','ema','pool','funnel','pool_counts','course',
              'pfsp','assist_level','assist_proof_hashes','dagger_buffer'}
    if 'distance_stage' in saved['live']:required.add('distance_stage')
    assert set(saved['live'])==required
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(saved['live']))
    next_keys=np.asarray(jax.vmap(lambda key:jax.random.split(key)[0])(jnp.asarray(saved['live']['key'])))
    assert digest(next_keys)==proof['next_rng_digest']
    receipt=dict(passed=True,fresh_process=True,cpu_only=True,generation=proof['generation'],
        complete_fields=sorted(required),full_state_digest=proof['full_state_digest'],
        next_rng_digest=proof['next_rng_digest'],capability_gate=False,**source_proof)
    atomic_json(base/'state/TERMINAL_RESTORE_MONITOR.json',receipt)
    print(json.dumps(receipt),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');parser.add_argument('checkpoint')
    args=parser.parse_args();restore(ROOT/args.run,ROOT/args.checkpoint)
