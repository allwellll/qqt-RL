"""CPU-only, explicit K4 bun-mode export. Never replaces full checkpoint policy."""
import json
import os
import pickle
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from scripts.population_run_state import atomic_json,sha


def validate_derivative(full,derived):
    from scripts.r4_k4_observation import bun_mode_zero_actor
    import jax
    import numpy as np
    expected=bun_mode_zero_actor(full)
    if jax.tree.structure(derived)!=jax.tree.structure(expected):
        raise ValueError('mode0 derivative tree mismatch')
    if not all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves(derived),jax.tree.leaves(expected))):
        raise ValueError('mode0 derivative retained weights differ')
    return True


def export(cp,dest):
    cp=Path(cp);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    source=json.loads((cp/'COMMITTED.json').read_text())
    assert sha(cp/'policy.pt')==source['policy_sha256']
    derivative=cp/'policy_bun_mode0.pt'
    assert sha(derivative)==source['bun_mode_zero_policy_sha256']
    receipt=dest/'K4_MODE0_IDENTITY.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        assert saved['source_policy_sha256']==source['policy_sha256']
        assert all(sha(name)==value for name,value in saved['files'].items())
        return saved
    # Derivative is produced atomically by the worker. Check every retained
    # weight exactly, so an unrelated37-dimensional actor cannot be exported.
    from scripts.r3_items_export import export_weights
    full=pickle.load((cp/'policy.pt').open('rb'))
    derived=pickle.load(derivative.open('rb'))
    validate_derivative(full,derived)
    export_source=dict(source,policy_sha256=source['bun_mode_zero_policy_sha256'],
        full_k4_source_policy_sha256=source['policy_sha256'],mode='bun=0')
    result=export_weights(derived,dest,export_source)
    model=dest/'model.json';document=json.loads(model.read_text())
    document['meta'].update(r4_mode=0,r4_full_checkpoint_sha256=source['policy_sha256'],
        r4_full_input_abi='obs25-global38',r4_export='exact constant-mode0 derivative, existing R3 global37 ABI')
    atomic_json(model,document)
    result['files'][str(model)]=sha(model);result['model_sha256']=sha(model)
    atomic_json(dest/'EXPORT.json',result)
    identity=dict(status='PASS',source_checkpoint=str(cp),source_policy_sha256=source['policy_sha256'],
        derived_policy_sha256=source['bun_mode_zero_policy_sha256'],mode=0,teacher=False,
        exact_retained_weights=True,model_json=str(model),model_sha256=sha(model),files=result['files'])
    atomic_json(receipt,identity);return identity


if __name__=='__main__':
    import sys
    export(*sys.argv[1:])
