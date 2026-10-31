"""Exact slot07 recipe/full-state fork with a declared independent slot01 RNG."""
from pathlib import Path
from types import ModuleType


def worker_source():
    from scripts.bomb_threat_r1_continue_train import worker_source as parent
    source = parent()
    anchor = "        assert digest(ring)==expected['ring_digest']\n"
    assert source.count(anchor) == 2  # source handoff, then fresh-restore branch
    source = source.replace(anchor, anchor + """        original_rng_digest=digest(keys)
        retained=digest((params,adam,st,ax,ring,ring_cursor,fixed,warm_rows))
        if protocol.get('slot01_copy07_rekey',False):
            keys=jax.random.split(jax.random.PRNGKey(seed),n)[None]
            assert digest(keys)!=original_rng_digest,'PBT requires an independent RNG'
        assert digest((params,adam,st,ax,ring,ring_cursor,fixed,warm_rows))==retained
""", 1)
    source = source.replace('Actor_Critic_Adam_env_aux_RNG_ring_exact=True,',
        'Actor_Critic_Adam_env_aux_ring_exact=True,donor_RNG_exact_before_rekey=True,')
    source = source.replace('RNG_rekeyed=False,reward_ledger_reset=False,',
        "RNG_rekeyed=protocol.get('slot01_copy07_rekey',False),reward_ledger_reset=False,"
        "donor_parity_checked_before_rekey=True,source_rng_digest=original_rng_digest,"
        "target_rng_digest=digest(keys),")
    cadence = "if index in (int(protocol['chunk_start_update'])+1,int(protocol['chunk_start_update'])+2)"
    assert source.count(cadence) == 1
    source = source.replace(cadence, "if index in (int(protocol['chunk_start_update'])+1,int(protocol['chunk_start_update'])+2,int(protocol['chunk_start_update'])+3)")
    return source


def main():
    import os
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'],
        JAX_PLATFORMS='cuda', JAX_PLATFORM_NAME='cuda', JAXBOMB_RULE='bun',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.bomb_threat_r1_items_runtime import install
    install()
    module = ModuleType('slot01_exact_slot07_fork')
    module.__file__ = str(Path(__file__))
    exec(compile(worker_source(), __file__, 'exec'), module.__dict__)
    module.main()


if __name__ == '__main__':
    main()
