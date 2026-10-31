"""Opt-in graded custody-v2 worker; R2-a remains on its frozen old module."""
from pathlib import Path
from types import ModuleType


def worker_source():
    from scripts.bun_native_v2_train import worker_source as parent
    source=parent().replace('bun_native_v2_collector as M','bun_native_v2_custody_collector as M')
    anchor="    bs,ba=M.banks(bs,ba,bm)"
    # Old Bank and common parent carry full old reward ledgers. Add only the
    # versioned custody distance cache; no synthetic carrier states.
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+"\n    from scripts.bun_native_v2_custody_reward import cache_state\n    if 'custody_v2' not in ax:ax=dict(ax,custody_v2=jax.tree.map(lambda x:x[None],cache_state(n)))")
    source=source.replace("r2_antifarm=protocol['r2_antifarm'])", "r2_antifarm=protocol['r2_antifarm'],r2_reward_version=protocol['r2_reward_version'])")
    return source


def main():
    import os
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'],JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',
        JAXBOMB_RULE='bun',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.bun_native_v2_runtime import install
    install()
    module=ModuleType('custody_v2_worker');module.__file__=str(Path(__file__))
    exec(compile(worker_source(),__file__,'exec'),module.__dict__);module.main()


if __name__=='__main__':main()
