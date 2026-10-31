"""Second authorized seed; only the persistent namespace differs from SP1."""
import os
from types import ModuleType
from scripts.r3_selfplay_train_20261031 import worker_source as parent_source

NAMESPACE='runs/r3_selfplay_ramp_20261031/seed02'


def worker_source():
    source=parent_source();anchor="BASE=ROOT/'runs/r3_selfplay_ramp_20261031'"
    assert source.count(anchor)==1
    return source.replace(anchor,f"BASE=ROOT/'{NAMESPACE}'")


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'],JAX_PLATFORMS='cuda',
        JAX_PLATFORM_NAME='cuda',JAXBOMB_RULE='bun',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.r3_items_runtime import install
    install()
    from scripts.bun_native_v2_custody_fast2 import install as fast
    fast()
    from scripts.bun_spawn_side import install as ownside
    ownside()
    module=ModuleType('r3_sp_seed2_worker');module.__file__=__file__
    exec(compile(worker_source(),__file__,'exec'),module.__dict__)
    module.main()


if __name__=='__main__':main()
