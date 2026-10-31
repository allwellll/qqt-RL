"""Second opt-in performance worker; all reward/state/optimizer ABI retained."""
import os
from pathlib import Path
from types import ModuleType
from scripts.bun_native_v2_custody_fast_train import worker_source as previous_source


def worker_source():
    source = previous_source().replace("out/'PERFORMANCE_RESTORE.json'", "out/'PERFORMANCE2_RESTORE.json'")
    anchor = "        if index in (int(protocol['chunk_start_update'])+1,"
    assert source.count(anchor) == 1
    # Additional evidence checkpoints after this handoff. The original chunk
    # end, environment/update budget and regular40-update cadence stay fixed.
    return source.replace(anchor, "        if index in (start+1,start+2,start+3) or index in (int(protocol['chunk_start_update'])+1,")


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'], JAX_PLATFORMS='cuda',
        JAX_PLATFORM_NAME='cuda', JAXBOMB_RULE='bun', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.bun_native_v2_runtime import install
    install()
    from scripts.bun_native_v2_custody_fast2 import install as fast
    fast()
    module = ModuleType('custody_v2_performance2_worker')
    module.__file__ = str(Path(__file__))
    exec(compile(worker_source(), __file__, 'exec'), module.__dict__)
    module.main()


if __name__ == '__main__':
    main()
