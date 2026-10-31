"""Performance-only custody worker; retains every old checkpoint field."""
from pathlib import Path
from types import ModuleType


def worker_source():
    from scripts.bun_native_v2_custody_train import worker_source as original
    s=original()
    # Separate acceptance from the old (preserved) restore receipt. No smoke
    # repeats: already completed updates and their budgets are retained.
    return s.replace("out/'RESTORE.json',dict(passed=True,fresh_process=True", "out/'PERFORMANCE_RESTORE.json',dict(passed=True,fresh_process=True")


def main():
    import os
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'],JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',
        JAXBOMB_RULE='bun',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.bun_native_v2_runtime import install
    install()
    from scripts.bun_native_v2_custody_fast import install as fast
    fast()
    module=ModuleType('custody_v2_performance_worker');module.__file__=str(Path(__file__))
    exec(compile(worker_source(),__file__,'exec'),module.__dict__);module.main()


if __name__=='__main__':main()
