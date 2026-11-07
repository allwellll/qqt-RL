"""ROCm process configuration. Call before importing JAX or creating devices."""
import os


def configure(device):
    if not isinstance(device, int) or device < 0:
        raise ValueError('An explicit nonnegative DCU device is required')
    os.environ.update(HIP_VISIBLE_DEVICES=str(device),
        CUDA_VISIBLE_DEVICES=str(device), JAX_PLATFORMS='rocm',
        JAX_PLATFORM_NAME='rocm', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        JAXBOMB_DEVICE_OVERRIDE=str(device), JAXBOMB_RULE='bun',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # Conservative, executable baseline for the DTK 26.04 / JAX 0.6 compiler.
    # The full native collector still has a substantial first compilation cost.
    # These process-local compiler choices leave the simulator/PPO source intact.
    # An explicit XLA_FLAGS supplied by the caller takes precedence.
    os.environ.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=0 '
        '--xla_disable_hlo_passes=constant_folding,multi_output_fusion '
        '--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true')
