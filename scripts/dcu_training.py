"""ROCm adapter for the sealed portable DUEL/SP trainer.

Retains its CLI, bundle verification, stop gates, bounded updates, league pool
and policy-only checkpoints. Only the CUDA process setup is replaced in memory;
the release's hash-pinned source and manifest remain unchanged on disk.
"""
import hashlib
import inspect

from qqt_rl.training.dcu import configure
from scripts import portable_training as portable

SOURCE_SHA256 = 'fea8266e7c8072713e5034ee9144eeb595c80ad4e09d9634a68fa68a97333e13'
CUDA_SETUP = """    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), JAX_PLATFORMS='cuda', JAX_PLATFORM_NAME='cuda',
        JAXBOMB_DEVICE_OVERRIDE=str(args.gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true --xla_disable_hlo_passes=multi_output_fusion')
    from scripts.model_cuda_overlay import install
    install()
"""


def adapted_train():
    source_path = portable.ROOT / 'scripts/portable_training.py'
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('Portable trainer changed; review the DCU runtime adapter before use')
    source = inspect.getsource(portable.train)
    if source.count(CUDA_SETUP) != 1:
        raise RuntimeError('CUDA setup anchor changed; refusing an ambiguous runtime adaptation')
    scope = dict(vars(portable), _configure_dcu=configure)
    exec(compile(source.replace(CUDA_SETUP, '    _configure_dcu(args.gpu)\n'),
        str(source_path) + ':rocm-adapter', 'exec'), scope)
    return scope['train']


def main():
    portable.train = adapted_train()
    portable.main()


if __name__ == '__main__':
    main()
