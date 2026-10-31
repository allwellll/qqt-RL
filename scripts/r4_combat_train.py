"""R4 bounded K1/K3 worker: parent complete Adam/Actor/Critic, new episodes."""
import json
import os
from pathlib import Path
from types import ModuleType


def worker_source():
    from scripts.bun_spawn_side_train import worker_source as parent
    source = parent(True, True)
    source = source.replace("BASE=ROOT/'runs/r3_items_route_20261030'", "BASE=ROOT/'runs/r4_combat_20261031'")
    anchor = "    M.configure(protocol['r1_attack'],protocol['r2_antifarm'])"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + "\n    from scripts.r4_combat_collector import install as combat_install\n    combat_install(M,spec['curriculum_fraction'])")
    source = source.replace("protocol['round14_collector_seed']^0x806,.10", "protocol['round14_collector_seed']^0x806,spec['curriculum_fraction']")
    source = source.replace("protocol['round14_collector_seed'],.10,0.,spawns", "protocol['round14_collector_seed'],spec['curriculum_fraction'],0.,spawns")
    anchor = "        assert min(row['exposure']['bank_seat_resets'])>0,'no genuine Bank reset on both physical seats'"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + "\n        assert row['exposure']['r4_curriculum_ticks']>0\n        assert min(row['exposure']['r4_lesson_ticks'][1:])>0,'missing real lessons7/12'")
    return source


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'], JAX_PLATFORMS='cuda',
        JAX_PLATFORM_NAME='cuda', JAXBOMB_RULE='bun', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.r3_items_runtime import install
    install()
    from scripts.bun_native_v2_custody_fast2 import install as fast
    fast()
    from scripts.bun_spawn_side import install as ownside
    ownside()
    module = ModuleType('r4_worker'); module.__file__ = __file__
    exec(compile(worker_source(), __file__, 'exec'), module.__dict__)
    module.main()


if __name__ == '__main__':
    main()
