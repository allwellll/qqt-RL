"""GPU4-7 continuation (v2: survival-filtered placement self-imitation, cc_sil_updater_v2) of the healthy G_REGIONS_V2 league recipe (CC owner).

Runs the unmodified combat_specialist_g_league_train worker with the same text
replacements as combat_regions_train_v2, plus:
  * CUDA cond-guard overlay installed before any reward module is imported;
  * BASE / league snapshots / stop files redirected to runs/cc_train_20261105/training;
  * optional advantage-weighted placement self-imitation (protocol arm field sil_coef);
  * per-device lock and reservation check for physical GPU 4-7.
usage: model_training.py <config> args=[protocol, arm, seed, (--smoke|--verify-restore)]
"""
import fcntl, json, os, sys
from pathlib import Path

BASE = Path('runs/cc_train_20261105/training')


def gate():
    device = int(os.environ['JAXBOMB_DEVICE_OVERRIDE'])
    grant = json.loads(Path('.hermes/GPU_USER_RESERVATION.json').read_text())
    assert grant['status'] == 'USER_AUTHORIZED_GPU4567' and device in grant['allowed_gpus'] and device not in grant['disabled_gpus']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(device)
    BASE.mkdir(parents=True, exist_ok=True)
    handle = (BASE / f'GPU{device}.lock').open('a+')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def main():
    guard = gate()
    from scripts.model_cuda_overlay import install
    overlay = install()
    protocol = json.loads(Path(sys.argv[1]).read_text())
    coef = float(protocol['arms'][sys.argv[2]].get('sil_coef', 0.))
    source = Path('scripts/combat_specialist_g_league_train.py').read_text()
    replacements = {
        "BASE = Path('runs/combat_specialist_league_20261101')": f"BASE = Path('{BASE}')",
        'combat_specialist_g_mixed as mixed': 'combat_regions_mixed_v2 as mixed',
        'combat_specialist_snapshots as snapshots': 'cc_snapshots_v1 as snapshots',
        "    env_steps = int(payload['environment_steps']); opt_steps = int(payload['optimizer_steps'])":
            "    aux = mixed.migrate_aux(aux)\n    env_steps = int(payload['environment_steps']); opt_steps = int(payload['optimizer_steps'])",
        "    for path, expected in protocol['source_sha256'].items(): assert sha(path) == expected, path":
            "    for path, expected in protocol['source_sha256'].items(): assert sha(path) == expected, path\n    course=protocol['curriculum']; assert sha(course['train_manifest']) == course['train_manifest_sha256']",
        "XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',":
            "XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true --xla_disable_hlo_passes=multi_output_fusion',",
        "JAX_COMPILATION_CACHE_DIR=str(BASE/'cache/G-league-v1')": "JAX_COMPILATION_CACHE_DIR=str(BASE/'cache'/os.environ['JAXBOMB_DEVICE_OVERRIDE'])",
    }
    if coef:
        replacements["    update = CompiledOperator(mixed.updater(spec['global_minibatch'], optimizer))"] = \
            "    from scripts.cc_sil_updater_v2 import updater as sil_updater\n    update = CompiledOperator(sil_updater(float(spec['sil_coef']))(spec['global_minibatch'], optimizer))"
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    out = Path(protocol['namespace']) / sys.argv[2]; out.mkdir(parents=True, exist_ok=True)
    os.environ['CC_SNAPSHOT_ROOT'] = str(BASE / 'league' / sys.argv[2]); assert (BASE / 'league' / sys.argv[2] / 'league_snapshots/G.json').exists()
    from scripts.population_run_state import atomic_json
    atomic_json(out / f'RUNTIME_{os.getpid()}.json', dict(overlay=overlay, sil_coef=coef, argv=sys.argv[1:],
                device=os.environ['JAXBOMB_DEVICE_OVERRIDE']))
    scope = {'__name__': 'cc_safe_worker', '__file__': __file__}
    exec(compile(source, __file__, 'exec'), scope)
    scope['main']()


if __name__ == '__main__':
    main()
