"""CC_LEAGUE_V2 trainer (cc-league-v2): G_REGIONS_V2 league worker with peer opponents and progressive weights.

Same unmodified combat_specialist_g_league_train worker and replacements as cc_train_v1/v3, plus:
  * opponent weights taken from the arm spec (protocol opponent_weights, 9 slots, see cc_league2_lib);
  * C family = live peer arms' COMMITTED saves (cc_league2_lib.load), G_frozen fallback, recorded per refresh;
  * on restore every env's stored opponent kind is remapped to the static allocation of the current weights;
  * fork (no own COMMITTED yet, source carries a league bank): both bank generations rebuilt from the v2
    manifests, collector/optimizer keys folded with the arm salt (independent RNG only on fork);
  * after every 80-update save the progression rule runs; if it fires the worker saves and exits 0
    with PROGRESSION_REQUEST.json (the supervisor writes the reweighted protocol and re-verifies).
usage: model_training.py <config> args=[protocol, arm, seed, (--smoke|--verify-restore)]
"""
import fcntl, json, os, sys, zlib
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


FORK = '''
    from scripts import cc_league2_lib as league2
    if 'league_bank' in payload:
        aux, remap = league2.remap_kinds(aux, n, args.seed, mixed.WEIGHTS, mixed.base.allocation)
        if not cps:
            salt = zlib.crc32(args.arm.encode()) & 0x7FFFFFFF
            keys = jax.vmap(lambda k: jax.random.fold_in(k, salt))(keys); update_key = jax.random.fold_in(update_key, salt)
            selected, source_manifest = next_generation(start)
            bank = jax.tree.map(lambda a: j.stack([a, a]), selected)
            manifests = {'0': source_manifest, '1': source_manifest}
        atomic_json(out/f'LEAGUE2_START_u{start:06d}.json', dict(fork=not cps, checkpoint=str(cp), update=start,
            weights=list(mixed.WEIGHTS), rng_salt=None if cps else salt, peers=manifests[str(generation)]['C'].get('peer_arm'),
            fallback=manifests[str(generation)]['C'].get('fallback'), unix=time.time(), **remap))

    def operation(p, a, st, ax, keys, uk, bank, gen):'''


def main():
    guard = gate()
    from scripts.model_cuda_overlay import install
    overlay = install()
    protocol = json.loads(Path(sys.argv[1]).read_text())
    spec = protocol['arms'][sys.argv[2]]
    coef = float(spec.get('sil_coef', 0.)); sil = spec['league2']['sil_updater']
    assert sil in ('cc_sil_updater_v1', 'cc_sil_updater_v3')
    source = Path('scripts/combat_specialist_g_league_train.py').read_text()
    replacements = {
        "BASE = Path('runs/combat_specialist_league_20261101')": f"BASE = Path('{BASE}')",
        'combat_specialist_g_mixed as mixed': 'combat_regions_mixed_v2 as mixed',
        'from scripts import combat_specialist_snapshots as snapshots': 'import zlib\n    from scripts import cc_league2_lib as snapshots',
        "    env_steps = int(payload['environment_steps']); opt_steps = int(payload['optimizer_steps'])":
            "    aux = mixed.migrate_aux(aux)\n    env_steps = int(payload['environment_steps']); opt_steps = int(payload['optimizer_steps'])",
        "    for path, expected in protocol['source_sha256'].items(): assert sha(path) == expected, path":
            "    for path, expected in protocol['source_sha256'].items(): assert sha(path) == expected, path\n    course=protocol['curriculum']; assert sha(course['train_manifest']) == course['train_manifest_sha256']",
        "XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',":
            "XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true --xla_disable_hlo_passes=multi_output_fusion',",
        "JAX_COMPILATION_CACHE_DIR=str(BASE/'cache/G-league-v1')": "JAX_COMPILATION_CACHE_DIR=str(BASE/'cache'/os.environ['JAXBOMB_DEVICE_OVERRIDE'])",
        "    update = CompiledOperator(mixed.updater(spec['global_minibatch'], optimizer))":
            f"    from scripts.{sil} import updater as sil_updater\n    update = CompiledOperator(sil_updater(float(spec['sil_coef']))(spec['global_minibatch'], optimizer))",
        "    collect = CompiledOperator(":
            "    mixed.WEIGHTS = tuple(float(x) for x in spec['opponent_weights'])\n    collect = CompiledOperator(",
        "            manifest = snapshots.load(family); provenance[family] = manifest":
            "            manifest = snapshots.load(family, index); provenance[family] = manifest",
        "\n    def operation(p, a, st, ax, keys, uk, bank, gen):": FORK,
        "            stage('ATOMIC_SAVE',index);cp,receipt=save(index,last)":
            "            stage('ATOMIC_SAVE',index);cp,receipt=save(index,last)\n"
            "            if not args.smoke and counted%80==0 and snapshots.progression(out,spec,index):\n"
            "                stage('PROGRESSION',index); return",
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    out = Path(protocol['namespace']) / sys.argv[2]; out.mkdir(parents=True, exist_ok=True)
    os.environ['CC_SNAPSHOT_ROOT'] = str(BASE / 'league' / sys.argv[2]); assert (BASE / 'league' / sys.argv[2] / 'league_snapshots/G.json').exists()
    os.environ['CC_LEAGUE2_ARM'] = sys.argv[2]; os.environ['CC_LEAGUE2_PEERS'] = json.dumps(spec['league2']['peers'])
    from scripts.population_run_state import atomic_json
    atomic_json(out / f'RUNTIME_{os.getpid()}.json', dict(overlay=overlay, sil_coef=coef, sil_updater=sil, argv=sys.argv[1:],
                weights=spec['opponent_weights'], device=os.environ['JAXBOMB_DEVICE_OVERRIDE']))
    scope = {'__name__': 'cc_league2_worker', '__file__': __file__}
    exec(compile(source, __file__, 'exec'), scope)
    scope['main']()


if __name__ == '__main__':
    main()
