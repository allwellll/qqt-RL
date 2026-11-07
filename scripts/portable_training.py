"""Portable DUEL/SP policy-only new rounds. CPU validation never collects or updates."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path('data/portable-training-20261107')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    value = Path(path)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError('Bundle paths must be relative without parent traversal')
    return value


def verify_bundle():
    manifest = read(BUNDLE / 'MANIFEST.json')
    for row in manifest['files']:
        path = relative(row['path'])
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or linked bundle file: ' + str(path))
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Bundle integrity mismatch: ' + str(path))
    for path, expected in manifest['code_sha256'].items():
        if sha(relative(path)) != expected:
            raise ValueError('Code integrity mismatch: ' + path)
    return manifest


def require_training_authorization():
    """Deny before importing JAX, loading any model, or choosing a device."""
    if not read(ROOT / 'training_launch_policy.json').get('training_enabled', False):
        raise RuntimeError('Persistent user stop: training disabled; renewed user authorization required')
    stop = ROOT / '.hermes/TRAINING_STOP_STATE.json'
    if stop.exists() and not read(stop).get('automatic_resume_allowed', False):
        raise RuntimeError('Persistent private user stop still active')


def configure_runtime(protocol):
    from scripts.cx_items_install_v2 import install
    rule = install(protocol)
    import jax
    import jax.numpy as j
    from scripts.combat_specialist_g_league_train import setup
    backend = setup()
    from scripts import combat_regions_mixed_v2 as regions
    regions.B = Path(protocol['train_manifest']).parent
    if protocol['line'] == 'DUEL':
        os.environ['DUEL_FRACTION'] = str(protocol['duel_fraction'])
        from scripts import cx_duel_mixed_v3 as mixed
        mixed.CATALOG = Path(protocol['duel_scenarios'])
    else:
        mixed = regions
    mixed.WEIGHTS = tuple(protocol['spec']['opponent_weights'])
    # The source is sealed, but data locations are an explicit portable protocol input.
    for name, path in rule['private_modules'].items():
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != Path(path).resolve():
            raise ValueError('Private source not selected: ' + name)
    from jax_bomb import je07_bun_env_r1_items as env
    if env.native_items.ITEM_RULE_VERSION != protocol['ruleset']:
        raise ValueError('Runtime ruleset mismatch')
    bank_manifest = read(protocol['bank_manifest'])
    bs, ba = backend.banks(None, None, bank_manifest)
    spawns = j.asarray(read(protocol['spawn_manifest'])['spawn_cells'])
    return jax, mixed, bs, ba, spawns


def load_actor(protocol, key):
    """All pickle assets are publisher generated and hash verified before unpickling."""
    row = protocol['models'][key]
    if sha(relative(row['path'])) != row['sha256']:
        raise ValueError('Actor SHA mismatch: ' + key)
    with Path(row['path']).open('rb') as stream:
        actor = pickle.load(stream)
    from scripts.r4_k4_observation import migrate
    return migrate(actor)


def fresh_parameters(actor, seed):
    """The old critic is unavailable. Initialize it independently; preserve actor exactly."""
    import jax
    from jax_bomb.jax_net import init_transformer
    embed = actor['state_w'].shape[-1]
    patch = round((actor['tok'][0].shape[0] / 25) ** .5)
    ff_factor = actor['blocks'][0]['ff1'][0].shape[-1] / embed
    critic = init_transformer(jax.random.PRNGKey(seed ^ 0xC817), 25, 13, 15,
        embed=embed, depth=len(actor['blocks']), patch=patch,
        state_dim=actor['state_w'].shape[-2], ff_factor=ff_factor,
        aux_heads='wsafe' in actor['heads'])
    if jax.tree.structure(actor) != jax.tree.structure(critic):
        raise ValueError('Fresh critic schema mismatch')
    if any(a.shape != b.shape for a, b in zip(jax.tree.leaves(actor), jax.tree.leaves(critic))):
        raise ValueError('Fresh critic shape mismatch')
    return jax.tree.map(jax.numpy.asarray, dict(actor=actor, critic=critic))


def initialize(protocol, count):
    jax, mixed, bs, ba, spawns = configure_runtime(protocol)
    import jax.numpy as j
    import numpy as np
    seed = int(protocol['seed'])
    keys = jax.random.split(jax.random.PRNGKey(seed), count)
    backend = mixed.bun_backend()
    physical = j.asarray(mixed.seats(count, seed) if protocol['line'] == 'DUEL'
        else np.tile(np.array([0, 1], np.int32), count // 2))
    kinds = mixed.base.allocation(count, seed, mixed.WEIGHTS)
    def fresh(ks):
        st, ax = backend.fresh(ks, bs, ba, j.zeros(count, bool), spawns, physical)
        meta = backend.metadata(st, kinds, j.zeros(count, bool), physical, ax['r1_attribute_highwater'])
        ax = mixed.migrate_aux(dict(ax, combat_mix=dict(meta, opponent_generation=j.zeros(count, j.int32))))
        return st, ax, ks
    states, aux, keys = jax.jit(fresh)(keys)
    jax.block_until_ready((states, aux, keys))
    actors = {key: jax.tree.map(j.asarray, load_actor(protocol, key)) for key in protocol['models']}
    params = fresh_parameters(actors[protocol['initial_actor']], seed)
    selected = {slot: actors[key] for slot, key in protocol['pool'].items()}
    bank = jax.tree.map(lambda a: j.stack([a, a]), selected)
    return jax, mixed, bs, ba, spawns, params, states, aux, keys, bank


def validate(protocol):
    jax, mixed, bs, ba, spawns, params, states, aux, keys, bank = initialize(protocol, 4)
    import numpy as np
    from scripts.r4_k4_observation import actor_critic_forward
    backend = mixed.bun_backend()
    obs, global_state, mm, bm = backend.inputs(states, aux['combat_mix']['learner_seat'], aux['combat_mix']['bomb_threat']['r3'])
    outputs = actor_critic_forward(params, obs, global_state)
    jax.block_until_ready(outputs)
    assert jax.default_backend() == 'cpu'
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(outputs))
    assert obs.shape == (4, 25, 13, 15) and global_state.shape == (4, 38)
    result = dict(status='PASS_CPU_ONLY', line=protocol['line'], ruleset=protocol['ruleset'],
        devices=[d.platform for d in jax.devices()], private_modules_loaded=4,
        model_count=len(protocol['models']), opponent_slots=sorted(bank), bank_rows=[int(s.t.size) for s in bs],
        fresh_environments=4, scenes=np.asarray(states.scene).tolist(),
        obs_shape=list(obs.shape), global_shape=list(global_state.shape),
        forward_finite=True, optimizer_updates=0, rollout_calls=0, state_pickle_reads=0,
        state_pickle_writes=0, training_enabled=False, critic='fresh deterministic initialization',
        actor_source_update=protocol['models'][protocol['initial_actor']]['source_update'])
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def save_policy(out, index, actor, protocol):
    """Only actor weights and metadata: no optimizer/env/RNG/full-state checkpoint."""
    import jax
    from scripts.population_run_state import atomic_json
    target = out / f'u{index:06d}'
    target.mkdir(parents=True, exist_ok=False)
    path = target / 'policy.pt'
    temporary = target / 'policy.pt.tmp'
    temporary.write_bytes(pickle.dumps(jax.device_get(actor), protocol=5))
    temporary.replace(path)
    atomic_json(target / 'POLICY.json', dict(schema='qqt.policy-only/v1', update=index,
        policy_sha256=sha(path), source_update=protocol['models'][protocol['initial_actor']]['source_update'],
        ruleset=protocol['ruleset'], full_state_resume=False, optimizer_saved=False))
    return path


def train(protocol, args):
    """Bounded new round only; caller has passed the persistent authorization gate."""
    require_training_authorization()
    import numpy as np
    # No implicit device choice and no automatic successor or resume.
    if args.gpu is None or args.updates is None or args.updates < 1:
        raise ValueError('--train requires explicit --gpu and a positive --updates budget')
    if args.out is None:
        raise ValueError('--train requires a new repository-relative --out')
    out = relative(args.out)
    if out.exists():
        raise ValueError('Output already exists: choose a new round directory')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), JAX_PLATFORMS='cuda', JAX_PLATFORM_NAME='cuda',
        JAXBOMB_DEVICE_OVERRIDE=str(args.gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true --xla_disable_hlo_passes=multi_output_fusion')
    from scripts.model_cuda_overlay import install
    install()
    n = args.envs or protocol['spec']['envs_per_device']
    mb = args.minibatch or protocol['spec']['global_minibatch']
    ticks = protocol['spec']['rollout_ticks']
    if n % 4 or (n * ticks) % mb:
        raise ValueError('envs must be divisible by4; rollout size divisible by minibatch')
    jax, mixed, bs, ba, spawns, params, states, aux, keys, bank = initialize(protocol, n)
    import jax.numpy as j
    import optax
    assert jax.default_backend() == 'gpu' and jax.local_device_count() == 1
    updater_module = 'scripts.cx_duel_updater_v1' if protocol['line'] == 'DUEL' else 'scripts.cc_sil_updater_v1'
    optimizer = optax.adam(protocol['spec']['lr'], eps=1e-8)
    adam = {branch: optimizer.init(params[branch]) for branch in ('actor', 'critic')}
    update = importlib.import_module(updater_module).updater(protocol['spec']['sil_coef'])(mb, optimizer)
    collect = jax.jit(mixed.collector(ticks, n, protocol['seed'], spawns))
    key = jax.random.PRNGKey(protocol['seed'] ^ 0x1EA6)
    out.mkdir(parents=True)
    from scripts.population_run_state import atomic_json
    atomic_json(out / 'ROUND.json', dict(schema='qqt.policy-new-round/v1', protocol=protocol,
        optimizer='zero', critic='fresh', pool='rebuilt', successor=False, updates=args.updates))
    pool = [dict(update=0, policy_sha256='initial', actor=params['actor'])]
    rates = {}; gate_rows = []; generation = 0; current_ring = 'initial'
    for index in range(1, args.updates + 1):
        require_training_authorization()
        if (out / 'USER_STOP').exists():
            break
        if index > 1 and (index - 1) % 40 == 0:
            dest = 1 - generation
            if np.any(np.asarray(aux['combat_mix']['opponent_generation']) == dest):
                raise RuntimeError('Old opponent generation still attached to an unfinished episode')
            if protocol['line'] == 'SP':
                from scripts.cc_league2_sp_lib import ring_from_pool, pfsp_weights
                ring = ring_from_pool(pool)
                weights, _ = pfsp_weights(ring, rates)
                picked = ring[int(np.random.default_rng(protocol['seed'] ^ index).choice(len(ring), p=weights))]
            else:
                picked = pool[-1 - int(np.random.default_rng(protocol['seed'] ^ index).integers(min(4, len(pool))))]
            current_ring = picked['policy_sha256']
            selected = {slot: jax.tree.map(lambda a: a[generation], bank[slot]) for slot in bank}
            selected.update(Gcurrent=params['actor'], Gring=picked['actor'])
            bank = jax.tree.map(lambda a, b: a.at[dest].set(b), bank, selected)
            generation = dest
        (states, aux, keys), batch, exposure = collect(params, states, aux, keys, bs, ba, bank, j.asarray(generation, j.int32))
        key, next_key = jax.random.split(key)
        (params, adam), loss = update(params, adam, batch, next_key)
        jax.block_until_ready((params, adam, loss))
        if not all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves((params, adam, loss))):
            raise RuntimeError('Nonfinite training result')
        e = jax.device_get(exposure['bun'])
        w, g = float(e['opponent_win'][4]), sum(float(e[k][4]) for k in ('opponent_win','opponent_loss','opponent_draw'))
        old = rates.setdefault(current_ring, [0., 0.]); old[0] += w; old[1] += g
        gate_rows.append((current_ring,w,g))
        row = dict(update=index, ruleset=protocol['ruleset'], rollout_environment_steps=n*ticks,
            optimizer_steps=n*ticks//mb, loss=np.asarray(loss).mean(0).tolist(),
            exposure=jax.tree.map(lambda a:np.asarray(a).tolist(),exposure))
        with (out / 'updates.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if index % 40 == 0 or index == args.updates:
            path = save_policy(out, index, params['actor'], protocol)
            recent=[x for x in gate_rows[-40:] if x[0]==pool[-1]['policy_sha256']]
            games=sum(x[2] for x in recent); wins=sum(x[1] for x in recent)
            admit = protocol['line'] != 'SP' or games < 40 or wins / max(games,1) >= .5
            if admit:
                pool.append(dict(update=index, policy_sha256=sha(path), actor=jax.device_get(params['actor'])))
            atomic_json(out/'POOL.json', dict(admitted=admit, games=games, wins=wins,
                entries=[{k:v for k,v in x.items() if k!='actor'} for x in pool]))
    atomic_json(out/'STOPPED.json',dict(automatic_successor=False, full_state_resume=False, model_only=True))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--line', choices=('DUEL', 'SP'), required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--updates', type=int)
    parser.add_argument('--envs', type=int)
    parser.add_argument('--minibatch', type=int)
    parser.add_argument('--out')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.train:
        require_training_authorization()
    else:
        os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu')
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        JAXBOMB_RULE='bun', JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS='true')
    verify_bundle()
    protocol = read(BUNDLE / 'protocols' / (args.line + '.json'))
    train(protocol, args) if args.train else validate(protocol)


if __name__ == '__main__':
    main()
