"""JE09 H6 full-trajectory failure-fragment DAgger campaign."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = os.environ.get('JE09_NAMESPACE',
    'runs/je09_bun_20261016/stage_h6_full_trajectory_fragment_dagger_v28')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('JAXBOMB_RULE', 'bun')
os.environ.setdefault('JE09_NAMESPACE', NAMESPACE)
sys.path.insert(0, str(ROOT))
from scripts import je09_stage_h5c as base
BASE = ROOT / NAMESPACE


def schema_version(namespace):
    return next((tag for tag in ('v32', 'v31', 'v30', 'v29', 'v28') if tag in namespace), 'v28')


def configure():
    ctl = base.configure()
    version = schema_version(NAMESPACE)
    ctl.SCHEMA = f'je09-eight-device-stage-h6-target-stratified-fragment-dagger-{version}'
    ctl.SMOKE_NAME = os.environ.get('JE09_SMOKE_NAME', 'production_smoke_stage_h6')
    ctl.ENTRY = 'scripts/je09_stage_h6.py'
    original = ctl.configure_worker

    def configure_worker(run):
        worker = original(run)
        from scripts import je09_stage_h6_ops as ops
        from scripts import je09_stage_h5c_ops as base_ops
        from scripts import je09_stage_h5c_dagger as dagger
        worker_module = __import__('scripts.je09_stage_h5c_worker', fromlist=['worker'])
        declaration = json.loads((BASE / 'ROUND_DECLARATION.json').read_text())
        declared_slots = declaration['dagger']['stratified_query']['verifier_slots_per_rollout']['A']
        configured_slots = int(os.environ.get('JE09_H6_VERIFIER_SLOTS', '1'))
        if configured_slots != declared_slots:
            raise RuntimeError(f'verifier slot config {configured_slots} != declaration {declared_slots}')
        worker_module.BASE = BASE
        worker_module.RUN = Path(run)
        worker_module.ops = ops
        worker_module.__dict__['RUN'] = Path(run)
        sys.modules['scripts.je09_bun_worker'] = worker_module
        original_aggregate = dagger.aggregate
        base_ops.collect = ops.collect

        def aggregate(old, batch, key, generation):
            updated, replay, stats = original_aggregate(old, batch, key, generation)
            ctx = batch[-1].reshape((-1,) + batch[-1].shape[2:])
            fragment = ops.earliest_fragment(ctx)
            for i in range(5):
                stats[f'fragment_{i}_ticks'] = jnp.sum(fragment == i).astype(jnp.float32)
            audit_offset = ctx.shape[-1] - 9
            stats['teacher_eligible'] = jnp.sum(ctx[:, audit_offset]).astype(jnp.float32)
            stats['teacher_proposed'] = jnp.sum(ctx[:, audit_offset + 1]).astype(jnp.float32)
            stats['teacher_accepted'] = jnp.sum(ctx[:, audit_offset + 2]).astype(jnp.float32)
            stats['gradient_query'] = jnp.sum(ctx[:, audit_offset + 3]).astype(jnp.float32)
            stats['teacher_rejected'] = jnp.sum(ctx[:, audit_offset + 4]).astype(jnp.float32)
            action_set_size = (ctx[:, 48:53] > 0).sum(-1)
            query = ctx[:, 53] > 0
            for size in range(1, 6):
                stats[f'action_set_size_{size}'] = jnp.sum(query & (action_set_size == size)).astype(jnp.float32)
            legal = batch[6][0].reshape((-1, 5))
            legal_rows = jnp.all((ctx[:, 48:53] <= 0) | legal, axis=-1)
            stats['legal_query'] = jnp.sum(query & legal_rows).astype(jnp.float32)
            stats['full_ticks'] = jnp.sum(ctx[:, 0] == 8).astype(jnp.float32)
            stats['full_episode_ends'] = jnp.sum((ctx[:, 0] == 8) & (ctx[:, 47] > 0)).astype(jnp.float32)
            d = ctx[:, 48:60]
            active = (d[:, 7] > 0) & (ctx[:, 0] == 8)
            selected = (d[:, 8] > 0) & active
            accepted = (d[:, 5] > 0) & selected
            query = (ctx[:, 53] > 0) & active
            kind = d[:, 6].astype(jnp.int32)
            delivery = d[:, 9] > 0
            fragment = jnp.where(delivery, 4, jnp.where(kind == 3, 3,
                jnp.where(kind == 2, 2, jnp.where((kind == 1) | (kind == 4), 1, 0))))
            names = {1: 'a_loose', 2: 'b_stock', 3: 'carry_return', 4: 'cage_delivery'}
            for frag_id, name in names.items():
                stats[f'{name}_candidate_ticks'] = jnp.sum(active & (fragment == frag_id)).astype(jnp.float32)
                stats[f'{name}_proposed'] = jnp.sum(selected & (fragment == frag_id)).astype(jnp.float32)
                stats[f'{name}_accepted'] = jnp.sum(ctx[:, audit_offset + 5 + frag_id - 1]).astype(jnp.float32)
                stats[f'{name}_query'] = jnp.sum(query & (fragment == frag_id)).astype(jnp.float32)
                stats[f'{name}_noneligible_query'] = jnp.sum(query & ~active & (fragment == frag_id)).astype(jnp.float32)
            stats['combat_control_dagger_query'] = jnp.float32(0)
            raw = batch[-1]
            rd = raw[..., 48:60]
            rk = rd[..., 6].astype(jnp.int32)
            rf = jnp.where(rd[..., 9] > 0, 4, jnp.where(rk == 3, 3,
                jnp.where(rk == 2, 2, jnp.where((rk == 1) | (rk == 4), 1, 0))))
            ractive = (rd[..., 7] > 0) & (raw[..., 0] == 8)
            rquery = (raw[..., 53] > 0) & ractive
            signals = jnp.stack([ractive & (rf == i) for i in range(1, 5)], -1)
            qsignals = jnp.stack([rquery & (rf == i) for i in range(1, 5)], -1)
            def count_episode_events(values):
                def step(seen, value):
                    signal, terminal = value
                    first = signal & ~seen
                    seen = jnp.where(terminal[..., None], jnp.zeros_like(seen), seen | signal)
                    return seen, first.sum(0)
                _, counts = jax.lax.scan(step, jnp.zeros_like(values[0][0], dtype=jnp.bool_),
                    (values[0], values[1]))
                return counts.sum(0)
            terminal = raw[..., 47] > 0
            candidate_episodes = count_episode_events((signals, terminal))
            query_episodes = count_episode_events((qsignals, terminal))
            for frag_id, name in enumerate(['a_loose', 'b_stock', 'carry_return', 'cage_delivery']):
                stats[f'{name}_candidate_episodes'] = candidate_episodes[frag_id].astype(jnp.float32)
                stats[f'{name}_query_episodes'] = query_episodes[frag_id].astype(jnp.float32)
            return updated, replay, stats

        worker.ops = ops
        worker.dagger.aggregate = aggregate
        return worker
    ctl.configure_worker = configure_worker
    return ctl


def prepare(ctl):
    from scripts import je09_stage_h5c as h5c
    from scripts.population_run_state import atomic_json, sha
    d = json.loads((BASE / 'ROUND_DECLARATION.json').read_text())
    if not (BASE / 'PREPARED.json').exists():
        h5c.prepare(ctl)
    p = json.loads((BASE / 'plan.json').read_text())
    p['campaign'].update(stage='H6', change=d['change'], stopping='fixed32; no capability early stop')
    p['state_schema'] = ctl.SCHEMA
    p['full_scene_coverage'] = d['full_scene_coverage']
    p['parent_selection'] = d['parent_selection']
    p['dagger'] = d['dagger']
    p['dagger_opponent_course'] = d['dagger_opponent_course']
    atomic_json(BASE / 'plan.json', p)
    seal = json.loads((BASE / 'SOURCE_SEAL.json').read_text())
    seal['plan_sha256'] = sha(BASE / 'plan.json')
    for name in ['scripts/je09_stage_h6.py', 'scripts/je09_bun_continuation.py', 'scripts/je09_stage_h6_ops.py',
                 'scripts/je09_stage_h6_fragments.py', 'scripts/je09_stage_h6_declare.py',
                 'scripts/je09_stage_h6_quality.py', 'scripts/je09_stage_h6_quality_live.py',
                 'scripts/je09_stage_h5c.py', 'scripts/je09_stage_h5c_ops.py', 'scripts/je09_stage_h5_course.py',
                 'scripts/je09_stage_h5c_worker.py', 'tests/test_je09_stage_h6.py']:
        seal['sources'][name] = sha(ROOT / name)
        target = BASE / 'inputs/sealed_sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    atomic_json(BASE / 'SOURCE_SEAL.json', seal)
    atomic_json(BASE / 'PREPARED.json', dict(passed=True, plan_sha256=sha(BASE / 'plan.json'),
        source_seal_sha256=sha(BASE / 'SOURCE_SEAL.json')))
    atomic_json(BASE / 'FROZEN_BEFORE_SMOKE.json', dict(
        round_sha256=sha(BASE / 'ROUND_DECLARATION.json'), plan_sha256=sha(BASE / 'plan.json'),
        source_seal_sha256=sha(BASE / 'SOURCE_SEAL.json')))


def run(ctl):
    quality_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
        JE09_NAMESPACE=str(NAMESPACE), JE09_SMOKE_NAME=os.environ.get('JE09_SMOKE_NAME', 'production_smoke_stage_h6'))
    watcher = subprocess.Popen([sys.executable, '-m', 'scripts.je09_stage_h6_quality_live', '--watch'],
        cwd=ROOT, env=quality_env,
        stdout=(BASE / 'quality_watch.log').open('a'), stderr=subprocess.STDOUT,
        start_new_session=True)
    try:
        from scripts.je09_stage_h3 import run as parent_run
        parent_run(ctl)
    finally:
        watcher.terminate()
        watcher.wait(timeout=30)
        from scripts.je09_stage_h6_quality_live import main as quality_main
        quality_main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'worker', 'restore', 'replay'])
    parser.add_argument('--run', default=NAMESPACE)
    parser.add_argument('--resume')
    args = parser.parse_args()
    ctl = configure()
    if args.command == 'prepare':
        prepare(ctl)
    elif args.command == 'run':
        run(ctl)
    elif args.command == 'restore':
        ctl.restore(ROOT / args.run, ROOT / args.resume)
    else:
        from scripts import je09_stage_h5c_worker as worker
        from scripts import je09_stage_h6_ops as ops
        worker.ops = ops
        ctl.run_worker(ROOT / args.run, ROOT / args.resume if args.resume else None, args.command == 'replay')
