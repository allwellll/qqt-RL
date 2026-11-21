"""Export complete map806 games with a frozen carry defender and SP_web."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from scripts.train_dcu_jax_carry import SP_SHA, load_verified, read_checkpoint


DEFAULT_CHECKPOINT = Path('runs/carry-defender-u244-20261121/u000104')
DEFAULT_OUTPUT = Path('web/carry-replays/u104-vs-spweb-20261121.json')
SEEDS = (20261121, 20261122)
EVENTS = ('steal', 'capture', 'drop', 'death', 'recover', 'return_bun')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--courier-checkpoint', type=Path, help='Use a trained courier Actor instead of SP_web')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    meta = read_checkpoint(args.checkpoint, 'interceptor')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'
    os.environ.setdefault('JAX_DISABLE_MOST_OPTIMIZATIONS', 'true')
    os.environ.setdefault('XLA_FLAGS', '--xla_cpu_enable_fast_math=false '
                          '--xla_disable_hlo_passes=constant_folding')

    import jax
    import jax.numpy as j
    import numpy as np
    import torch
    from qqt_rl.training import jax_carry_env as env
    from scripts import portable_training as portable
    from scripts.je03_native_vit_export_20261011 import NativeViT
    from scripts.retention_phase_conditioning import features

    torch.set_num_threads(2)

    protocol = portable.read(portable.BUNDLE / 'protocols/SP.json')
    if protocol['models']['SP_web']['sha256'] != SP_SHA:
        raise ValueError('Unexpected SP_web model')
    courier = portable.load_actor(protocol, 'SP_web')
    defender = load_verified(args.checkpoint / 'actor.pt', meta['actor_sha256'], courier)
    courier_model = dict(sha256=SP_SHA, update=23302, name='SP_web')
    if args.courier_checkpoint:
        courier_meta = read_checkpoint(args.courier_checkpoint, 'courier')
        digest = courier_meta['actor_sha256']
        courier = load_verified(args.courier_checkpoint / 'actor.pt', digest, courier)
        courier_model = dict(sha256=digest, update=courier_meta['update'], name='courier',
                             environment_steps=courier_meta['global_environment_steps'])
    global_rows = courier['state_w'].shape[-2]
    if global_rows not in (38, 43) or courier['tok'][0].shape[0] != 25 * 9:
        raise ValueError('Replay inference requires the map806 ViT observation ABI')
    defender, courier = NativeViT(defender).eval(), NativeViT(courier).eval()
    env.prepare()

    specs = [(seed, seat) for seed in SEEDS for seat in (0, 1)]
    records = [dict(seed=seed, defender_seat=seat, frames=[], events=[])
               for seed, seat in specs]
    seats = j.array([seat for _, seat in specs])
    rows = j.arange(len(specs))
    can_transport = j.arange(2)[None, :] != seats[:, None]
    worlds = jax.vmap(env.fresh)(j.stack([jax.random.PRNGKey(seed) for seed, _ in specs]))
    max_ticks = int(env.E.LESSON_MAX_STEPS[env.E.LESSON_FULL])
    tick_hz = env.native.C.TICK_HZ
    done = np.zeros(len(specs), dtype=bool)
    totals = np.zeros((len(specs), 2, len(env.METRICS)), dtype=np.int64)
    randoms = np.random.default_rng(2026112104)

    @jax.jit
    def inputs(world, player):
        obs, gv, mm, bm, _ = env.inputs(world, player)
        if global_rows == 43:
            gv = j.concatenate([gv[:, :24], features(obs[:, :24], gv[:, :24]), gv[:, 24:]], -1)
        return obs, gv, mm, bm

    @jax.jit
    def transition(world, actions, keys):
        after, _, finished, metrics, info = jax.vmap(env.advance)(
            world, actions, keys, can_transport)
        return after, finished, metrics, info['winner'], {
            name: info[name] for name in EVENTS
        }

    def sample(logits, mask):
        if not mask.any(axis=-1).all():
            raise ValueError('Empty replay action mask')
        if not np.isfinite(logits[mask]).all():
            raise ValueError(f'Nonfinite legal replay logits: {logits.tolist()} with mask {mask.tolist()}')
        z = np.where(mask, logits, -np.inf).astype(np.float64)
        probabilities = np.exp(z - z.max(axis=-1, keepdims=True))
        cumulative = (probabilities / probabilities.sum(axis=-1, keepdims=True)).cumsum(axis=-1)
        cumulative[:, -1] = 1
        return (randoms.random((len(specs), 1)) > cumulative).sum(axis=-1)

    def observe(actor, player):
        obs, gv, mm, bm = jax.device_get(inputs(worlds, player))
        with torch.inference_mode():
            move, ability, _ = actor(torch.from_numpy(obs.copy()), torch.from_numpy(gv.copy()))
        return j.asarray(np.stack([sample(move.numpy(), mm), sample(ability.numpy(), bm)], -1), j.int32)

    def snapshot(world):
        st = world.state
        core = st.core
        pos, alive, trapped, carry, stored, loose, fuse, owner, brick, blast, ticks, overflow = (
            jax.device_get((core.pos, core.alive, st.trapped, st.bun_carried, st.bun_stored,
                            st.bun_loose, core.fuse, core.owner, core.brick, core.blast_linger,
                            st.t, st.native_items.overflow))
        )
        if np.any(overflow) or not np.isfinite(pos).all():
            raise ValueError('Invalid simulator state')
        for i, record in enumerate(records):
            if done[i]:
                continue
            if carry[i, record['defender_seat']] >= 0:
                raise ValueError('Defender acquired a bun')
            record['frames'].append(dict(
                tick=int(ticks[i]), p=np.round(pos[i], 3).tolist(),
                alive=alive[i].tolist(), trapped=trapped[i].tolist(),
                carry=carry[i].tolist(), stored=stored[i].tolist(),
                loose=np.argwhere(loose[i].sum(axis=-1) > 0).tolist(),
                bombs=[[int(r), int(c), int(fuse[i, r, c]), int(owner[i, r, c])]
                       for r, c in np.argwhere(fuse[i] > 0)],
                bricks=np.flatnonzero(brick[i]).tolist(),
                blasts=np.flatnonzero(blast[i]).tolist(),
            ))

    snapshot(worlds)
    print(f'Recording {len(specs)} games, at most {max_ticks} ticks each', flush=True)
    for tick in range(1, max_ticks + 1):
        actions = j.zeros((len(specs), 2, 2), j.int32)
        actions = actions.at[rows, seats].set(observe(defender, seats))
        actions = actions.at[rows, 1 - seats].set(observe(courier, 1 - seats))
        keys = j.stack([jax.random.fold_in(jax.random.PRNGKey(seed), tick)
                        for seed, _ in specs])
        worlds, finished, metrics, winners, events = transition(worlds, actions, keys)
        finished, metrics, winners, events = jax.device_get((finished, metrics, winners, events))
        snapshot(worlds)
        for i, record in enumerate(records):
            if done[i]:
                continue
            frame = record['frames'][-1]
            if frame['tick'] != tick:
                raise ValueError('Discontinuous replay frames')
            totals[i] += metrics[i].astype(np.int64)
            for name, observed in events.items():
                record['events'].extend([tick, int(p), name] for p in np.flatnonzero(observed[i]))
            if finished[i]:
                defender_seat = record['defender_seat']
                courier_seat = 1 - defender_seat
                by_seat = [dict(zip(env.METRICS, map(int, totals[i, p]))) for p in (0, 1)]
                record['result'] = dict(
                    tick=tick, winner=int(winners[i]), captures=by_seat[courier_seat]['captures'],
                    defender_kills=by_seat[defender_seat]['kills'],
                    courier_kills=by_seat[courier_seat]['kills'], by_seat=by_seat,
                )
        done |= finished
        if tick == 1 or tick % 200 == 0 or done.all():
            print(json.dumps(dict(tick=tick, completed=int(done.sum()))), flush=True)
        if done.all():
            break

    if not done.all():
        raise ValueError('Incomplete carry game')
    for record in records:
        result = record['result']
        if len(record['frames']) != result['tick'] + 1 or result['winner'] not in (-1, 0, 1):
            raise ValueError('Invalid terminal record')
    level = next(x for x in json.loads(Path('web/assets/maps/levels.json').read_text())
                 if x.get('qqt_id') == 806)
    document = dict(
        schema='qqt.full-jax-carry-replay/v1',
        generated_utc=datetime.now(timezone.utc).isoformat(),
        map=dict(id=806, width=level['w'], height=level['h'], bases=level['bun_bases'],
                 walls=[i for i, wall in enumerate(level['wall']) if wall]),
        protocol=dict(tick_hz=tick_hz, max_ticks=max_ticks, seeds=SEEDS,
                      backend=jax.default_backend(), precision='float32', temperature=1.0),
        models=dict(defender=dict(sha256=meta['actor_sha256'], update=meta['update'],
                                  parent=meta['initial_actor']),
                    courier=courier_model),
        replays=records,
    )
    encoded = (json.dumps(document, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
               + '\n').encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix('.tmp')
    tmp.write_bytes(encoded)
    tmp.replace(args.out)
    print(json.dumps(dict(file=str(args.out), bytes=len(encoded),
                          sha256=hashlib.sha256(encoded).hexdigest(),
                          results=[r['result'] for r in records])), flush=True)


if __name__ == '__main__':
    main()
