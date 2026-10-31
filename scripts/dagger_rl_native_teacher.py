"""CPU diagnostic adapter of the frozen item forecast to public native fields.

Only the legal-mask provider is supplied by production JS. This does not claim
JS/JAX engine parity, and the adapted teacher never enters deployment/training.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                  JAX_PLATFORM_NAME='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import inspect
import json
import sys
from typing import NamedTuple

import jax
import jax.numpy as j
import numpy as np
from scripts import dagger_rl_oracle as O


def _adapt():
    source = inspect.getsource(O.single)
    assert source.count('def single(st,ax):') == 1
    assert source.count('    mm,bm=E.env.legal_mask(st)\n') == 1
    source = source.replace('def single(st,ax):', 'def single_with_masks(st,ax,mm,bm):')
    source = source.replace('    mm,bm=E.env.legal_mask(st)\n', '')
    scope = dict(vars(O))
    exec(compile(source, '<native-mask-diagnostic-adapter>', 'exec'), scope)
    function = scope['single_with_masks']
    function.adapted_source = source
    return function


single_with_masks = _adapt()


class Core(NamedTuple):
    pushable: object
    blast_linger: object


class PublicState(NamedTuple):
    pos: object
    wall: object
    brick: object
    core: object
    fuse: object
    alive: object
    bun_respawn: object
    bun_spawn_pos: object
    trapped: object
    field_item: object
    field_owner: object
    field_armed: object
    bun_carried: object
    move_status: object
    held_item: object


@jax.jit
def query(state, aux, move_mask, ability_mask):
    mask, diagnostic = single_with_masks(state, aux, move_mask, ability_mask)
    pos = state.pos[0]
    own = (state.field_item == 1) & (state.field_owner == 0)
    near = own & (j.abs(j.indices((13, 15))[0] + .5 - pos[0]) +
                  j.abs(j.indices((13, 15))[1] + .5 - pos[1]) <= 2.)
    has = near.any(); idx = j.argmax(near.reshape(-1))
    field = j.where(has, j.array([idx // 15, idx % 15]), pos.astype(j.int32))
    plans = jax.vmap(lambda direction: O.one_plan(
        state, aux['carry_items']['distance'][0], direction, field, ~has))(j.arange(4))
    return mask, diagnostic, field, plans


def answer(payload):
    fields = payload['state']
    core = Core(j.asarray(fields['pushable'], bool).reshape(13,15),
                j.asarray(fields['blast_linger'], j.int32).reshape(13,15))
    arrays = {}
    grid = {'wall','brick','fuse','field_item','field_owner','field_armed'}
    boolean = {'wall','brick','alive','field_armed'}
    floating = {'pos','bun_spawn_pos'}
    for name in PublicState._fields:
        if name == 'core': continue
        value = j.asarray(fields[name], bool if name in boolean else j.float32 if name in floating else j.int32)
        arrays[name] = value.reshape(13,15) if name in grid else value
    state = PublicState(core=core, **arrays)
    distance = j.asarray(payload['distance'], j.int32).reshape(13,15)
    aux = {'carry_items': {'distance': j.stack([distance,distance]),
                          'ledger': {'custody': j.asarray([payload['real_custody'],False])}},
           'terminal': j.asarray(payload['terminal'])}
    mask, diagnostic, field, plans = query(state, aux,
        j.asarray(payload['move_mask'],bool), j.asarray(payload['ability_mask'],bool))
    valid, gain, first, ticks, end = map(np.asarray, plans)
    return dict(mask=np.asarray(mask).tolist(), diagnostic=np.asarray(diagnostic).tolist(),
                field=np.asarray(field).tolist(), plans=[dict(direction=d, valid=bool(valid[d]),
                    gain=float(gain[d]), first=int(first[d]), ticks=int(ticks[d]),
                    end=end[d].tolist()) for d in range(4)],
                scope='original JAX item forecast/eligibility with production native legal masks; not an engine parity claim')


if __name__ == '__main__':
    for line in sys.stdin:
        try:
            print(json.dumps(answer(json.loads(line))), flush=True)
        except Exception as error:
            print(json.dumps({'error': str(error)}), flush=True)
