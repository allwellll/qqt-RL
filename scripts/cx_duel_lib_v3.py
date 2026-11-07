"""Private dynamic own/strong snapshot league and explicit task reward receipt."""
import json,os
from pathlib import Path
import jax
import jax.numpy as j
from scripts import cc_league2_lib as parent
from scripts.population_run_state import atomic_json,sha

SCHEMA,entry,actor,publish=parent.SCHEMA,parent.entry,parent.actor,parent.publish
HEALTH='runs/cc_train_20261105/training/formal/L2_S05g/u019302'
J_SOURCE='runs/cc_train_20261105/training/formal/L2_S05j/u019462'
I_SOURCE='runs/cc_train_20261105/training/formal/L2_S05i/u019382'


def load(family,index=None):
    # G manifest advances on every COMMITTED publish; generation refresh40.
    if family=='G':return parent.load('G',index)
    peer=parent.peer_manifest(os.environ.get('CC_LEAGUE2_ARM','L2_DUEL'),json.loads(os.environ['CC_LEAGUE2_PEERS']),int(index))
    if peer['peer_arm'] is None:
        peer.update(current=entry(J_SOURCE),ring=[entry(I_SOURCE)],fallback='strong j/i when no active peer')
    root=Path(os.environ['CC_SNAPSHOT_ROOT'])/'league_snapshots'
    atomic_json(root/'C.json',peer)
    return peer


def remap_kinds(*args):return parent.remap_kinds(*args)
def progression(*args):return False


def fork(mixed,keys,bs,ba,spawns,seed):
    n=len(keys);b=mixed.bun_backend();physical=j.asarray(mixed.seats(n,seed))
    kinds=mixed.base.allocation(n,seed,mixed.WEIGHTS)
    def build(ks):
        st,ax=b.fresh(ks,bs,ba,j.zeros(n,bool),spawns,physical)
        meta=b.metadata(st,kinds,j.zeros(n,bool),physical,ax['r1_attribute_highwater'])
        return st,mixed.migrate_aux(dict(ax,combat_mix=dict(meta,opponent_generation=j.zeros(n,j.int32)))),ks
    return jax.jit(build)(keys)


def receipt(out,spec,protocol):
    from scripts import cx_duel_reward_v1 as R
    from scripts import r3_items_reward as full
    from scripts.cx_duel_mixed_v3 import CATALOG
    atomic_json(out/'REWARD_RECEIPT.json',dict(status='LOADED_DUEL_NATIVE_ONE_LIFE_V1',
        version=R.VERSION,module='scripts.cx_duel_reward_v1',source_sha256=sha('scripts/cx_duel_reward_v1.py'),
        full_reward_source_sha256=sha('scripts/r3_items_reward.py'),full_cuda_overlay=bool(getattr(full,'CUDA_OVERLAY',False)),
        mode='global38 last column: full0/duel1; full mode0 export remains compatible',
        mix=protocol["duel"]["mix"],kill=R.KILL,death=-R.DEATH,self_extra=-R.SELF_EXTRA,horizon=R.HORIZON,
        trap_enter_reward=0,bomb_count_reward=0,idle_penalty=0,duel_transport_reward=0,
        full_reward_unchanged=True,sil_full_only=spec['sil_coef'],duel_dagger=0,mask=False,
        scenarios=str(CATALOG),scenarios_sha256=sha(CATALOG),
        seed=protocol['seed'],weights=spec['opponent_weights'],protocol_id=protocol['id'],
        source_checkpoint=HEALTH,full_state_restore_fields=['Actor','Critic','Adam','env','aux','RNG','league'],
        physical_device=int(os.environ["JAXBOMB_DEVICE_OVERRIDE"])))
