"""Explicit per-chunk reset-distribution migration, retaining learned state."""
import copy,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.bun_spawn_side import VERSION

ROOT=Path('runs/v2_ownside_selection_20261031')

def enabled():return (ROOT/'TRAINING_MIGRATION_AUTHORIZED.json').exists()

def configure(family,arm,entry,protocol):
    if not enabled():return entry,protocol
    p=copy.deepcopy(protocol);e=dict(entry)
    migration=p.get('spawn_side_version')!=VERSION
    name=e['run'];before=name
    prefix,chunk=name.rsplit('_c',1)
    if not prefix.endswith('_ownside'):prefix+='_ownside'
    name=prefix+'_c'+chunk
    recipe=next(iter(p['arms'].values()))
    p.update(arms={name:recipe},spawn_side_version=VERSION,spawn_side_migration=migration,
        spawn_manifest=str(ROOT/'train_spawns.json'),spawn_manifest_sha256=sha(ROOT/'train_spawns.json'),
        spawn_side_protocol=str(ROOT/'SPAWN_PROTOCOL.json'))
    for f,h in list(p.get('source_sha256',{}).items()):
        if sha(Path(f))!=h:
            assert f in ('scripts/r3_items_eval_owner.py','scripts/bun_native_v2_eval_owner.py','scripts/bun_native_v2_diagnostic.py'),f
            p['source_sha256'][f]=sha(Path(f))
    for f in ('scripts/bun_spawn_side.py','scripts/bun_spawn_side_train.py'):
        p.setdefault('source_sha256',{})[f]=sha(Path(f))
    e.update(run=name,worker_module='scripts.bun_spawn_side_train',spawn_side_version=VERSION,
        performance_restore_pending=False)
    if family=='R2':e['run_prefix']=prefix
    out=Path(p['namespace']).parent/'protocols'/f'{name}.json'
    e['protocol']=str(out)
    atomic_json(ROOT/'training_handoffs'/f'{name}.json',dict(status='REGISTERED_AT_COMPLETE_CHUNK',family=family,arm=arm,
        old_next_name=before,entry=e,migration=migration,source=p['continuation_source'],
        Actor_Critic_Adam_ring='exact retained',env_aux_RNG='rebuild at migration; exact continuation thereafter',unix=time.time()))
    return e,p
