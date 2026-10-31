"""Consume the 21:10 decision inside the existing sole BOMB owner."""
import copy
import json
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/bomb_threat_v1'
OUT = ROOT / 'runs/slot01_copy07_gpu3_20261031'
MODULE = 'scripts.bomb_threat_slot01_copy07_train'
TARGET_SEED = 2026312111


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def paused():
    return any((b / n).exists() for b in (BASE, ROOT / 'runs/iter_pbt_v1')
               for n in ('USER_STOP.json', 'USER_PAUSE.json', 'STOP_FOR_REBOOT.json'))


def verify(source):
    cp = ROOT / source['checkpoint']
    committed = read(cp / 'COMMITTED.json')
    assert source['passed'] and committed['passed']
    for key in ('checkpoint', 'update', 'state_sha256', 'policy_sha256'):
        assert committed[key] == source[key]
    for file, key in [('state.pkl', 'state_sha256'), ('policy.pt', 'policy_sha256')]:
        assert sha(cp / file) == source[key]
    token = cp.parent / f"NEXT_STEP_u{source['update']:06d}.json"
    record = read(token)
    assert all(record.get(k) for k in ('digest', 'Adam_digest', 'full_environment_digest', 'ring_digest'))
    return token


def select_latest_multiple40():
    candidates = []
    for p in (BASE / 'formal').glob('bt_slot07_c*/u*/COMMITTED.json'):
        if (int(p.parent.name[1:]) % 40 or not (p.parent / 'state.pkl').exists()
                or not (p.parent.parent / f'NEXT_STEP_{p.parent.name}.json').exists()):
            continue
        source = read(p)
        if source.get('passed'):
            candidates.append(source)
    assert candidates, 'no retained complete slot07 multiple40 source'
    source = max(candidates, key=lambda x: (x['update'], x['checkpoint']))
    verify(source)
    return source


def protocol(source, chunk, rekey):
    donor_run = Path(source['checkpoint']).parent.name
    donor_path = BASE / 'protocols' / f'{donor_run}.json'
    original = read(donor_path)
    value = copy.deepcopy(original)
    recipe = copy.deepcopy(original['arms'][donor_run])
    assert recipe['iter2_arm'] == 'A3SR'
    name = f'bt_slot01_c{chunk:06d}'
    value.update(arms={name: recipe}, sources={str(TARGET_SEED): source},
        continuation_source=source, chunk_start_update=source['update'],
        environment_steps_per_run=(source['update'] + original['chunk_updates']) * 262144,
        slot=1, chunk=chunk, r1_seed=TARGET_SEED, round14_rekey=False,
        worker_module=MODULE, prewarm_acceptance=None,
        slot01_copy07_rekey=rekey,
        slot01_copy07_authorization='runs/slot01_copy07_gpu3_20261031/TASK_CONSUMED.json',
        pbt_lineage=dict(source=source, target_seed=TARGET_SEED,
            copied=['Actor', 'Critic', 'Adam', 'env', 'aux', 'combat_ring', 'combat_ring_cursor'],
            donor_rng='restored and next-step verified before declared independent RNG rekey',
            RNG_rekeyed=rekey, reward_ledger_reset=False, critic_warmup_updates=0),
        source_protocol=str(donor_path.relative_to(ROOT)), source_protocol_sha256=sha(donor_path))
    value['source_sha256']['scripts/bomb_threat_slot01_copy07_train.py'] = sha(ROOT / 'scripts/bomb_threat_slot01_copy07_train.py')
    assert all(sha(ROOT / f) == h for f, h in value['source_sha256'].items())
    assert value['common'] == original['common'] and value['arms'][name] == original['arms'][donor_run]
    path = BASE / 'protocols' / f'{name}.json'
    if path.exists():
        assert read(path) == value, 'refuse an inconsistent frozen successor'
    else:
        atomic_json(path, value)
    return path


def apply_switch(slot, entry, live, next_chunk):
    if slot != 1 or not entry or entry['run'] in live or paused():
        return None
    auth = read(OUT / 'TASK_CONSUMED.json', {})
    if auth.get('status') != 'AUTHORIZED':
        return None
    assert sha(ROOT / auth['instruction']) == auth['instruction_sha256']
    applied = entry.get('slot01_copy07_applied', False)
    if not applied and entry['run'] != auth['boundary_run']:
        raise RuntimeError('slot01 boundary changed without authorized handoff')
    done = read(BASE / 'formal' / entry['run'] / 'COMPLETE.json')
    if not done:
        return None
    old_protocol = read(BASE / 'protocols' / f"{entry['run']}.json")
    assert done['update'] == old_protocol['chunk_start_update'] + old_protocol['chunk_updates']
    verify(done)
    from scripts.bomb_threat_round15 import pin
    pin([done['checkpoint']], '21:10 slot01 old lineage complete boundary; preserve exact recovery state')
    source = done if applied else select_latest_multiple40()
    pin([source['checkpoint']], '21:10 slot07 full-state donor for independent slot01 fork')
    path = protocol(source, next_chunk, rekey=not applied)
    if not applied:
        gate_path = BASE / 'iter2/r1_attack_20261030/stop2255/slot01.json'
        request = read(gate_path)
        assert request['boundary_run'] == entry['run']
        atomic_json(OUT / 'SLOT01_STOPPED.json', dict(status='COMPLETE_CHUNK_THEN_STOPPED',
            old_run=entry['run'], last_atomic=done, training_worker_exited=True,
            training_worker_signals=0, old_lineage_successors_blocked=True, unix=time.time()))
        atomic_json(OUT / 'COPY_HANDOFF.json', dict(status='PROTOCOL_READY', source=source,
            stopped=done, protocol=str(path.relative_to(ROOT)), protocol_sha256=sha(path),
            target_seed=TARGET_SEED, reward_and_recipe_unchanged=True, unix=time.time()))
        atomic_json(gate_path, dict(request, status='CONSUMED_REPLACED_AFTER_COMPLETE',
            receipt='runs/slot01_copy07_gpu3_20261031/SLOT01_STOPPED.json', unix=time.time()))
        from scripts.iter_pbt_owner import notify
        notify('SLOT01_COPY07_BOUNDARY_20261031',
            f"slot01旧A3 u{done['update']}完整保存停止；最新slot07 u{source['update']}全state复制，独立seed{TARGET_SEED}，原配方先smoke/restore再正式。")
    return dict(entry, run=path.stem, chunk=next_chunk, source=source,
        iter2_arm='A3SR', role='SLOT01_EXACT_SLOT07_PBT', slot01_copy07_applied=True,
        authorized_shaping_override=True,
        authorized_shaping_override_basis='runs/slot01_copy07_gpu3_20261031/TASK_CONSUMED.json',
        supersedes=entry['run'])
