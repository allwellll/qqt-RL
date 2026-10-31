"""Consume the explicit 18:05 authorization at completed R1 chunk boundaries."""
import copy
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha
from scripts.bomb_threat_round15 import read, pin

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/bomb_threat_v1'
OUT = BASE / 'iter2/r1_attack_20261030/continuation_1805'
ITEMS = BASE / 'iter2/r1_attack_20261030/items_v2'
MODULE = 'scripts.bomb_threat_r1_continue_train'
DECISION = '.hermes/DECISION_r1_continue_20261030_1805.md'


def paused():
    return any((base / name).exists() for base in (BASE, ROOT / 'runs/iter_pbt_v1')
               for name in ('USER_STOP.json', 'USER_PAUSE.json', 'STOP_FOR_REBOOT.json'))


def apply_continue(slot, entry, done, chunk):
    authorization = read(OUT / 'AUTHORIZATION.json', {})
    if paused() or authorization.get('status') != 'AUTHORIZED':
        return None
    if slot not in (0, 3, 4, 7) or not entry.get('r1_items_applied') or entry.get('r1_smoke_fallback'):
        return None
    assert authorization['decision_sha256'] == sha(ROOT / DECISION)
    old = BASE / 'formal' / entry['run']
    assert read(old / 'COMPLETE.json') == done and done['passed']
    cp = ROOT / done['checkpoint']
    committed = read(cp / 'COMMITTED.json')
    assert committed['passed'] and all(committed[key] == done[key]
        for key in ('checkpoint', 'update', 'state_sha256', 'policy_sha256'))
    assert sha(cp / 'state.pkl') == done['state_sha256']
    assert sha(cp / 'policy.pt') == done['policy_sha256']
    assert (old / f"NEXT_STEP_u{done['update']:06d}.json").exists()
    prior = read(BASE / 'protocols' / f"{entry['run']}.json")
    assert done['update'] == prior['chunk_start_update'] + prior['chunk_updates']
    successor = f'bt_slot{slot:02d}_c{chunk:06d}'
    protocol = copy.deepcopy(prior)
    recipe = protocol['arms'][entry['run']]
    protocol.update(arms={successor: recipe}, continuation_source=done,
        sources={str(prior['r1_seed']): done}, chunk_start_update=done['update'],
        chunk_updates=640, environment_steps_per_run=(done['update'] + 640) * 262144,
        chunk=chunk, worker_module=MODULE, prewarm_acceptance=None,
        r1_same_version_continuation=True,
        r1_continuation_authorization=str((OUT / 'AUTHORIZATION.json').relative_to(ROOT)),
        pbt_lineage=dict(source=done, Actor_Critic_Adam_env_aux_RNG_ring_exact=True,
                         RNG_rekeyed=False, reward_ledger_reset=False))
    protocol['source_sha256']['scripts/bomb_threat_r1_continue_train.py'] = sha(ROOT / 'scripts/bomb_threat_r1_continue_train.py')
    assert all(sha(ROOT / f) == h for f, h in protocol['source_sha256'].items())
    path = BASE / 'protocols' / f'{successor}.json'
    if path.exists():
        assert read(path) == protocol, 'refuse protocol overwrite'
    else:
        atomic_json(path, protocol)
    handoff = dict(status='BOUNDARY_HANDOFF', previous_run=entry['run'], successor=successor,
        slot=slot, source=done, authorization=str((OUT / 'AUTHORIZATION.json').relative_to(ROOT)),
        no_worker_signals=True, unix=time.time())
    atomic_json(ITEMS / 'handoffs' / f'{successor}.json', handoff)
    atomic_json(OUT / 'handoffs' / f'{successor}.json', handoff)
    pin([str((BASE / 'formal' / successor / f'u{u:06d}').relative_to(ROOT))
         for u in range(done['update'] + 40, done['update'] + 641, 40)], 'R1 authorized same-version continuation')
    return dict(entry, run=successor, chunk=chunk, source=done,
                r1_same_version_continuation=True, supersedes=entry['run'])
