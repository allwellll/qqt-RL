"""Persisted per-slot retirement, consumed before every owner dispatch."""
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha
from scripts.bomb_threat_round15 import read, pin

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/bomb_threat_v1'
OUT = BASE / 'iter2/r1_attack_20261030/stop2255'


def consume(slot, entry, live):
    request = read(OUT / f'slot{slot:02d}.json', {})
    if request.get('status') != 'AUTHORIZED':
        return False
    assert sha(ROOT / request['instruction']) == request['instruction_sha256']
    name = entry['run'] if entry else None
    running = name in live if name else False
    done = read(BASE / 'formal' / name / 'COMPLETE.json', {}) if name else {}
    boundary = request['boundary_run']
    blocked = not running and (name != boundary or bool(done) or request['mode'] == 'RETIRE_FAILED')
    value = dict(status='BLOCKING_SUCCESSORS' if blocked else 'ALLOW_CURRENT_CHUNK_ONLY',
        slot=slot, boundary_run=boundary, observed_run=name, worker_alive=running,
        instruction=request['instruction'], training_worker_signals=0,
        block_all_other_dispatch=False, block_this_slot_successors=blocked,
        dispatch_scope='this_slot_only', unix=time.time())
    if done and name == boundary:
        receipt = OUT / f'slot{slot:02d}_STOPPED.json'
        accepted = read(receipt, {})
        if accepted.get('last_atomic') == done:
            value['last_atomic'] = done
            atomic_json(OUT / f'slot{slot:02d}_CONSUMED.json', value)
            return blocked
        protocol = read(BASE / 'protocols' / f'{name}.json')
        assert done['passed'] and done['update'] == protocol['chunk_start_update'] + protocol['chunk_updates']
        cp = ROOT / done['checkpoint']
        assert sha(cp / 'state.pkl') == done['state_sha256']
        assert sha(cp / 'policy.pt') == done['policy_sha256']
        assert read(cp / 'COMMITTED.json')['state_sha256'] == done['state_sha256']
        pin([done['checkpoint']], '22:55 final complete R1 state must be preserved')
        value['last_atomic'] = done
        if not receipt.exists():
            atomic_json(receipt, dict(value, status='COMPLETE_CHUNK_THEN_RETIRED'))
            from scripts.iter_pbt_owner import notify
            notify(f'R1_STOP2255_{name}', f'R1 slot{slot:02d} 完整chunk u{done["update"]} 已原子保存、SHA核验；禁止后继，完整state保留，资源让给现有R2。')
    atomic_json(OUT / f'slot{slot:02d}_CONSUMED.json', value)
    return blocked
