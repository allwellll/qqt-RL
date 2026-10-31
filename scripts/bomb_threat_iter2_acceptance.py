"""Read-only acceptance of the three authorized ITER2 initial launches."""
import json
import pickle
import time
from pathlib import Path

import numpy as np

from scripts.population_run_state import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/bomb_threat_v1'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def refresh():
    config = read(BASE / 'iter2/TRAINING_PROTOCOL.json')
    original = ROOT / config['policy_source']['checkpoint'] / 'policy.pt'
    assert sha(original) == config['policy_source']['policy_sha256']
    with original.open('rb') as stream:
        original_actor = pickle.load(stream)
    arms = []
    for slot, arm in ((4, 'A1'), (7, 'A2'), (1, 'A3')):
        run = f'bt_slot{slot:02d}_c000013'
        out = BASE / 'formal' / run
        receipts = {name: read(out / (name + '.json')) for name in ('INIT', 'WARMUP_GATE', 'SMOKE', 'RESTORE')}
        rows = [json.loads(line) for line in (out / 'updates.jsonl').read_text().splitlines()] if (out / 'updates.jsonl').exists() else []
        formal = [row for row in rows if row['new_update'] > 2]
        ready = bool(formal) and receipts['INIT'].get('status') == 'PASS' and all(receipts[name].get('passed') for name in ('WARMUP_GATE', 'SMOKE', 'RESTORE'))
        record = dict(slot=slot, arm=arm, run=run, receipts={name: str((out / (name + '.json')).relative_to(ROOT)) for name in receipts},
                      status='FORMAL_RUNNING_VERIFIED' if ready else 'TECHNICAL_STAGE_RUNNING',
                      latest_update=rows[-1]['new_update'] if rows else 0)
        if ready:
            assert all(row['finite'] and all(row['update_l2'][key] > 0 for key in ('actor', 'critic')) for row in formal)
            warm = read(out / 'WARMUP_PROGRESS.json')['rows']
            assert len(warm) == 20 and all(row['update_l2']['actor'] == 0 and row['Actor_frozen_exact'] for row in warm)
            assert receipts['INIT']['actor_logits_first8_bitwise']
            assert receipts['RESTORE']['fresh_process'] and receipts['RESTORE']['next_step_exact']
            assert all(receipts['RESTORE'][key] for key in ('Adam_exact', 'full_env_aux_RNG_exact', 'ring_exact'))
            commits = [path for path in sorted(out.glob('u*/COMMITTED.json')) if (path.parent / 'state.pkl').exists()]
            if not commits:
                raise RuntimeError('Active run has no retained atomic state: ' + run)
            cp = commits[-1].parent
            cm = read(cp / 'COMMITTED.json')
            assert sha(cp / 'state.pkl') == cm['state_sha256'] and sha(cp / 'policy.pt') == cm['policy_sha256']
            with (cp / 'policy.pt').open('rb') as stream:
                actor = pickle.load(stream)
            deltas = {name: float(np.sqrt(sum(np.sum((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2)
                       for a, b in zip(actor['heads'][name], original_actor['heads'][name])))) for name in ('wb', 'wm')}
            assert all(value > 0 for value in deltas.values())
            record.update(first_post_restore_update=formal[0], latest_update_metrics=formal[-1],
                          latest_atomic=cm, commit_sha256=sha(cp / 'COMMITTED.json'),
                          latest_atomic_includes_post_restore_update=cm['update'] > 2,
                          ability_head_L2_vs_u960=deltas['wb'], movement_head_L2_vs_u960=deltas['wm'],
                          warmup_environment_steps=20 * 262144, technical_smoke_environment_steps=2 * 262144,
                          post_restore_environment_steps=len(formal) * 262144,
                          warmup_critic_Adam_steps=20 * 32, post_restore_Actor_Adam_steps=len(formal) * 32,
                          sampled_steady_env_per_second=262144 / formal[-1]['cycle_seconds'])
        arms.append(record)
    result = dict(status='ALL_FORMAL_RUNNING_VERIFIED' if all(arm['status'] == 'FORMAL_RUNNING_VERIFIED' for arm in arms) else 'PARTIAL_FORMAL_RUNNING_VERIFIED',
                  arms=arms, source_policy_sha256=config['policy_source']['policy_sha256'],
                  ability_head_evidence='parameter change from source Actor; not per-phase gradient attribution',
                  training_worker_signals=0, old_queues_resumed=False, BEST_changed=False, unix=time.time())
    atomic_json(BASE / 'iter2/FIRST_FORMAL_ACCEPTANCE.json', result)
    atomic_json(BASE / 'iter2/EXECUTION_CURRENT.json', dict(status=result['status'], source_block_resolved=True,
                acceptance='runs/bomb_threat_v1/iter2/FIRST_FORMAL_ACCEPTANCE.json', unix=time.time()))
    return result


if __name__ == '__main__':
    result = refresh()
    print(result['status'], [(arm['arm'], arm['latest_update']) for arm in result['arms']])
