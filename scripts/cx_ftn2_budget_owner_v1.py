"""Accept the sealed 160-update trial, then continue the authorized fixed budget."""
import fcntl
import json
import os
import signal
import shutil
import sys
import subprocess
import time
from pathlib import Path

from scripts.cx_ftn2_switch_v1 import audit, held
from scripts.cx_ftn2_budget_policy_v1 import point_totals, trend_stop
from scripts.model_launch import command, manifest, public_project
from scripts.population_run_state import atomic_json, sha

BASE = Path('runs/cc_train_20261105/training')
ROOT = BASE / 'ftn2_20261106'
OUT = ROOT / 'L2_FTN2'
PARENT = BASE / 'protocols/CX_L2_FTN2_20261106_V1.json'
PROTOCOL = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / 'protocols/CX_L2_FTN2_20261106_V2.json'
TOKEN = 'CX-FTN2-20261106-FIXED1200'
POINTS = Path('runs/cc_train_20261105/eval_strong_v1/points')
STOPS = ('USER_STOP', 'USER_STOP.json', 'USER_PAUSE.json', 'STOP_FOR_REBOOT.json')


def read(path):
    return json.loads(Path(path).read_text())


def note(event, **fields):
    row = dict(event=event, owner='CX', pid=os.getpid(), unix=time.time(), **fields)
    atomic_json(ROOT / 'CONTINUATION_STATUS.json', row)
    with (ROOT / 'continuation_events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    if event in ('SUCCESSOR_ACCEPTED', 'SUCCESSOR_FIRST_UPDATE', 'WORKER_STARTED', 'COMPLETED_FIXED1200', 'TECHNICAL_BLOCKER', 'STOPPED_AT_SAVED_BOUNDARY'):
        with Path('reports/cx_training_followup_20261106.md').open('a') as f:
            f.write('\nFTN2固定预算事件：`' + json.dumps(row, ensure_ascii=False) + '`\n')
        notify = shutil.which('jdme-notify')
        if notify:
            try:
                subprocess.run([notify, 'send', f'FTN2 {event}: {fields}'], capture_output=True, timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                pass
    print(json.dumps(row), flush=True)


def stopped():
    return any((d / n).exists() for d in (BASE, OUT) for n in STOPS)


def regression():
    """Do not skip a pending committed point to manufacture two declines."""
    rows = []
    for cp in sorted(OUT.glob('u*/COMMITTED.json')):
        u = int(cp.parent.name[1:])
        if u < 19342:
            continue
        point = POINTS / f'L2_FTN2_u{u:06d}__seats01.json'
        if not point.exists():
            break
        row = point_totals(read(point))
        row.update(point=str(point), point_sha256=sha(point))
        rows.append(row)
    decision = dict(status='STOP_TWO_CONSECUTIVE_JOINT_DECLINES' if trend_stop(rows) else 'CONTINUE_FIXED_BUDGET',
                    completed_contiguous_points=rows, unix=time.time())
    atomic_json(ROOT / 'REGRESSION_DECISION.json', decision)
    return decision


def boundary_stop(reason):
    marker = OUT / 'USER_STOP.json'
    if not marker.exists():
        atomic_json(marker, dict(owner=TOKEN, reason=reason, unix=time.time()))


def run(phase, *flags, timeout, allow_trend=False):
    p = read(PROTOCOL)
    cfg = manifest(dict(module='scripts.cx_ftn2_train_v1', gpu=4,
                        args=[str(PROTOCOL), 'L2_FTN2', str(p['seed'])] + list(flags)))
    with (OUT / f'{phase}.log').open('a') as log:
        child = subprocess.Popen(['taskset', '-c', '48-55'] + command('training', cfg),
                                 cwd=public_project(), stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    atomic_json(OUT / 'SUPERVISOR.json', dict(status='RUNNING', owner='CX', pid=os.getpid(),
                worker_pid=child.pid, gpu=4, cpus='48-55', protocol=str(PROTOCOL), phase=phase,
                target_update=20502, fixed_budget=1200, unix=time.time()))
    note('WORKER_STARTED', phase=phase, worker_pid=child.pid, config=cfg, target_update=20502)
    end = time.monotonic() + timeout
    next_check = 0
    while child.poll() is None:
        if time.monotonic() > end:
            boundary_stop(f'technical timeout {phase}')
            try:
                child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
            raise TimeoutError(f'{phase}: exceeded {timeout}s')
        if time.monotonic() >= next_check:
            try:
                row = audit()
            except BaseException:
                boundary_stop('technical reward/finite audit failure')
                raise
            if allow_trend and regression()['status'].startswith('STOP_'):
                boundary_stop('two consecutive full strong-eval declines in net wins AND deliveries')
            next_check = time.monotonic() + 60
            first = ROOT / 'FIXED1200_FIRST_UPDATE.json'
            if phase == 'fixed1200_formal' and row['new_update'] > 19462 and not first.exists():
                first_row = next(json.loads(line) for line in (OUT / 'updates.jsonl').read_text().splitlines()
                                 if json.loads(line)['new_update'] == 19463)
                atomic_json(first, dict(status='PASS_REAL_FINITE_SUCCESSOR_UPDATE',
                            worker_pid=child.pid, config=cfg, protocol=str(PROTOCOL),
                            protocol_sha256=sha(PROTOCOL), update=first_row,
                            reward_receipt=str(OUT / f'REWARD_RECEIPT_{child.pid}.json'), unix=time.time()))
                note('SUCCESSOR_FIRST_UPDATE', worker_pid=child.pid, update=first_row['new_update'],
                     finite=True, envsteps_s=first_row['env_per_second'])
            note('WORKER_RUNNING', phase=phase, worker_pid=child.pid, config=cfg,
                 update=row['new_update'], finite=row['finite'], envsteps_s=row['env_per_second'])
        time.sleep(10)
    assert child.returncode == 0, (phase, child.returncode)
    note('WORKER_EXIT', phase=phase, code=child.returncode)


def main():
    guard = (ROOT / 'continuation.lock').open('a+')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = read(PROTOCOL)
    for file, expected in p['source_sha256'].items():
        assert sha(file) == expected, file
    old = read(PARENT)
    assert p['arms'] == old['arms'] and p['common'] == old['common']
    assert p['namespace'] == old['namespace'] and p['seed'] == old['seed']
    assert p['chunk_start_update'] + p['chunk_updates'] == 20502
    assert sha(PARENT) in p['compatible_checkpoint_protocol_sha256']
    note('WAIT_PARENT_TECHNICAL_ACCEPTANCE', protocol=str(PROTOCOL), protocol_sha256=sha(PROTOCOL),
         parent_target=19462, cumulative_target=20502, minimum_updates=1200)
    deadline = time.monotonic() + 7200
    while True:
        state = read(ROOT / 'SWITCH_STATUS.json')
        if state['stage'] in ('FAILED', 'FALLBACK_RELAUNCHED'):
            note('BLOCKED_PARENT_FAILURE', parent=state)
            return
        if stopped():
            note('STOPPED_BY_EXISTING_MARKER')
            return
        if state['stage'] == 'COMPLETED_BOUNDED':
            break
        assert time.monotonic() < deadline, 'parent did not finish within finite wait budget'
        time.sleep(10)
    # Parent's state transition precedes lock release by a few instructions.
    with (ROOT / 'switch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not held(BASE / 'GPU4.lock') and not held(OUT / 'worker.lock')
        row = audit()
        assert row['new_update'] == 19462 and row['finite']
        restore = read(OUT / 'RESTORE_u019462.json')
        assert restore['status'] == 'PASS' and restore['full_Actor_Critic_Adam_env_aux_RNG_league_exact']
        atomic_json(ROOT / 'PARENT_FIXED_BUDGET_ACCEPTANCE.json', dict(status='PASS',
                    update=19462, restore=restore, reward_audit_sha256=sha(ROOT / 'REWARD_ROLLOUT_ACCEPTANCE.json'),
                    parent_protocol_sha256=sha(PARENT), successor_protocol_sha256=sha(PROTOCOL), unix=time.time()))
    # Same exact worker/reward/seed/scale: reuse the immutable smoke evidence;
    # prove full-next-step identity under the compatible successor protocol before training.
    run('fixed1200_entry_restore', '--verify-restore', timeout=1200)
    assert read(OUT / 'RESTORE_u019462.json')['full_Actor_Critic_Adam_env_aux_RNG_league_exact']
    note('SUCCESSOR_ACCEPTED', from_update=19462, to_update=20502, protocol=str(PROTOCOL))
    run('fixed1200_formal', '--until', '20502', timeout=36000, allow_trend=True)
    row = audit()
    if stopped():
        note('STOPPED_AT_SAVED_BOUNDARY', update=row['new_update'], marker=read(OUT / 'USER_STOP.json') if (OUT / 'USER_STOP.json').exists() else 'global')
        atomic_json(OUT / 'SUPERVISOR.json', dict(status='STOPPED', owner='CX', pid=os.getpid(),
                    latest=row['new_update'], protocol=str(PROTOCOL), unix=time.time()))
        return
    assert row['new_update'] == 20502
    run('fixed1200_final_restore', '--verify-restore', timeout=1200)
    assert read(OUT / 'RESTORE_u020502.json')['full_Actor_Critic_Adam_env_aux_RNG_league_exact']
    atomic_json(OUT / 'SUPERVISOR.json', dict(status='COMPLETED_FIXED1200', owner='CX', pid=os.getpid(),
                latest=20502, protocol=str(PROTOCOL), unix=time.time()))
    note('COMPLETED_FIXED1200', update=20502, updates=1200,
         fresh_ticks=1200 * 262144, adam_steps=1200 * 32, final_restore='PASS')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        note('TECHNICAL_BLOCKER', error=f'{type(exc).__name__}: {exc}')
        raise
