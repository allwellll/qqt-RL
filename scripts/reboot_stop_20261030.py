"""Publication-gated atomic stop of this worktree's identified processes only."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from scripts.population_run_state import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'runs/bomb_threat_v1'
O = B / 'publish_best_20261030'
TASK = '.hermes/TASK_stop_for_reboot_20261030.md'
IMMEDIATE_TASK = '.hermes/STEER_stop_now_20261030.md'


def read(p, default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default


def classify(args, cwd_matches=True):
    if not cwd_matches or not args:
        return None
    if '-m' in args:
        module = args[args.index('-m') + 1]
        if module.startswith('scripts.combat_mix'):
            return None
        if module.startswith('scripts.best_publish_') or module == 'scripts.best_model_cleanup_20261030':
            return 'publication_controller_or_evaluator'
        if module.startswith('scripts.bomb_threat_'):
            if module.endswith('_train') or module.endswith('_prewarm') or module == 'scripts.bomb_threat_reboot_worker':
                return 'training_worker'
            return 'project_controller_or_evaluator'
        if module.startswith('scripts.iter_pbt_'):
            return 'training_worker' if module == 'scripts.iter_pbt_train' else 'project_controller_or_evaluator'
        if module in ('scripts.checkpoint_retention', 'scripts.build_model_metrics_table', 'scripts.authorized_campaign_watchdog'):
            return 'maintenance'
        if module == 'http.server' and any(port in args for port in ('8123', '8124', '8109')):
            return 'verified_local_replay_server'
        return None
    if any(a == 'runs/authorized_campaign_20261026/scale_env_batch_pilot_v1/scrub_runtime_ids.py' for a in args):
        return 'maintenance'
    if Path(args[0]).name == 'node' and (any(a.startswith('scripts/bomb_threat_') and a.endswith('.js') for a in args[1:]) or
                                                    any(a.startswith('runs/bomb_threat_v1/') for a in args[1:])):
        return 'JS_evaluator'
    if any(Path(a).name in ('native_drain_gpu2.py', 'native_drain_gpu2_lane.py') for a in args):
        return 'native_drain'
    return None


def processes():
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            state = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if state[0] == 'Z' or p.stat().st_uid != os.getuid():
                continue
            args = [s.decode(errors='replace') for s in (p / 'cmdline').read_bytes().split(b'\0') if s]
            role = classify(args, (p / 'cwd').resolve() == ROOT)
            if role:
                found.append(dict(pid=int(p.name), starttime=state[19], args=args, role=role))
        except (OSError, IndexError):
            pass
    return found


def identity(p):
    args = p['args']
    if '-m' in args:
        module = args[args.index('-m') + 1]
        relative = [a for a in args[args.index('-m') + 2:] if a.startswith('runs/')]
        return dict(module=module, arguments=relative, role=p['role'])
    return dict(module='node' if p['role'] == 'JS_evaluator' else 'native_drain',
                arguments=[a for a in args[1:] if a.startswith(('scripts/', 'runs/'))], role=p['role'])


def terminate_identified(p):
    # Revalidate exact start time, cwd and argv immediately before the signal.
    fresh = next((q for q in processes() if q['pid'] == p['pid']), None)
    if fresh and (fresh['starttime'], fresh['args'], fresh['role']) == (p['starttime'], p['args'], p['role']):
        assert p['role'] != 'training_worker', 'GPU workers stop through their atomic pause path only'
        os.kill(p['pid'], signal.SIGTERM)
        return identity(p)
    return None


def latest_slots():
    slots = {}
    for slot, value in read(B / 'SLOTS.json').items():
        run = B / 'formal' / value['run']
        cp = sorted(run.glob('u*/COMMITTED.json'))[-1].parent
        cm = read(cp / 'COMMITTED.json')
        assert cm['passed'] and (cp / 'state.pkl').exists()
        assert sha(cp / 'state.pkl') == cm['state_sha256']
        assert sha(cp / 'policy.pt') == cm['policy_sha256']
        token = run / f'NEXT_STEP_u{cm["update"]:06d}.json'
        # Legacy long_train's pause save commits a complete state but does not
        # write its periodic next-step token. Preserve that state without
        # inventing a probe result; fresh restore validation is due on resume.
        token_present = token.exists()
        if not token_present:
            assert cm.get('metrics', {}).get('user_paused'), 'Unexpected missing next-step token'
        proto = read(B / 'protocols' / f'{value["run"]}.json')
        spec = proto['arms'][value['run']]
        slots[slot] = dict(run=str(run.relative_to(ROOT)), checkpoint=str(cp.relative_to(ROOT)),
                          last_atomic_update=cm['update'], state_sha256=cm['state_sha256'], policy_sha256=cm['policy_sha256'],
                          COMMITTED_sha256=sha(cp / 'COMMITTED.json'), NEXT_STEP_sha256=sha(token) if token_present else None,
                          next_step_probe_available=token_present,
                          resume_validation='fresh restore and deterministic next-step check required before training',
                          protocol=str((B / 'protocols' / f'{value["run"]}.json').relative_to(ROOT)),
                          protocol_sha256=sha(B / 'protocols' / f'{value["run"]}.json'), recipe=spec,
                          protocol_snapshot=proto, arm=value.get('r1_arm', spec.get('iter2_arm', value.get('iter2_arm'))),
                          r1_attack=proto.get('r1_attack'), r1_reward=proto.get('r1_reward'),
                          rule_version=proto.get('rule_version'), opponent_weights=proto.get('A1_weights'),
                          seed=proto.get('r1_seed', int(next(iter(proto['sources'])))),
                          next_chunk=value['chunk'] + 1, resume_current_chunk_before_next=True,
                          chunk_end_update=proto['chunk_start_update'] + proto['chunk_updates'])
    return slots


def pending_evaluations():
    pending = []
    for base in (B, ROOT / 'runs/iter_pbt_v1', ROOT / 'runs/authorized_campaign_20261026'):
        for request in sorted((base / 'eval_requests').glob('*.json')):
            out = base / 'eval' / request.stem
            native = 'native_hunter256_v1/NATIVE_HUNTER_COMPLETE.json' if (base / 'native_fullgame_v2/PROTOCOL.json').exists() else 'native/NATIVE_COMPLETE.json'
            missing = [n for n in ('COMPLETE.json', native) if not (out / n).exists()]
            if missing:
                pending.append(dict(request=str(request.relative_to(ROOT)), request_sha256=sha(request), missing=missing,
                                    completed_outputs=[str(p.relative_to(ROOT)) for p in sorted(out.rglob('*.outcome.json'))],
                                    partial_outputs=[str(p.relative_to(ROOT)) for p in sorted(out.rglob('*.partial.json'))]))
    return pending


def stop(immediate=False):
    publication = read(O / 'DONE.json', {})
    if not immediate:
        assert publication.get('status') in ('COMPLETED', 'COMPLETED_RETAINED_BASELINE'), 'Publication/replay acceptance must precede shutdown'
        assert publication.get('online_verified') and publication.get('replay_verified')
    task = IMMEDIATE_TASK if immediate else TASK
    if immediate:
        assert (ROOT / task).exists()
        publication = dict(status='DEFERRED_UNTIL_AFTER_REBOOT',
                           previous_live_default='bun392-a3sr-u1280',
                           preliminary_selection=read(O / 'SELECTION.json', {}),
                           eligibility_progress=read(O / 'ELIGIBILITY_STATUS.json', {}),
                           pending_order=['finish slot03 u2400 matched comparison using saved outcomes',
                                          'final model selection', 'publication and new replay verification'])
        publication['completed_outcome_files'] = [dict(path=str(p.relative_to(ROOT)), sha256=sha(p))
                for p in sorted((O / 'additional_eligible').rglob('*.outcome.json'))]
        publication['partial_files'] = [dict(path=str(p.relative_to(ROOT)), sha256=sha(p))
                for p in sorted((O / 'additional_eligible').rglob('*.partial.json'))]
        publication['no_new_commit_tag_release_or_main_update'] = not any((O / n).exists()
                for n in ('LOCAL_COMMIT.json', 'RELEASE_RECEIPT.json', 'MAIN_PUSH.json'))
    authorization = dict(status='STOP_FOR_REBOOT', instruction=task, instruction_sha256=sha(ROOT / task),
                         publication_receipt_sha256=sha(O / 'DONE.json') if (O / 'DONE.json').exists() else None, requires_explicit_resume=True,
                         action='finish in-flight update, atomically save full state/policy/parity token, exit; no next run', unix=time.time())
    # Both files are consumed by the already-running owners/workers/watchdogs.
    for base in (B, ROOT / 'runs/iter_pbt_v1'):
        for name in ('STOP_FOR_REBOOT.json', 'USER_STOP.json', 'USER_PAUSE.json'):
            path = base / name
            if path.exists():
                old = read(path)
                authorization_with_history = dict(authorization, previous_receipt=old)
            else:
                authorization_with_history = authorization
            atomic_json(path, authorization_with_history)
    for name in ('USER_STOP_EVAL.json', 'USER_STOP_NATIVE.json'):
        atomic_json(B / name, authorization)
    atomic_json(B / 'REBOOT_STOP_PROGRESS_20261030.json', dict(status='GATES_INSTALLED_WORKERS_ATOMIC_DRAINING',
                actual_consumers=['USER_STOP.json', 'USER_PAUSE.json', 'USER_STOP_EVAL.json', 'USER_STOP_NATIVE.json'],
                training_worker_signals=0, unix=time.time()))
    notified = subprocess.run(['jdme-notify', 'send', ('发布已按最新指令推迟到重启后。' if immediate else '网页模型与录像核验已完成。') + 'STOP_FOR_REBOOT持久闸门已生效，正在等待训练最近完整原子保存并停止评估/守护；尚未通知可以重启。'], capture_output=True)
    # Freeze schedulers first; workers have the already-installed file gate.
    stopped = []
    for p in processes():
        if p['role'] != 'training_worker':
            done = terminate_identified(p)
            if done:
                stopped.append(done)
    deadline = time.monotonic() + 600
    while any(p['role'] == 'training_worker' for p in processes()):
        if time.monotonic() > deadline:
            raise RuntimeError('Atomic pause did not finish within600seconds; workers not signaled; inspect live run')
        time.sleep(3)
    # No GPU training process remains before validating every state/hash.
    slots = latest_slots()
    payload_acceptance = read(B / 'REBOOT_CHECKPOINT_PAYLOAD_ACCEPTANCE_20261030.json', {})
    assert payload_acceptance.get('status') == 'PASS', 'Full checkpoint payload verification required'
    for slot, value in slots.items():
        assert payload_acceptance['slots'][slot]['update'] == value['last_atomic_update']
    for p in processes():
        done = terminate_identified(p)
        if done:
            stopped.append(done)
    time.sleep(3)
    remaining = [identity(p) for p in processes()]
    if remaining:
        raise RuntimeError('Known project processes remain after graceful SIGTERM; no SIGKILL used')
    pending = pending_evaluations()
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    gpu = [dict(index=int(a), memory_MiB=int(b), utilization=int(c)) for a, b, c in (line.split(',') for line in raw.strip().splitlines())]
    # Do not store machine process identifiers or terminate another project.
    applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    free = not applications and all(x['memory_MiB'] == 0 for x in gpu)
    manifest = dict(status='READY_FOR_REBOOT' if free else 'PROJECT_STOPPED_EXTERNAL_GPU_APPS_REMAIN',
                    authorization=authorization, slots=slots, pending_evaluations=pending,
                    stopped_processes=stopped, remaining_project_processes=remaining,
                    GPU=gpu, compute_app_count=len(applications), training_worker_signals=0,
                    unknown_process_signals=0, combat_mix_eval_owner_untouched=True,
                    cleanup_receipt='runs/bomb_threat_v1/publish_best_20261030/CLEANUP_RECEIPT_20261030.json',
                    cleanup_progress=dict(status='BATCH_COMPLETED_NO_FURTHER_DELETION',
                        deleted_by_type=read(O / 'CLEANUP_RECEIPT_20261030.json')['deleted_by_type']),
                    checkpoint_payload_acceptance='runs/bomb_threat_v1/REBOOT_CHECKPOINT_PAYLOAD_ACCEPTANCE_20261030.json',
                    earlier_stop_receipts=['runs/bomb_threat_v1/REBOOT_INITIAL_STOPPED_20261030.json',
                                           'runs/bomb_threat_v1/REBOOT_ADDITIONAL_STOPPED_20261030.json'],
                    publication=publication,
                    resume_order=['new explicit authorization', 'archive and revoke STOP_FOR_REBOOT + actual USER_STOP/USER_PAUSE gates',
                                  'unique bomb_threat_owner adopts slots', 'fresh restore/smoke of exact latest state, then first formal nonzero update',
                                  'eval owners', 'watchdog', 'retention after fresh restore'],
                    resume_commands={'owner': '<PYTHON> -m scripts.bomb_threat_owner',
                                     'eval': '<PYTHON> -m scripts.bomb_threat_eval_owner',
                                     'native_eval': '<PYTHON> -m scripts.bomb_threat_native_owner',
                                     'watchdog': '<PYTHON> -m scripts.bomb_threat_watchdog',
                                     'retention': '<PYTHON> -m scripts.checkpoint_retention --apply --watch --interval 30'},
                    old_R2_O5_reserve_queues_remain_paused=True, unix=time.time())
    path = B / 'REBOOT_RESUME_MANIFEST_20261030.json'
    atomic_json(path, manifest)
    if not free:
        raise RuntimeError('Project stopped but external GPU usage remains; do not kill unknown processes or claim machine ready')
    message = '可以重启：' + ('发布/新录像按最新指令推迟，已完成复核断点保留；' if immediate else '网页模型/新录像已核验；') + '七路完整原子保存后停止，训练/评估/守护均退出，8卡显存0/compute app0。最后update ' + ', '.join(f's{s}:u{x["last_atomic_update"]}' for s, x in slots.items()) + '；恢复清单 runs/bomb_threat_v1/REBOOT_RESUME_MANIFEST_20261030.json；' + publication.get('release_url', '') + ' ' + publication.get('replay_url', '')
    subprocess.run(['jdme-notify', 'send', message], capture_output=True, check=True)
    atomic_json(B / 'REBOOT_READY_20261030.json', dict(status='READY_FOR_REBOOT', manifest_sha256=sha(path), notification='可以重启', unix=time.time()))
    print(json.dumps(dict(status='READY_FOR_REBOOT', slots={s: x['last_atomic_update'] for s, x in slots.items()}, compute_apps=0)))


if __name__ == '__main__':
    import sys
    stop(immediate='--immediate' in sys.argv)
