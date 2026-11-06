"""CC_LEAGUE_V2 per-GPU supervisor (cc_supervise_v3 + per-arm stop + difficulty progression).

usage: model_diagnostics.py <config> args=[arm, gpu, cpus, family]   (family = protocol file stem without _V<n>)
  * each protocol: smoke2 -> fresh-process restore -> formal; receipts of a superseded protocol are archived;
  * evaluation: every COMMITTED u with (u - campaign_start) % 80 == 0 -> fixed 88-game eval (CPU, async);
  * progression: worker exits 0 with PROGRESSION_REQUEST.json -> next protocol = same arm with the new
    weights, chunk resumes from the saved update to the same chunk end, previous protocol sha accepted;
  * completion -> successor chunk; failure -> one resume retry, then FAILED.json;
  * stop: global USER_STOP/USER_PAUSE/STOP_FOR_REBOOT in runs/cc_train_20261105/training (all arms) or
    <arm dir>/USER_STOP.json (this arm only); the trainer saves at the boundary before exiting.
"""
import json, os, subprocess, sys, threading, time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha
from scripts.model_launch import manifest, command

ROOT = Path('runs/cc_train_20261105'); BASE = ROOT / 'training'
STOPS = ('USER_STOP', 'USER_STOP.json', 'USER_PAUSE.json', 'STOP_FOR_REBOOT.json')
TRAINER = 'scripts.cc_league2_train'


def version(p):
    return int(p.stem.rsplit('_V', 1)[1])


def latest(out):
    cps = sorted(out.glob('u*/COMMITTED.json'))
    return int(json.loads(cps[-1].read_text())['update']) if cps else None


def main():
    arm, gpu, cpus, family = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    chain = sorted((BASE / 'protocols').glob(f'{family}_V*.json'), key=version)
    protocol_path = chain[-1]; protocol = json.loads(protocol_path.read_text())
    out = Path(protocol['namespace']) / arm; out.mkdir(parents=True, exist_ok=True)
    log = out / 'supervisor.jsonl'
    stopped = lambda: any((BASE / n).exists() for n in STOPS) or (out / 'USER_STOP.json').exists()
    def note(**kw):
        with log.open('a') as f:
            f.write(json.dumps(dict(unix=time.time(), **kw)) + '\n')
    def run(name, *extra):
        cfg = manifest(dict(module=TRAINER, gpu=gpu, args=[str(protocol_path), arm, str(protocol['seed'])] + list(extra)))
        note(event='start', phase=name, config=cfg, protocol=str(protocol_path))
        with (out / f'{name}.log').open('a') as f:
            r = subprocess.run(['taskset', '-c', cpus] + command('training', cfg), stdout=f, stderr=subprocess.STDOUT)
        note(event='exit', phase=name, code=r.returncode, latest=latest(out))
        return r.returncode
    def verify():
        # Receipts belong to one protocol; a new protocol (fork or reweight) is smoked and fresh-restored again.
        for name in ('SMOKE', 'RESTORE'):
            f = out / f'{name}.json'
            if f.exists() and json.loads(f.read_text()).get('protocol') != str(protocol_path):
                f.rename(out / f'{name}_{f.stat().st_mtime_ns}.json')
        for name, flag in (('SMOKE', '--smoke'), ('RESTORE', '--verify-restore')):
            if not (out / f'{name}.json').exists():
                assert run(f'{name.lower()}_{protocol_path.stem}', flag) == 0 and (out / f'{name}.json').exists(), f'{name} failed'
                f = out / f'{name}.json'; atomic_json(f, dict(json.loads(f.read_text()), protocol=str(protocol_path)))
    start = int(protocol['campaign_start_update'])
    done_evals = set(); running = {}
    def evaluate_loop():
        while True:
            for f in sorted(out.glob('u*/COMMITTED.json')):
                u = int(f.parent.name[1:])
                if (u - start) % 80 or u == start or u in done_evals or u in running:
                    continue
                label = f'{arm}_u{u:06d}'
                if (ROOT / 'eval_v1' / label / 'SUMMARY.json').exists():
                    done_evals.add(u); continue
                cfg = manifest(dict(module='scripts.cc_fixed_eval_v1', args=[label, str(f.parent), str(ROOT / 'eval_v1')],
                                    env=dict(CC_EVAL_WORKERS='22')))
                with (out / 'evaluation.log').open('a') as lf:
                    running[u] = subprocess.Popen(command('evaluation', cfg), stdout=lf, stderr=subprocess.STDOUT)
                note(event='eval_start', update=u, label=label)
            for u, p in list(running.items()):
                if p.poll() is not None:
                    note(event='eval_exit', update=u, code=p.returncode); done_evals.add(u); running.pop(u)
            time.sleep(30)
    threading.Thread(target=evaluate_loop, daemon=True).start()
    atomic_json(out / 'SUPERVISOR.json', dict(status='RUNNING', pid=os.getpid(), gpu=gpu, cpus=cpus, protocol=str(protocol_path),
                                              owner='cc-league-v2', unix=time.time()))
    retries = 0
    while not stopped():
        verify()
        target = int(protocol['chunk_start_update']) + int(protocol['chunk_updates'])
        code = run(f'formal_{protocol_path.stem}')
        if stopped():
            break
        request = out / 'PROGRESSION_REQUEST.json'
        nxt = BASE / 'protocols' / f'{family}_V{version(protocol_path) + 1}.json'
        if code == 0 and request.exists():
            row = json.loads(request.read_text()); saved = latest(out); assert saved == row['update']
            if not nxt.exists():
                arms = dict(protocol['arms'])
                arms[arm] = dict(arms[arm], opponent_weights=row['new_weights'],
                                 league2=dict(arms[arm]['league2'], weights_since_update=saved,
                                              weight_history=arms[arm]['league2'].get('weight_history', []) + [
                                                  dict(update=saved, weights=row['new_weights'], previous=row['weights'],
                                                       strong_win_rate=row['strong_win_rate'], strong_games=row['strong_games'])]))
                atomic_json(nxt, dict(protocol, arms=arms, chunk_start_update=saved, chunk_updates=target - saved,
                    compatible_checkpoint_protocol_sha256=protocol['compatible_checkpoint_protocol_sha256'] + [sha(protocol_path)]))
            request.rename(out / f'PROGRESSION_CONSUMED_u{saved:06d}.json')
            protocol_path = nxt; protocol = json.loads(nxt.read_text()); retries = 0
            note(event='progression', protocol=str(nxt), update=saved, weights=row['new_weights'], strong_win_rate=row['strong_win_rate'])
            continue
        if code == 0 and latest(out) == target:
            if not nxt.exists():
                atomic_json(nxt, dict(protocol, chunk_start_update=target, chunk_updates=int(protocol['league2_chunk_updates']),
                    compatible_checkpoint_protocol_sha256=protocol['compatible_checkpoint_protocol_sha256'] + [sha(protocol_path)]))
            protocol_path = nxt; protocol = json.loads(nxt.read_text()); retries = 0
            note(event='successor', protocol=str(nxt), start=target); continue
        retries += 1
        if retries > 1:
            atomic_json(out / 'FAILED.json', dict(status='FAILED_AFTER_RETRY', latest=latest(out), unix=time.time())); break
        time.sleep(30)
    deadline = time.time() + 3600
    while running and time.time() < deadline:
        time.sleep(30)
    atomic_json(out / 'SUPERVISOR.json', dict(status='STOPPED' if stopped() else 'EXITED', latest=latest(out), owner='cc-league-v2', unix=time.time()))


if __name__ == '__main__':
    main()
