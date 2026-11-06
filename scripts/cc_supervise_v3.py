"""Per-GPU supervisor: smoke2 -> fresh-process restore -> formal chunk -> declared successor chunk.

usage: model_diagnostics.py <config> args=[arm, gpu, cpus, family, trainer_module]
  * evaluation: every committed u where (u - campaign_start) % 80 == 0 is evaluated on the fixed
    88-game protocol in a separate CPU process; training never waits for it.
  * failure: one resume retry from the last COMMITTED checkpoint, then FAILED.json and exit.
  * completion: the same arm continues with a successor protocol (+chunk_updates), until STOP.
  * stop: any of USER_STOP/USER_PAUSE/STOP_FOR_REBOOT(.json) in runs/cc_train_20261105/training;
    the trainer itself saves at the boundary before exiting.
"""
import json, os, subprocess, sys, threading, time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha
from scripts.model_launch import manifest, command

ROOT = Path('runs/cc_train_20261105'); BASE = ROOT / 'training'
STOPS = ('USER_STOP', 'USER_STOP.json', 'USER_PAUSE.json', 'STOP_FOR_REBOOT.json')


def stopped():
    return any((BASE / n).exists() for n in STOPS)


def latest(out):
    cps = sorted(out.glob('u*/COMMITTED.json'))
    return int(json.loads(cps[-1].read_text())['update']) if cps else None


def main():
    arm, gpu, cpus, family, trainer = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    chain = sorted((BASE / 'protocols').glob(f'{family}_V*.json'), key=lambda p: int(p.stem.split('_V')[1]))
    protocol_path = chain[-1]; protocol = json.loads(protocol_path.read_text())
    out = Path(protocol['namespace']) / arm; out.mkdir(parents=True, exist_ok=True)
    log = out / 'supervisor.jsonl'
    def note(**kw):
        with log.open('a') as f:
            f.write(json.dumps(dict(unix=time.time(), **kw)) + '\n')
    def run(name, *extra):
        cfg = manifest(dict(module=trainer, gpu=gpu, args=[str(protocol_path), arm, str(protocol['seed'])] + list(extra)))
        note(event='start', phase=name, config=cfg, protocol=str(protocol_path))
        with (out / f'{name}.log').open('a') as f:
            r = subprocess.run(['taskset', '-c', cpus] + command('training', cfg), stdout=f, stderr=subprocess.STDOUT)
        note(event='exit', phase=name, code=r.returncode, latest=latest(out))
        return r.returncode
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
            for u, p in list(running.items()):  # only this thread mutates `running`
                if p.poll() is not None:
                    note(event='eval_exit', update=u, code=p.returncode); done_evals.add(u); running.pop(u)
            time.sleep(30)
    threading.Thread(target=evaluate_loop, daemon=True).start()
    atomic_json(out / 'SUPERVISOR.json', dict(status='RUNNING', pid=os.getpid(), gpu=gpu, cpus=cpus, protocol=str(protocol_path), unix=time.time()))
    if not (out / 'SMOKE.json').exists():
        assert run('smoke2', '--smoke') == 0 and (out / 'SMOKE.json').exists(), 'smoke failed'
    if not (out / 'RESTORE.json').exists():
        assert run('restore', '--verify-restore') == 0 and (out / 'RESTORE.json').exists(), 'fresh restore failed'
    retries = 0
    while not stopped():
        target = int(protocol['chunk_start_update']) + int(protocol['chunk_updates'])
        code = run(f'formal_{protocol_path.stem}')
        if stopped():
            break
        if code == 0 and latest(out) == target:
            # Declared successor: same arm, next chunk, previous protocol accepted for checkpoint compatibility.
            index = int(protocol_path.stem.split('_V')[1]) + 1
            nxt = BASE / 'protocols' / f'{family}_V{index}.json'
            if not nxt.exists():
                atomic_json(nxt, dict(protocol, chunk_start_update=target,
                    compatible_checkpoint_protocol_sha256=protocol['compatible_checkpoint_protocol_sha256'] + [sha(protocol_path)]))
            protocol_path = nxt; protocol = json.loads(nxt.read_text()); retries = 0
            note(event='successor', protocol=str(nxt), start=target); continue
        retries += 1
        if retries > 1:
            atomic_json(out / 'FAILED.json', dict(status='FAILED_AFTER_RETRY', latest=latest(out), unix=time.time())); break
        time.sleep(30)
    # Let in-flight evaluations finish before exiting (the eval thread clears `running`).
    deadline = time.time() + 3600
    while running and time.time() < deadline:
        time.sleep(30)
    atomic_json(out / 'SUPERVISOR.json', dict(status='STOPPED' if stopped() else 'EXITED', latest=latest(out), unix=time.time()))


if __name__ == '__main__':
    main()
