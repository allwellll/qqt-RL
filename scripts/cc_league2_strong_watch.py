"""Run the strong-opponent eval (cc_strong_eval_v1, seats 0,1 vs G_frozen) at every 80-update COMMITTED point of league v2 arms.
usage: model_diagnostics.py <config> args=[ARM ...]
Separate process: cc_league2_supervise is sha-pinned by running protocols. One eval at a time (the eval's own
cpu_slots flock caps total concurrency); newest point first so selection evidence stays current. CPU only.
Exits when every listed arm's supervisor is STOPPED/EXITED and nothing is pending.
"""
import json, subprocess, sys, time
from pathlib import Path
from scripts.model_launch import manifest, command
from scripts.cc_fixed_eval_v1 import CPU_ENV

ROOT = Path('runs/cc_train_20261105'); FORMAL = ROOT / 'training/formal'; POINTS = ROOT / 'eval_strong_v1/points'
CAMPAIGN = 18262
BAR = (.550, .436)  # G_frozen two-seat score + 1 SE (greedy .500, categorical .386)


def pending(arms):
    todo = []
    for arm in arms:
        for f in (FORMAL / arm).glob('u*/COMMITTED.json'):
            u = int(f.parent.name[1:])
            if u > CAMPAIGN and (u - CAMPAIGN) % 80 == 0 and not (POINTS / f'{arm}_u{u:06d}__seats01.json').exists():
                todo.append((u, arm, f.parent))
    # Replication first: the next pending point after an evaluated candidate (both modes above G_frozen + 1 SE)
    # is evaluated before newer points; otherwise newest first. (08:40 fix: L2_S05b u019862 starved.)
    def candidate(arm, u):
        f = POINTS / f'{arm}_u{u:06d}__seats01.json'
        if not f.exists():
            return False
        g = json.loads(f.read_text())['G_frozen']
        return g['greedy']['score'] > BAR[0] and g['categorical']['score'] > BAR[1]
    def priority(t):
        u, arm, _ = t
        return (candidate(arm, u - 80), u)
    return sorted(todo, key=priority, reverse=True)


def main():
    arms = sys.argv[1:]; failed = set()
    log = ROOT / 'training/logs/strong_watch_league2.jsonl'
    note = lambda **kw: log.open('a').write(json.dumps(dict(unix=time.time(), **kw)) + '\n')
    while True:
        todo = [t for t in pending(arms) if t[2] not in failed]
        if todo:
            u, arm, cp = todo[0]
            cfg = manifest(dict(module='scripts.cc_strong_eval_v1', args=['point', str(cp), '--seats', '0,1', '--opponents', 'G_frozen'], env=CPU_ENV))
            note(event='start', arm=arm, update=u, config=cfg)
            with (FORMAL / arm / 'strong_eval.log').open('a') as f:
                code = subprocess.run(command('evaluation', cfg), stdout=f, stderr=subprocess.STDOUT).returncode
            note(event='exit', arm=arm, update=u, code=code)
            if code:
                failed.add(cp)
            continue
        done = all(json.loads((FORMAL / a / 'SUPERVISOR.json').read_text()).get('status') in ('STOPPED', 'EXITED') for a in arms)
        if done:
            break
        time.sleep(120)


if __name__ == '__main__':
    main()
