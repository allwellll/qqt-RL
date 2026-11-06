"""Write .hermes/CC_TRAIN_PROGRESS.json every 5 minutes from run artifacts (no process introspection of training)."""
import json, subprocess, time
from pathlib import Path
from scripts.population_run_state import atomic_json

ROOT = Path('runs/cc_train_20261105'); FORMAL = ROOT / 'training/formal'
ARMS = None  # discovered from formal/*/SUPERVISOR.json (gpu field); stopped arms keep their last record
KEYS = ('wins', 'losses', 'draws', 'deaths', 'self_deaths', 'causal_kills', 'placed', 'zero_placement_games',
        'opposite_rate', 'acquisitions', 'deliveries', 'carry_failures', 'item_pickups')


def brief(path):
    v = json.loads(path.read_text())
    return {m: {k: (round(t[k], 3) if isinstance(t[k], float) else t[k]) for k in KEYS} for m, t in v['by_mode'].items()}


def snapshot():
    arms = {}
    for out in sorted(p.parent for p in FORMAL.glob('*/SUPERVISOR.json')):
        arm = out.name; sup = json.loads((out / 'SUPERVISOR.json').read_text())
        gpu = sup.get('gpu') or next((json.loads(l).get('gpu') for l in []), None)
        cps = sorted(out.glob('u*/COMMITTED.json'))
        rows = (out / 'updates.jsonl').read_text().splitlines() if (out / 'updates.jsonl').exists() else []
        last = json.loads(rows[-1]) if rows else {}
        evals = {p.parent.name: brief(p) for p in sorted((ROOT / 'eval_v1').glob(f'{arm}_u*/SUMMARY.json'))}
        arms[arm] = dict(gpu=gpu, latest_update=last.get('new_update'), last_saved=cps[-1].parent.name if cps else None,
                         stage=json.loads((out / 'CYCLE_STAGE.json').read_text()) if (out / 'CYCLE_STAGE.json').exists() else None,
                         cycle_seconds=last.get('cycle_seconds'), failed=(out / 'FAILED.json').exists(),
                         supervisor=json.loads((out / 'SUPERVISOR.json').read_text()).get('status') if (out / 'SUPERVISOR.json').exists() else None,
                         last_exposure={k: last.get('exposure', {}).get('bun', {}).get(k) for k in ('learner_bombs_placed', 'r1_kills', 'r1_selfharm')},
                         evaluations=evals)
    gpus = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip().splitlines()
    return dict(owner='Claude Code (cc-train-gpu4567)', phase='FORMAL_4ARM (see arms[*].supervisor; RUNNING = on GPU)',
                protocol='runs/cc_train_20261105/training/protocols/CC_REGIONS_V1.json',
                baseline_eval={n: brief(ROOT / 'eval_v1' / n / 'SUMMARY.json') for n in ('G_frozen', 'health18270', 'attack18278', 'control18278')
                               if (ROOT / 'eval_v1' / n / 'SUMMARY.json').exists()},
                arms=arms, gpus=gpus, blockers=[], ownership={'GPU4': 'cc-league-v2 (A05_kl)', 'GPU5': 'cc-league-v2 (S05_sil)', 'GPU6': 'IDLE - returned to orchestration (S10_sil stopped u019637)', 'GPU7': 'cc-league-v2 (S10_trap stopped u018724)'}, decisions=json.loads((ROOT / 'DECISIONS.json').read_text()) if (ROOT / 'DECISIONS.json').exists() else [],
                stop_method='touch runs/cc_train_20261105/training/USER_STOP.json (trainer saves at boundary, supervisors exit)',
                ledger='docs/training_experiment_ledger.md#cc-gpu4-7', handoff='.hermes/CC_TRAIN_HANDOFF.md',
                updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'), unix=time.time())


def main():
    while True:
        atomic_json(Path('.hermes/CC_TRAIN_PROGRESS.json'), snapshot())
        if all(json.loads(p.read_text()).get('status') in ('STOPPED', 'EXITED') for p in FORMAL.glob('*/SUPERVISOR.json')):
            break
        time.sleep(300)


if __name__ == '__main__':
    main()
