"""Write .hermes/CC_LEAGUE_V2_PROGRESS.json every 10 minutes from CC_LEAGUE_V2 artifacts (read only).
usage: model_diagnostics.py <config> args=[ARM ...]   exits when every listed arm's supervisor is STOPPED/EXITED."""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json
from scripts.cc_status_v1 import brief

ROOT = Path('runs/cc_train_20261105'); BASE = ROOT / 'training'; FORMAL = BASE / 'formal'
POOLS = ['rule', 'hunter', 'hard', 'Gcurrent', 'Gring', 'Gfixed', 'peer_latest', 'peer_ring', 'G_frozen']


def arm_status(arm):
    out = FORMAL / arm; sup = json.loads((out / 'SUPERVISOR.json').read_text())
    protocols = sorted((BASE / 'protocols').glob(f'CC_LEAGUE2_{arm}_V*.json'), key=lambda p: int(p.stem.rsplit('_V', 1)[1]))
    spec = json.loads(protocols[-1].read_text())['arms'][arm]
    rows = [json.loads(l) for l in (out / 'updates.jsonl').read_text().splitlines()][-80:]
    W = np.zeros(9); G = np.zeros(9); T = np.zeros(9)
    for r in rows:
        e = r['exposure']['bun']; W += e['opponent_win']; T += e['pool_ticks']
        G += np.asarray(e['opponent_win']) + e['opponent_loss'] + e['opponent_draw']
    prog = [json.loads(l) for l in (out / 'PROGRESSION.jsonl').read_text().splitlines()] if (out / 'PROGRESSION.jsonl').exists() else []
    hist = BASE / 'league' / arm / 'league_snapshots/C_history.jsonl'
    peers = [json.loads(l) for l in hist.read_text().splitlines()][-3:] if hist.exists() else []
    return dict(supervisor=sup.get('status'), gpu=sup.get('gpu'), protocol=protocols[-1].name,
                latest_update=rows[-1]['new_update'] if rows else None,
                stage=json.loads((out / 'CYCLE_STAGE.json').read_text()),
                weights=dict(zip(POOLS, spec['opponent_weights'])), weight_history=spec['league2']['weight_history'],
                last80=dict(rows=len(rows), tick_share=dict(zip(POOLS, (T / max(T.sum(), 1)).round(3).tolist())),
                            win_rate=dict(zip(POOLS, (W / np.maximum(G, 1)).round(3).tolist())), games=dict(zip(POOLS, G.astype(int).tolist()))),
                progression_checks=[{k: p[k] for k in ('update', 'ready', 'fired', 'strong_win_rate', 'strong_games')} for p in prog],
                recent_peers=[dict(index=p['index'], peer=p['peer_arm'], current=p['current'], fallback=p['fallback']) for p in peers],
                fixed_eval={p.parent.name: brief(p) for p in sorted((ROOT / 'eval_v1').glob(f'{arm}_u*/SUMMARY.json'))},
                failed=(out / 'FAILED.json').exists())


def main():
    arms = sys.argv[1:]
    while True:
        value = dict(owner='cc-league-v2', protocol_family='CC_LEAGUE2_<ARM>_V<n>', usage='.hermes/CC_LEAGUE_V2_USAGE.md',
                     arms={a: arm_status(a) for a in arms},
                     strong_eval={f.name[:-len('__seats01.json')]: dict(key_vs_G_frozen=json.loads(f.read_text())['selection_key_vs_G_frozen'],
                         greedy_score=round(json.loads(f.read_text())['G_frozen']['greedy']['score'], 3),
                         categorical_score=round(json.loads(f.read_text())['G_frozen']['categorical']['score'], 3))
                         for f in sorted((ROOT / 'eval_strong_v1/points').glob('*__seats01.json'))
                         if f.name.startswith(('L2_', 'G_frozen', 'S05_sil_u018862'))},
                     strong_eval_note='seats 0,1 vs G_frozen; key=[net W-L, deliveries, causal kills, -self deaths]; primary for selection',
                     selection=[d for d in json.loads((ROOT / 'DECISIONS.json').read_text()) if d.get('session') == 'cc-league-v2'][-3:],
                     gpus=subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader', '-i', '4,5'],
                                         capture_output=True, text=True).stdout.strip().splitlines(),
                     updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'), unix=time.time())
        atomic_json(Path('.hermes/CC_LEAGUE_V2_PROGRESS.json'), value)
        if all(value['arms'][a]['supervisor'] in ('STOPPED', 'EXITED') for a in arms):
            break
        time.sleep(600)


if __name__ == '__main__':
    main()
