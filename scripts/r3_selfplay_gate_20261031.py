"""Frozen own-side parent gate; incomplete matrices can never admit training."""
from pathlib import Path
import json
import time
from scripts.population_run_state import atomic_json, sha

BASE = Path('runs/v2_ownside_selection_20261031')
OUT = Path('runs/r3_selfplay_ramp_20261031')


def decide(results, order):
    assert results['status'] == 'COMPLETE' and results['games'] == 6144
    candidates = [k for k in order if results['models'][k]['r3']]
    # Same primary as the previously frozen selection; ties retain frozen order.
    parent = max(candidates, key=lambda k: results['selection']['comparison'][k]['P1_sum'])
    conditions = {}
    for condition in ('hunter_hard', 'coop_hard'):
        seats = results['metrics'][parent][condition]
        p0, p1 = seats['P0'], seats['P1']
        assert p0['games'] == p1['games'] == 128
        n = p1['games']
        checks = dict(P1_win=p1['wins']/n >= .55, P1_loss=p1['losses']/n <= .15,
                      P0_not_below_P1=p0['win_minus_loss'] >= p1['win_minus_loss'])
        conditions[condition] = dict(P0=p0, P1=p1, checks=checks,
                                    win_rate_gap=max(0., .55-p1['wins']/n),
                                    loss_rate_excess=max(0., p1['losses']/n-.15))
    passed = all(all(x['checks'].values()) for x in conditions.values())
    return dict(status='PASS' if passed else 'NOT_ELIGIBLE', parent=parent,
                parent_source=results['models'][parent], conditions=conditions,
                P0_interpretation='win-minus-loss on equal128 games in each hard condition',
                primary='fixed three-tier P1 win-minus-loss; first frozen candidate on ties',
                mode='production JS greedy; teacher off', rule='bun-native-v2',
                spawn_side_version='own-side-native-v1', no_holdout_tuning=True)


def evaluate():
    complete = BASE/'COMPLETE.json'
    if not complete.exists():
        return dict(status='WAIT_OWNSIDE_6144', training_started=False, unix=time.time())
    done = json.loads(complete.read_text())
    assert done['status'] == 'COMPLETE' and done['games'] == 6144
    assert sha(BASE/'RESULTS.json') == done['results_sha256']
    results = json.loads((BASE/'RESULTS.json').read_text())
    p = json.loads((BASE/'PROTOCOL.json').read_text())
    assert p['spawn_side_version'] == 'own-side-native-v1'
    result = decide(results, p['candidate_order'])
    result.update(results_sha256=done['results_sha256'], protocol_sha256=sha(BASE/'PROTOCOL.json'), unix=time.time())
    atomic_json(OUT/'PARENT_GATE.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(evaluate(), ensure_ascii=False))
