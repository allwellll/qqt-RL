"""Audit settled Hunter safety without rewriting historical attribution counts."""
import json
from pathlib import Path

from scripts.population_run_state import sha


def episode_outcome(row):
    counts = row['counts']
    assert row['residual']['hazards_remaining'] is False
    kill = bool(counts['surviving_causal_kill'] or counts['surviving_physical_kill'])
    death = bool(counts['deaths'])
    alive = bool(row['residual']['alive'][0])
    return dict(safe_win=int(kill and not death and alive),
                death_episode=int(death), final_dead=int(not alive),
                kill_then_death=int(kill and death))


def paired_outcomes(directory):
    directory = Path(directory)
    metrics = {a: {} for a in ('control', 'treatment')}
    delta = {}
    sources = {}
    for difficulty in ('hard', 'normal', 'easy'):
        records = {}
        for arm in metrics:
            path = directory / f'{arm}_{difficulty}.json'
            sources[path.name] = sha(path)
            records[arm] = json.loads(path.read_text())['episodes']
            assert len(records[arm]) == 8
            assert [row['game'] for row in records[arm]] == list(range(8))
            outcomes = [episode_outcome(row) for row in records[arm]]
            metrics[arm][difficulty] = {key: sum(row[key] for row in outcomes)
                                        for key in outcomes[0]}
        for control, treatment in zip(records['control'], records['treatment']):
            assert control['game'] == treatment['game']
            assert control['spawn_cells'] == treatment['spawn_cells']
        delta[difficulty] = {key: metrics['treatment'][difficulty][key] -
                              metrics['control'][difficulty][key]
                              for key in metrics['control'][difficulty]}
    return dict(passed=True, protocol='settled_hunter_actual_outcomes_v1',
                episodes_total=48, metrics=metrics, delta=delta,
                total_delta={key: sum(row[key] for row in delta.values())
                             for key in delta['hard']}, raw_shard_sha256=sources,
                semantics='safe win requires credited kill, zero actor deaths, and settled actor alive')
