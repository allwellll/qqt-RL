"""Read-only training metrics; imports no JAX and loads no model checkpoints."""
import argparse
import json
from pathlib import Path


def summarize(out):
    config = json.loads((out/'PLAN.json').read_text())
    last = None
    updates = out/'updates.jsonl'
    if not updates.is_file():
        return dict(stage=config['stage'], status='no completed update')
    with updates.open() as f:
        for line in f:
            # A writer may currently be appending the last JSONL record.
            if line.endswith('\n') and line.strip():
                last = json.loads(line)
    if last is None:
        return dict(stage=config['stage'], status='no completed update')
    names = last['metric_names']
    groups = []
    for kind, pair in zip(last.get('opponent_ids', ('interceptor', 'SP_web')),
                          last['cumulative_metrics'], strict=True):
        a, b = [dict(zip(names, row)) for row in pair]
        games = a['games']
        if games == 0:
            continue
        ratio = lambda x, y: x/y if y else None
        groups.append(dict(opponent=kind, completed_games=int(games),
            learner_delivery_rate=ratio(a['captures'], games),
            opponent_delivery_rate=ratio(b['captures'], games),
            learner_win_rate=ratio(a['wins'], games),
            draw_rate=ratio(a['draws'], games),
            learner_kills_per_game=ratio(a['kills'], games),
            learner_deaths_per_game=ratio(a['deaths'], games),
            learner_carrier_kills=a['carrier_kills'],
            opponent_carrier_kills=b['carrier_kills'], learner_self_deaths=a['self_deaths'],
            opponent_self_deaths=b['self_deaths'], learner_drops=a['drops'], opponent_drops=b['drops']))
    return dict(stage=config['stage'], update=last['update'],
                environment_steps=last['environment_steps'],
                recent_environment_steps_per_second=last['environment_steps_per_second'],
                groups=groups, frozen_opponents=True,
                warning='Training aggregates include unfinished games and changing learner weights; not held-out evaluation')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('run', type=Path)
    args = p.parse_args()
    print(json.dumps(summarize(args.run), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
