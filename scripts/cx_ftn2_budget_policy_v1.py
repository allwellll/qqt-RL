"""Pure fixed-budget stopping rule; evaluated points must be complete and ordered."""


def trend_stop(points):
    """Two successive strict declines in BOTH pooled net wins and deliveries."""
    if len(points) < 3:
        return False
    last = points[-3:]
    assert all(a['update'] < b['update'] for a, b in zip(last, last[1:]))
    return all(b['net_wins'] < a['net_wins'] and b['deliveries'] < a['deliveries']
               for a, b in zip(last, last[1:]))


def point_totals(point):
    assert point['status'] == 'STRONG_EVAL_V1_POINT' and point['seats'] == [0, 1]
    modes = point['G_frozen']
    assert set(modes) == {'greedy', 'categorical'}
    assert all(sum(modes[m][k] for k in ('wins', 'losses', 'draws')) == 88
               for m in modes), 'partial point cannot stop training'
    return dict(update=point['learner']['update'],
                net_wins=sum(m['wins'] - m['losses'] for m in modes.values()),
                deliveries=sum(m['deliveries'] for m in modes.values()))
