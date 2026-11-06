"""Boundary tests for the authorized two-point regression rule."""
from scripts.cx_ftn2_budget_policy_v1 import trend_stop, point_totals


def main():
    def rows(values):
        return [dict(update=i, net_wins=w, deliveries=d) for i, (w, d) in enumerate(values)]
    cases = [([], False), ([(5, 8), (4, 7)], False),
             ([(5, 8), (4, 7), (3, 6)], True),
             ([(5, 8), (4, 7), (4, 6)], False),
             ([(5, 8), (4, 7), (3, 7)], False),
             ([(5, 8), (4, 9), (3, 8)], False),
             ([(5, 8), (4, 7), (6, 9)], False),
             ([(5, 8), (6, 9), (4, 7), (3, 6)], True)]
    for values, expected in cases:
        assert trend_stop(rows(values)) == expected, values
    point = dict(status='STRONG_EVAL_V1_POINT', seats=[0, 1], learner=dict(update=19462),
                 G_frozen={m: dict(wins=30, losses=20, draws=38, deliveries=11)
                           for m in ('greedy', 'categorical')})
    assert point_totals(point) == dict(update=19462, net_wins=20, deliveries=22)
    point['G_frozen']['categorical']['draws'] = 0
    try:
        point_totals(point)
    except AssertionError:
        pass
    else:
        raise AssertionError('partial point accepted')
    print('PASS 10 fixed-budget stop/partial-point cases', flush=True)


if __name__ == '__main__':
    main()
