from scripts.bomb_threat_monitor0200 import regression


def test_regression_requires_four_observed_points_and_strict_thresholds():
    points = {6960: dict(games=64, causal_kills=6, zero_bomb_games=13),
              7000: dict(games=64, causal_kills=5, zero_bomb_games=6),
              7040: dict(games=64, causal_kills=10, zero_bomb_games=7)}
    assert regression(points)['status'] == 'INCOMPLETE'
    points[7080] = dict(games=64, causal_kills=10, zero_bomb_games=7)
    assert regression(points)['status'] == 'THRESHOLD_NOT_MET'
    assert regression(points)['flagged_updates'] == [6960, 7000]
    points[7080]['zero_bomb_games'] = 8
    assert regression(points)['status'] == 'REGRESSION'
    points[7080] = dict(games=63, causal_kills=5, zero_bomb_games=0)
    assert regression(points)['status'] == 'INCOMPLETE'
