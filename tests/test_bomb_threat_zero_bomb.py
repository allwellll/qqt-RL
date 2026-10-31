from scripts.bomb_threat_eval import summarize
from scripts.bomb_threat_gates_v2 import evaluate


def episode(seat, own, other):
    placed = [other, other]
    placed[seat] = own
    return dict(seat=seat, winner=None, causal_kills=[0, 0], cage_kills=[0, 0],
                placed=placed, requests=placed, deaths=[0, 0], self_deaths=[0, 0],
                stored=[[1, 0], [0, 1]])


def test_zero_bombs_excludes_bilateral_zero_and_respects_seat():
    r = summarize([episode(0, 0, 0), episode(1, 0, 2), episode(0, 3, 0)])
    assert r['zero_bomb_games'] == 1
    assert r['structural_zero_bomb_games'] == 1
    assert r['raw_zero_bomb_games'] == 2
    assert r['opponent_bomb_opportunity_games'] == 1
    assert r['games'] == 3  # Do not remove games from any ability denominator.


def test_absolute_zero_bomb_limit_is_seven_not_half_redefined_parent():
    p = dict(games=64, self_deaths=0, cage_kill_fraction=0, zero_bomb_games=0,
             bombs_per_game=25, draw_fraction=.8, win_minus_loss=0, causal_kills=30, deaths=15)
    g = dict(p, zero_bomb_games=7, win_minus_loss=8)
    n = dict(greedy=g, categorical=dict(games=64))
    refs = {k: dict(greedy=p) for k in ('reference_transport1536', 'reference_selfplay3072')}
    assert evaluate(n, dict(greedy=p), dict(threat_reward_fraction_total=.1), 'holdout', refs)['absolute']['status'] == 'PASS'
    g['zero_bomb_games'] = 8
    assert 'ZERO_BOMBS_GT_7' in evaluate(n, dict(greedy=p), dict(threat_reward_fraction_total=.1), 'holdout', refs)['absolute']['failures']


def test_table_handles_json_reference_without_checkpoint(tmp_path, monkeypatch):
    import json
    from scripts import build_model_metrics_table as T
    monkeypatch.setattr(T, 'NEW', tmp_path)
    monkeypatch.setattr(T, 'legacy_rows', lambda: [])
    out = tmp_path/'eval/reference'
    out.mkdir(parents=True)
    (out/'NATURAL_COMPLETE.json').write_text(json.dumps(dict(measurement_version=2,
        checkpoint=None, split='dev', seed=100, natural_js_full2400=dict(greedy=dict(
            games=64, zero_bomb_games=1, structural_zero_bomb_games=4, raw_zero_bomb_games=5)))))
    rows = T.rows()
    assert len(rows) == 1 and rows[0]['structural_zero_bomb_games'] == 4
