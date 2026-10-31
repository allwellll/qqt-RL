from scripts.bomb_threat_eval import summarize


def row(seat, winner, placed, kills):
    return dict(seat=seat, winner=winner, placed=placed, causal_kills=kills,
                cage_kills=[0, 0], requests=placed, deaths=[0, 0],
                self_deaths=[0, 0], stored=[[1, 0], [0, 1]])


def test_passive_seat_metrics_use_learner_seat_and_preserve_global_gate():
    r = summarize([row(0, 0, [2, 0], [3, 1]), row(1, 0, [4, 0], [2, 0]),
                   row(1, None, [0, 0], [0, 1])])
    assert r['games'] == 3 and r['zero_bomb_games'] == 1
    assert r['by_seat']['P0']['games'] == 1
    assert r['by_seat']['P0']['win'] == 1
    assert r['by_seat']['P0']['bombs_per_game'] == 2
    assert r['by_seat']['P1']['loss'] == 1
    assert r['by_seat']['P1']['draw'] == 1
    assert r['by_seat']['P1']['zero_bomb_games'] == 1
    assert r['by_seat']['P1']['structural_zero_bomb_games'] == 1
    assert r['by_seat']['P1']['causal_kills'] == 1


def test_uncovered_seat_is_unavailable_not_zero_games_of_capability():
    r = summarize([row(0, None, [1, 1], [0, 0])])
    assert r['by_seat']['P1'] is None


def test_actual_exposure_deduplicates_updates_and_never_counts_enemy_as_learner(tmp_path):
    import json
    from scripts.bomb_threat_seat_audit import collect_exposure
    p=tmp_path/'updates.jsonl'
    record=dict(new_update=1,exposure=dict(pool_ticks=[10,20,30,40,50,60],
        pool_live_ticks=[9,19,29,39,49,59],pool_new_episodes=[1]*6))
    p.write_text(json.dumps(record)+'\n'+json.dumps(record)+'\n'+ '{"partial":')
    r=collect_exposure([p])
    assert r['runs'][0]['updates']==1
    assert r['by_opponent']['selfplay_current']['P0_ticks']==40
    assert all(x['P1_ticks']==0 and x['learner_P1_fraction']==0 for x in r['by_opponent'].values())


def test_actual_exposure_reads_a3s_seat_counters_without_counting_enemy(tmp_path):
    import json
    from scripts.bomb_threat_seat_audit import collect_exposure
    p=tmp_path/'updates.jsonl';p.write_text(json.dumps(dict(new_update=1,exposure=dict(
        pool_ticks=[10]*6,pool_live_ticks=[8]*6,pool_new_episodes=[1]*6,
        pool_seat_ticks=[[4,6]]*6,pool_seat_live_ticks=[[3,5]]*6)))+'\n')
    r=collect_exposure([p])['by_opponent']['hunter_tier3a']
    assert r['P0_ticks']==4 and r['P1_ticks']==6 and r['P1_live_ticks']==5 and r['learner_P1_fraction']==.6


def test_table_exposes_seat_metrics_without_replacing_global_metrics():
    from scripts.build_model_metrics_table import seat_columns
    metrics=summarize([row(0,0,[2,0],[3,1]),row(1,0,[4,0],[2,0])])
    columns=seat_columns(metrics)
    assert columns['P0_w_d_l']=='1/0/0'
    assert columns['P1_zero_bomb_games']==1
    assert metrics['games']==2 and metrics['loss']==1
