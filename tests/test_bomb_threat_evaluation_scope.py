import json
from scripts import bomb_threat_eval as E


def test_legacy_parent_missing_exposure_is_unavailable_not_zero_or_crash(tmp_path,monkeypatch):
    monkeypatch.setattr(E,'BASE',tmp_path/'new')
    run=tmp_path/'legacy';run.mkdir()
    (run/'updates.jsonl').write_text(json.dumps({'new_update':1})+'\n')
    assert E.native_reward_exposures(run/'u000001',1)==[]


def test_reward_scope_uses_new_version_and_target_prefix_only(tmp_path,monkeypatch):
    monkeypatch.setattr(E,'BASE',tmp_path)
    run=tmp_path/'formal/run';run.mkdir(parents=True)
    rows=[{'new_update':0},{'new_update':1,'exposure':{'chain_reward':100}},
          {'new_update':2,'exposure':{'threat_events':3,'chain_reward':.02}},
          {'new_update':3,'exposure':{'threat_events':9,'chain_reward':.1}}]
    (run/'updates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert E.native_reward_exposures(run/'u000002',2)==[rows[2]['exposure']]


def test_draw_kill_difference_uses_both_sides_and_only_draws():
    def row(seat,winner,kills):
        return dict(seat=seat,winner=winner,causal_kills=kills,cage_kills=[0,0],
            placed=[1,2],requests=[1,2],deaths=[0,0],self_deaths=[0,0],stored=[[0,0],[0,0]])
    v=E.summarize([row(0,-1,[4,1]),row(1,None,[3,1]),row(0,0,[10,0])])
    assert v['draw_kill_difference']==1
    assert v['draw_own_causal_kills']==5 and v['draw_opponent_causal_kills']==4
    assert v['win_minus_loss']==1
