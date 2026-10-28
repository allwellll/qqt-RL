import json
from scripts import combat_mix_exposure_summary as S

def test_nominal_legacy_dispatch_is_never_presented_as_executed_exposure(tmp_path,monkeypatch):
    monkeypatch.setattr(S,'BASE',tmp_path)
    out=tmp_path/'formal/legacy_mix_s1';out.mkdir(parents=True)
    row=dict(new_update=161,actual_environment_steps_this_update=8,actual_optimizer_steps_this_update=1,
        exposure=dict(scene_opponent_counts=[[[2],[1],[1],[4]]],league_counts=[[4,0,0,0,0,0]]))
    (out/'updates.jsonl').write_text(json.dumps(row)+'\n')
    S.main();d=json.loads((tmp_path/'TRAINING_EXPOSURE.json').read_text())['runs']['legacy_mix_s1']
    assert d['executed_source_live_tick_fraction'] is None
    assert d['nominal_dispatch_live_tick_fraction']['league']==.5
    assert 'missing_reason' in d


def test_override_denominator_is_versioned_and_actual_ring_separate(tmp_path,monkeypatch):
    monkeypatch.setattr(S,'BASE',tmp_path)
    out=tmp_path/'formal/legacy_mix_s1';out.mkdir(parents=True)
    row=dict(new_update=164,actual_environment_steps_this_update=8,actual_optimizer_steps_this_update=1,
        exposure=dict(executed_source_live_ticks=[1,1,1,3,0,0,2],executed_league_member_live_ticks=[1,0,0,0,0,0,1,1,0,0,0]))
    (out/'updates.jsonl').write_text(json.dumps(row)+'\n')
    S.main();d=json.loads((tmp_path/'TRAINING_EXPOSURE.json').read_text())['runs']['legacy_mix_s1']
    assert d['executed_source_live_tick_fraction']['assist']==.25
    assert d['actual_current_ring_fixed_per_live_tick']==dict(current=.125,ring=.125,fixed=.125)
    assert d['executed_telemetry_updates']==1
