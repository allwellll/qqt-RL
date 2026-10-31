import json
from scripts import bomb_threat_round_report as R


def test_current_pbt_lineage_excludes_retired_slot_branch_and_caps_donor(tmp_path,monkeypatch):
    monkeypatch.setattr(R,'BASE',tmp_path)
    (tmp_path/'protocols').mkdir()
    for name,cp,u in [
        ('bt_slot03_c000012','runs/bomb_threat_v1/formal/bt_slot00_c000003/u000360',360),
        ('bt_slot00_c000003','runs/bomb_threat_v1/formal/bt_slot00_c000002/u000320',320)]:
        (tmp_path/'protocols'/(name+'.json')).write_text(json.dumps({'sources':{'1':{'checkpoint':cp,'update':u}}}))
    limits=R.lineage_limits('bt_slot03_c000012')
    assert limits['bt_slot00_c000003']==360
    assert limits['bt_slot00_c000002']==320
    assert 'bt_slot03_c000010' not in limits
    assert 400>limits['bt_slot00_c000003']
