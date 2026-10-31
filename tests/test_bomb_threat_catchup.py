import json
from pathlib import Path
from scripts import bomb_threat_eval_owner as E


def request(tmp_path,name,slot,update):
    p=tmp_path/(name+'.json')
    p.write_text(json.dumps(dict(name=name,checkpoint=f'runs/bomb_threat_v1/formal/bt_slot{slot:02d}_c000009/u{update:06d}',split='dev')))
    return p


def test_latest_per_slot_first_then_eighty_backfill_boundary_is_technical(tmp_path):
    paths=[request(tmp_path,'s0_u1',0,721),request(tmp_path,'s0_latest',0,720),
           request(tmp_path,'s0_back80',0,640),request(tmp_path,'s0_other40',0,680),
           request(tmp_path,'s1_latest',1,760)]
    ordered=E.catchup_order(paths,active_runs={})
    assert set(p.stem for p in ordered[:2])=={'s0_latest','s1_latest'}
    assert [p.stem for p in ordered[2:]]==['s0_back80','s0_other40','s0_u1']
    assert E.request_kind(json.loads(paths[0].read_text()))=='technical_boundary'


def test_resource_plan_has_twelve_disjoint_three_core_lanes():
    lanes=E.cpu_lanes()
    assert len(lanes)>=12
    flat=[c for lane in lanes for c in lane]
    assert len(flat)==len(set(flat))
    reserved=set(range(32,48))|set(range(72,112))|set(range(128,136))
    assert not reserved.intersection(flat)
    assert all(len(lane)==3 for lane in lanes)


def test_holdout_gate_request_never_demoted_as_boundary(tmp_path):
    p=request(tmp_path,'candidate_holdout',0,721)
    value=json.loads(p.read_text());value['split']='holdout'
    assert E.request_kind(value)=='holdout'


def test_inflight_latest_does_not_promote_its_old_history_above_other_slot(tmp_path):
    old=request(tmp_path,'slot0_old',0,320)
    inflight=request(tmp_path,'slot0_active_latest',0,720)
    new=request(tmp_path,'slot1_latest',1,720)
    assert E.catchup_order([old,new],reference_paths=[old,inflight,new],active_runs={})==[new,old]


def test_pbt_fork_latest_is_not_starved_by_retired_higher_update(tmp_path):
    old=request(tmp_path,'retired_u1080',3,1080)
    current=request(tmp_path,'new_u400',3,400)
    d=json.loads(current.read_text());d['checkpoint']='runs/bomb_threat_v1/formal/bt_slot03_c000012/u000400'
    current.write_text(json.dumps(d))
    limits={'slot03':{'bt_slot03_c000012':float('inf'),'bt_slot00_c000003':360}}
    assert E.catchup_order([old,current],reference_paths=[old,current],active_runs=limits)==[current,old]


def test_eval_lanes_do_not_share_smt_physical_cores_with_training():
    reserved=set(range(32,48))|set(range(72,112))
    physical={c%96 for c in reserved}
    assert not physical.intersection(c%96 for lane in E.cpu_lanes() for c in lane)


def test_amend2_top_holdout_priority_precedes_other_pending_jobs(tmp_path,monkeypatch):
    monkeypatch.setattr(E,'BASE',tmp_path)
    top=request(tmp_path,'mandatory_u960_holdout',0,960)
    v=json.loads(top.read_text());v['split']='holdout';top.write_text(json.dumps(v))
    latest=request(tmp_path,'latest',1,1800)
    (tmp_path/'GATE_AMEND2_HOLDOUT_PRIORITY.json').write_text(json.dumps({'names':[top.stem]}))
    assert E.catchup_order([latest,top],active_runs={})==[top,latest]
