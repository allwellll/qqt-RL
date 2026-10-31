import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import r3_resume2110_20261031 as resume
from scripts.population_run_state import atomic_json,sha

def reader(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default

def test_delayed_historical_gate_kept_but_not_reconsumed(tmp_path):
    gate=tmp_path/'evaluation/r3_sp55_c000000_u003200/BOT_GATE.json'
    atomic_json(gate,dict(status='ROLLBACK'))
    atomic_json(tmp_path/'RESUME_2110.json',dict(status='CONSUMED',start_update=3316,superseded_bot_gates={}))
    owner=SimpleNamespace(OUT=tmp_path,read=reader)
    assert resume.superseded_bot_gate(owner,gate)
    assert reader(gate)['status']=='ROLLBACK'
    assert reader(tmp_path/'RESUME_2110.json')['superseded_bot_gates'][str(gate)]==sha(gate)

def test_new_checkpoint_guard_stays_active(tmp_path):
    gate=tmp_path/'evaluation/r3_sp55_c000000_u003320/BOT_GATE.json'
    atomic_json(gate,dict(status='ROLLBACK'))
    atomic_json(tmp_path/'RESUME_2110.json',dict(status='CONSUMED',start_update=3316,superseded_bot_gates={}))
    assert not resume.superseded_bot_gate(SimpleNamespace(OUT=tmp_path,read=reader),gate)

def test_old_gate_cannot_change_after_accepted_hash(tmp_path):
    gate=tmp_path/'evaluation/r3_sp55_c000000_u003200/BOT_GATE.json'
    atomic_json(gate,dict(status='ROLLBACK'))
    atomic_json(tmp_path/'RESUME_2110.json',dict(status='CONSUMED',start_update=3316,superseded_bot_gates={str(gate):sha(gate)}))
    atomic_json(gate,dict(status='PASS'))
    with pytest.raises(AssertionError):resume.superseded_bot_gate(SimpleNamespace(OUT=tmp_path,read=reader),gate)

def test_missing_authorization_is_noop(tmp_path):
    assert not resume.consume(SimpleNamespace(OUT=tmp_path,read=reader))

@pytest.mark.parametrize('busy',[True,False])
def test_control_handoff_requires_completed_verified_exclusive_card(tmp_path,monkeypatch,busy):
    monkeypatch.chdir(tmp_path)
    from scripts import r3_selfplay_owner_20261031 as sp
    monkeypatch.setattr(sp,'compute_app_count',lambda gpu:int(busy))
    out=Path('runs/r3_items_route_20261030');slots={};queue={}
    for arm,gpu in [('a',0),('b',5)]:
        cp=out/'formal'/f'r3_{arm}_ownside_c000001/u004002';cp.mkdir(parents=True)
        (cp/'state.pkl').write_bytes(b'state');(cp/'policy.pt').write_bytes(b'policy')
        atomic_json(cp/'COMMITTED.json',dict(update=4002,state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt')))
        atomic_json(cp.parent/'COMPLETE.json',dict(update=4002))
        run=f'old_{arm}';folder=Path('runs/bomb_threat_v1/formal')/run
        saved=folder/'u000640';saved.mkdir(parents=True)
        (saved/'state.pkl').write_bytes(b'old-state');(saved/'policy.pt').write_bytes(b'old-policy')
        atomic_json(folder/'COMPLETE.json',dict(update=640,checkpoint=str(saved),state_sha256=sha(saved/'state.pkl'),policy_sha256=sha(saved/'policy.pt')))
        atomic_json(folder/'NEXT_STEP_u000640.json',dict(digest='recorded'))
        atomic_json(out/'boundary_blocks'/f'r3_{arm}_ownside.json',dict(status='BLOCK'))
        slots[arm]=dict(run=cp.parent.name,gpu=2)
        queue[arm]=dict(gpu=gpu,source=str(cp),waiting_run=run)
    atomic_json(out/'R3_SLOTS.json',slots)
    atomic_json(resume.AUTH.parent/'R3_ALLOCATION_QUEUE.json',queue)
    calls=[];owner=SimpleNamespace(OUT=out,read=reader,notify=lambda *a,**k:calls.append(a))
    resume.allocate_controls(owner)
    actual=reader(out/'R3_SLOTS.json')
    assert actual['a']['gpu']==(2 if busy else 0)
    assert reader(out/'boundary_blocks/r3_a_ownside.json')['status']==('BLOCK' if busy else 'SUPERSEDED_BY_2110_RESUME')
    assert len(calls)==(0 if busy else 2)
