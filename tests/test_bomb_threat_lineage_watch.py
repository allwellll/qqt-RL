import pytest
from scripts import bomb_threat_a3sr_lineage_watch as W


def test_confirmation_requires_both_full_128_game_points_and_includes_equality():
    def point(n,w):return {'games':n,'by_seat':{'P0':{'games':n//2},'P1':{'games':n//2,'win_minus_loss':w}}}
    assert W.confirmation_gate({'2280':point(128,0)})['status']=='MISSING'
    assert W.confirmation_gate({'2280':point(128,0),'2320':point(120,0)})['status']=='MISSING'
    assert W.confirmation_gate({'2280':point(128,1),'2320':point(128,3)})['status']=='BLOCK'
    assert W.confirmation_gate({'2280':point(128,1),'2320':point(128,4)})['status']=='PASS'
    assert W.confirmation_gate({'2280':point(128,-2),'2320':point(128,5)})['status']=='BLOCK'


def test_boundary_gate_is_slot_scoped_and_missing_never_blocks(tmp_path,monkeypatch):
    monkeypatch.setattr(W,'OUT',tmp_path)
    assert W.boundary_block(5) is None
    W.atomic_json(tmp_path/'STOP_AT_CHUNK_BOUNDARY.json',{'status':'BLOCK','slot':5,'reason':'confirmation'})
    assert W.boundary_block(5)['reason']=='confirmation'
    for slot in (0,1,3,4,6,7):assert W.boundary_block(slot) is None


def test_rolling_stop_is_latched_and_not_erased_by_later_pass(tmp_path,monkeypatch):
    monkeypatch.setattr(W,'OUT',tmp_path)
    W.latch_stop({'status':'BLOCK','failures':['P1_win_minus_loss']},{'status':'MISSING'})
    assert W.boundary_block(5)['reasons']==['rolling_holdout_BLOCK']
    W.latch_stop({'status':'PASS'},{'status':'PASS'})
    assert W.boundary_block(5)['status']=='BLOCK'


def test_confirmation_freezes_same_new_seed_list_and_fresh_reuses_original_seeds():
    p=W.freeze()
    assert p['confirmation']['updates']==[2280,2320]
    assert p['confirmation']['games_per_point']==128
    assert p['confirmation']['seed_start']==2026173000
    assert p['fresh']['slots']==[0,3,4,7] and p['fresh']['update']==2400
    assert [p['fresh']['tiers'][k]['seed'] for k in ('hard','normal','easy')]==[2026142900,2026143900,2026144900]
    assert p['training_changes']==[] and p['GPU_training_cards_used']==[]


def test_training_owner_consumes_confirmation_block_before_creating_successor(tmp_path,monkeypatch):
    from scripts import bomb_threat_a3sr_promotion as P
    import scripts.iter_pbt_owner as I
    base=tmp_path/'runs/bomb_threat_v1';out=base/'promotion'
    monkeypatch.setattr(P,'ROOT',tmp_path);monkeypatch.setattr(P,'BASE',base);monkeypatch.setattr(P,'OUT',out);monkeypatch.setattr(P,'AUTH',out/'AUTHORIZATION.json')
    monkeypatch.setattr(W,'OUT',base/'watch');monkeypatch.setattr(I,'notify',lambda *a,**k:None)
    W.atomic_json(P.AUTH,{'status':'AUTHORIZED','boundary_runs':{}})
    run='bt_slot05_c000032';done={'checkpoint':f'runs/bomb_threat_v1/formal/{run}/u002560','update':2560,'state_sha256':'frozen'}
    W.atomic_json(base/'formal'/run/'COMPLETE.json',done)
    W.atomic_json(base/'protocols'/f'{run}.json',{'chunk_start_update':2240,'chunk_updates':320})
    W.atomic_json(W.OUT/'STOP_AT_CHUNK_BOUNDARY.json',{'status':'BLOCK','slot':5,'reason':'confirmed P1 mean<=2'})
    assert P.apply_switch(5,{'run':run,'a3sr_promotion_applied':True},done,33) is None
    assert P.read(out/'blocked'/f'{run}.json')['post_publish_lineage_watch']['reason']=='confirmed P1 mean<=2'
    assert not (base/'protocols/bt_slot05_c000033.json').exists()


def test_old_fixed_u2400_never_substitutes_r1_child(tmp_path,monkeypatch):
    from scripts import bomb_threat_a3sr_lineage_watch as W,bomb_threat_owner as O
    from scripts.population_run_state import atomic_json
    b=tmp_path/'runs/bomb_threat_v1';run='bt_slot04_c000034'
    monkeypatch.setattr(W,'BASE',b)
    atomic_json(b/'SLOTS.json',{'4':{'run':run}})
    atomic_json(b/'formal'/run/'u002400/COMMITTED.json',{'checkpoint':f'runs/bomb_threat_v1/formal/{run}/u002400','metrics':{'finite':True}})
    atomic_json(b/'protocols'/f'{run}.json',{'namespace_id':'A3SR-ATTACK-R1-20261030','arms':{run:{'iter2_arm':'A3SR'}}})
    monkeypatch.setattr(O,'causal_lineage_limits',lambda _:{run:float('inf')})
    assert W.find_checkpoint(4,2400) is None
