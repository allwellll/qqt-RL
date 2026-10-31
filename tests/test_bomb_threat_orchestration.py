import json
import pytest
from pathlib import Path
from scripts import bomb_threat_owner as O
from scripts.bomb_threat_eval_owner import request_order


def test_baselines_precede_all_followup_requests():
    names=['bt_slot00_c000001_u000161','baseline_slot01_latest','baseline_u1120','baseline_parent']
    assert [p.stem for p in sorted(map(lambda n:Path(n+'.json'),names),key=request_order)] == [
        'baseline_parent','baseline_u1120','baseline_slot01_latest','bt_slot00_c000001_u000161']


def test_excluded_ancestry_inherited_not_overridden_by_role(tmp_path,monkeypatch):
    monkeypatch.setattr(O,'BASE',tmp_path)
    (tmp_path/'formal/root').mkdir(parents=True)
    (tmp_path/'protocols').mkdir()
    (tmp_path/'formal/root/INVALID_STARTING_POINT.json').write_text('{"reason":"latest"}')
    for child,parent in [('child','root'),('grandchild','child')]:
        (tmp_path/'protocols'/f'{child}.json').write_text(json.dumps({
            'role':'mandated_parent','sources':{'1':{'checkpoint':f'runs/bomb_threat_v1/formal/{parent}/u000080'}}}))
    assert O.invalid_ancestry('root')
    assert O.invalid_ancestry('child')
    assert O.invalid_ancestry('grandchild')
    assert not O.invalid_ancestry('new_unallocated')


def test_fresh_namespace_never_reuses_invalid_protocol(tmp_path,monkeypatch):
    monkeypatch.setattr(O,'BASE',tmp_path)
    (tmp_path/'protocols').mkdir()
    for n in (0,1,3): (tmp_path/'protocols'/f'bt_slot07_c{n:06d}.json').write_text('{}')
    assert O.next_chunk(7)==4
    assert O.PARENT_GPUS=={0,1,3,4,7}


def test_formal_and_prewarm_hash_the_same_pbt_worker_abi():
    config={'source_files':['scripts/env.py','scripts/collector.py']}
    assert O.worker_source_files(config,True)==[
        'scripts/env.py','scripts/collector.py','scripts/bomb_threat_pbt_train.py',
        'scripts/bomb_threat_pbt.py','scripts/bomb_threat_schedule.py']
    assert O.worker_source_files(config,False)==[
        'scripts/env.py','scripts/collector.py','scripts/bomb_threat_long_train.py',
        'scripts/bomb_threat_schedule.py']


def test_valid_chunk_terminal_continues_its_own_atomic_state(tmp_path,monkeypatch):
    """A normal COMPLETE must never fall through to fresh-parent allocation."""
    # This is an owner queue test, not a live-monitor integration. Each
    # extension owns module-local paths, so patching O.BASE alone cannot
    # isolate its refresh() side effects from an active training workspace.
    import importlib
    for name in ('bomb_threat_pbt2350','bomb_threat_a3sr_schedule',
                 'bomb_threat_round14','bomb_threat_a3sr_promotion',
                 'bomb_threat_r1_schedule','bomb_threat_r1_items_schedule','bomb_threat_round16',
                 'bomb_threat_round15','bomb_threat_a3sr_verdict'):
        module=importlib.import_module('scripts.'+name)
        monkeypatch.setattr(module,'refresh',lambda:None)
    from scripts import bomb_threat_r1_schedule as R1,bomb_threat_r1_items_schedule as I1
    monkeypatch.setattr(R1,'revision_hold',lambda:False)
    monkeypatch.setattr(I1,'apply_switch',lambda *a:None)
    base=tmp_path/'new';old=tmp_path/'old';base.mkdir();old.mkdir()
    monkeypatch.setattr(O,'BASE',base);monkeypatch.setattr(O,'OLD',old);monkeypatch.setattr(O,'ROOT',tmp_path)
    monkeypatch.setattr(O,'GPUS',[4]);monkeypatch.setattr(O,'lock',lambda _:None)
    monkeypatch.setattr(O,'workers',lambda:{})
    monkeypatch.setattr(O,'resources',lambda:[])  # no launch, only queue transition
    monkeypatch.setattr(O,'notify',lambda *a,**k:None)
    name='bt_slot04_c000000';run=base/'formal'/name;run.mkdir(parents=True)
    (base/'protocols').mkdir();recipe={'lr':.000075}
    (base/'protocols'/f'{name}.json').write_text(json.dumps({'arms':{name:recipe},'sources':{}}))
    done={'checkpoint':f'runs/new/formal/{name}/u000240','state_sha256':'abc','update':240}
    (run/'COMPLETE.json').write_text(json.dumps(done))
    (base/'SLOTS.json').write_text(json.dumps({'4':{'run':name,'chunk':0,'gpu':4}}))
    (old/'BOMB_THREAT_HANDOFF.json').write_text('{"armed_slots":[4]}')
    (base/'PROTOCOL.json').write_text('{"seeds":{"4":123}}')
    calls=[]
    def make(g,chunk,source,arm):
        calls.append((g,chunk,source,arm))
        return base/'protocols'/f'bt_slot{g:02d}_c{chunk:06d}.json'
    monkeypatch.setattr(O,'make_protocol',make)
    class EndCycle(Exception):pass
    def end(_):raise EndCycle
    monkeypatch.setattr(O.time,'sleep',end)
    with pytest.raises(EndCycle):O.main()
    assert calls==[(4,1,done,recipe)]
    slots=json.loads((base/'SLOTS.json').read_text())
    assert slots['4']['source']==done


def test_shaping_pause_is_slot_scoped_and_only_blocks_at_safe_boundary(tmp_path,monkeypatch):
    monkeypatch.setattr(O,'BASE',tmp_path)
    assert O.shaping_pause_request(0) is None
    (tmp_path/'shaping_pause_requests').mkdir()
    (tmp_path/'shaping_pause_requests/slot00_dev.json').write_text(json.dumps({
        'status':'SHAPING_PAUSE_REQUIRED','slot':0,'checkpoint':'example'}))
    assert O.shaping_pause_request(0)['checkpoint']=='example'
    assert O.shaping_pause_request(1) is None


def test_approved_pbt_override_only_unblocks_the_named_slot(tmp_path,monkeypatch):
    monkeypatch.setattr(O,'BASE',tmp_path)
    (tmp_path/'shaping_pause_requests').mkdir()
    (tmp_path/'shaping_pause_requests/slot03_dev.json').write_text(json.dumps({
        'status':'SHAPING_PAUSE_REQUIRED','slot':3,'checkpoint':'safe-boundary'}))
    assert O.active_shaping_pause(3,{'authorized_shaping_override':True}) is None
    assert O.active_shaping_pause(3,{})['checkpoint']=='safe-boundary'
    assert O.active_shaping_pause(4,{'authorized_shaping_override':True}) is None


def test_worker_consumes_persisted_native_only_override_without_erasing_alarm(tmp_path,monkeypatch):
    monkeypatch.setattr(O,'BASE',tmp_path);monkeypatch.setattr(O,'ROOT',tmp_path)
    (tmp_path/'shaping_pause_requests').mkdir();(tmp_path/'protocols').mkdir()
    request={'status':'SHAPING_PAUSE_REQUIRED','slot':3,'checkpoint':'old-shaped-branch'}
    alarm=tmp_path/'shaping_pause_requests/slot03_dev.json';alarm.write_text(json.dumps(request))
    run='bt_slot03_c000012';recipe={'zero_stock_shaping':True,
        'threat_weight':0.,'topology_weight':0.,'chain_weight':0.}
    (tmp_path/'protocols'/f'{run}.json').write_text(json.dumps({'arms':{run:recipe}}))
    (tmp_path/'decision.json').write_text(json.dumps({'status':'AUTHORIZED','slots':{'3':{}}}))
    (tmp_path/'SLOTS.json').write_text(json.dumps({'3':{'run':run,
        'authorized_shaping_override':True,'authorized_shaping_override_basis':'decision.json'}}))
    assert O.shaping_pause_request(3) is None
    assert json.loads(alarm.read_text())==request
    recipe['threat_weight']=.02
    (tmp_path/'protocols'/f'{run}.json').write_text(json.dumps({'arms':{run:recipe}}))
    assert O.shaping_pause_request(3)==request


def test_pbt_protocol_records_source_reward_variant_and_rng_rekey(tmp_path,monkeypatch):
    root=tmp_path
    base=root/'runs/bomb_threat_v1';(base/'protocols').mkdir(parents=True)
    (base/'PROTOCOL.json').write_text(json.dumps({
        'seeds':{'3':2026112903},'common':{'envs_per_device':2048},'source_files':[],
        'chunk_updates':320,'rule_version':'stock-window-v1','reward_version':'bomb-threat-v1',
        'training_spawn_manifest':'runs/spawns.json'}))
    (root/'runs/combat_mix_v1').mkdir(parents=True)
    (root/'runs/combat_mix_v1/PROTOCOL.json').write_text(json.dumps({'sources':{'2026103051':{'checkpoint':'fixed','state_sha256':'x'}}}))
    source_run='bt_slot00_c000003'
    source_protocol={'rule_version':'stock-window-v1','reward_version':'bomb-threat-v1',
        'sources':{'2026112900':{'checkpoint':'origin'}},'arms':{source_run:{'lr':4.8e-5}}}
    (base/'protocols'/f'{source_run}.json').write_text(json.dumps(source_protocol))
    monkeypatch.setattr(O,'BASE',base);monkeypatch.setattr(O,'ROOT',root)
    recipe={'lr':7.5e-5,'zero_stock_shaping':True,'threat_weight':0.0,'topology_weight':0.0,
        'chain_weight':0.0,'zero_shaping_weights':['threat','topology','chain']}
    source={'checkpoint':f'runs/bomb_threat_v1/formal/{source_run}/u000360','update':360,
        'state_sha256':'state-sha','policy_sha256':'policy-sha'}
    metadata=O.pbt_source_metadata(source,recipe,3,2026112903)
    assert metadata['source_training_reward_variant']=='bomb-threat-v1-with-stock-shaping'
    assert metadata['training_reward_variant']=='bomb-threat-v1-native-only-zero-stock-shaping'
    assert metadata['pbt_rekey_source'] is True
    assert metadata['pbt_parent']==source['checkpoint']
    assert metadata['zero_shaping_weights']==['threat','topology','chain']

    pbt_protocol=dict(source_protocol,arms={source_run:recipe},
        training_reward_variant='bomb-threat-v1-native-only-zero-stock-shaping')
    (base/'protocols/bt_slot03_c000011.json').write_text(json.dumps(pbt_protocol))
    continuation=dict(source,checkpoint='runs/bomb_threat_v1/formal/bt_slot03_c000011/u000680',update=680)
    metadata=O.pbt_source_metadata(continuation,recipe,3,2026112903)
    assert metadata['source_training_reward_variant']=='bomb-threat-v1-native-only-zero-stock-shaping'
    assert metadata['pbt_rekey_source'] is False


def test_passive_gate_error_is_persistent_and_does_not_kill_queue(tmp_path,monkeypatch):
    from scripts import bomb_threat_eval_owner as EO
    from scripts import bomb_threat_gate_report as GR
    monkeypatch.setattr(EO,'BASE',tmp_path)
    def fail():raise ValueError('bad denominator')
    monkeypatch.setattr(GR,'refresh_all',fail)
    assert EO.refresh_gates() is False
    assert json.loads((tmp_path/'GATE_REPORT_FAILURE.json').read_text())['raw_games_preserved']
    monkeypatch.setattr(GR,'refresh_all',lambda: {})
    assert EO.refresh_gates() is True
    assert (tmp_path/'GATE_REPORT_RECOVERY.json').exists()
    assert not (tmp_path/'GATE_REPORT_FAILURE.json').exists()
