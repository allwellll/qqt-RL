import copy
import pickle
from pathlib import Path

import pytest

from scripts.population_run_state import atomic_json, sha
from scripts import bun_native_v2_owner as r2
from scripts import r3_selfplay_owner_20261031 as sp
from scripts import r3_sp_capacity_20261031 as capacity


def pause_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(r2,'OUT',tmp_path)
    monkeypatch.setattr(r2,'notification',lambda *a,**kw:None)
    monkeypatch.setattr(r2,'processes',lambda *a:iter(()))
    selection=tmp_path/'selection.json';atomic_json(selection,{'status':'SELECTED'})
    protocol=tmp_path/'protocol.json';atomic_json(protocol,dict(chunk_start_update=100,chunk_updates=640))
    entry=dict(run='r2_a_test',gpu=3,protocol=str(protocol))
    request=dict(status='AUTHORIZED',selection=str(selection),selection_sha256=sha(selection),
                 arms={'a':dict(run=entry['run'],target_update=740)})
    atomic_json(tmp_path/'controls/SP_CAPACITY_PAUSE_AFTER_CHUNK.json',request)
    cp=tmp_path/'formal'/entry['run']/'u000740';cp.mkdir(parents=True)
    payload=dict(params={'actor':{},'critic':{}},adam={},states={},aux={},keys={},
                 combat_ring=[],combat_ring_cursor=0,update=740)
    (cp/'state.pkl').write_bytes(pickle.dumps(payload));(cp/'policy.pt').write_bytes(b'actor')
    done=dict(checkpoint=str(cp),update=740,state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'))
    atomic_json(cp.parent/'NEXT_STEP_u000740.json',dict(digest='s',Adam_digest='a',full_environment_digest='e',ring_digest='r'))
    return entry,cp,done


def test_no_pause_or_successor_claim_at_partial_save(tmp_path,monkeypatch):
    entry,cp,done=pause_fixture(tmp_path,monkeypatch)
    atomic_json(cp/'COMMITTED.json',done)
    assert r2.capacity_pause_at_complete_chunk('a',entry) is None
    assert 'suspended' not in entry
    assert not (tmp_path/'capacity_pauses/a.json').exists()
    assert r2.capacity_pause_at_complete_chunk('b',dict(entry,run='unselected')) is None


def test_waits_for_terminal_worker_exit_and_preserves_full_state(tmp_path,monkeypatch):
    entry,cp,done=pause_fixture(tmp_path,monkeypatch)
    atomic_json(cp.parent/'COMPLETE.json',done)
    monkeypatch.setattr(r2,'processes',lambda *a:iter([object()]))
    assert r2.capacity_pause_at_complete_chunk('a',entry)=='WAIT_COMPLETE_CHUNK_WORKER_EXIT'
    assert 'suspended' not in entry
    monkeypatch.setattr(r2,'processes',lambda *a:iter(()))
    assert r2.capacity_pause_at_complete_chunk('a',entry)=='PAUSED_COMPLETE_CHUNK_FOR_SP'
    assert r2.read(tmp_path/'capacity_pauses/a.json')['successor_blocked']
    assert sha(cp/'state.pkl')==done['state_sha256']
    assert sha(cp/'policy.pt')==done['policy_sha256']
    assert entry['suspended']=='SP_CAPACITY_AUTHORIZED_20261031'


def test_corrupt_checkpoint_never_releases_card(tmp_path,monkeypatch):
    entry,cp,done=pause_fixture(tmp_path,monkeypatch)
    atomic_json(cp.parent/'COMPLETE.json',done)
    (cp/'state.pkl').write_bytes(b'broken')
    with pytest.raises(AssertionError):r2.capacity_pause_at_complete_chunk('a',entry)
    assert 'suspended' not in entry
    assert not (tmp_path/'capacity_pauses/a.json').exists()


def model_metrics(wl):
    return {c:{s:dict(games=64,wins=32+wl//2,draws=0,losses=32-wl//2,
                     win_minus_loss=wl,self_deaths=0) for s in ('P0','P1')}
            for c in ('hunter_hard','coop_hard')}


def test_shared_weakest_never_steals_protected_bt_card(tmp_path,monkeypatch):
    monkeypatch.setattr(capacity,'OUT',tmp_path)
    protocol=dict(models={'r2_a':{'entry':{'gpu':3}},'r2_b':{'entry':{'gpu':4}},'r2_c':{'entry':{'gpu':7}}},primary='P1')
    atomic_json(tmp_path/'PROTOCOL.json',protocol)
    metrics=dict(parent=model_metrics(45),r2_a=model_metrics(25),r2_b=model_metrics(30),r2_c=model_metrics(10))
    d=capacity.select(protocol,metrics)
    assert d['pause_models']==['r2_c','r2_a'] and d['exclusive_gpu']==3
    metrics['r2_a']=model_metrics(43);metrics['r2_b']=model_metrics(43)
    d=capacity.select(protocol,metrics)
    assert d['status']=='BLOCKED_NO_AUTHORIZED_EXCLUSIVE_CARD'
    assert d['pause_models']==['r2_c'] and d['exclusive_gpu'] is None


@pytest.mark.parametrize('attempt',[None,'gpu3_guard1600'])
def test_exclusive_handoff_requires_driver_release_and_new_smoke_restore(tmp_path,monkeypatch,attempt):
    monkeypatch.setattr(sp,'OUT',tmp_path)
    monkeypatch.setattr(sp,'processes',lambda *a:iter(()))
    monkeypatch.setattr(sp,'reference_snapshot',lambda:{})
    monkeypatch.setattr(sp,'notify',lambda *a,**kw:None)
    folder=tmp_path/'formal/r3_sp55_c000000';cp=folder/'u003040';cp.mkdir(parents=True)
    (cp/'state.pkl').write_bytes(b'complete SP state');(cp/'policy.pt').write_bytes(b'actor')
    cm=dict(update=3040,state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'),
            metrics={'formal_updates_discarded':0})
    atomic_json(cp/'COMMITTED.json',cm);atomic_json(folder/'NEXT_STEP_u003040.json',{'digest':'token'})
    atomic_json(tmp_path/'REGISTRATION.json',dict(entry=dict(run='r3_sp55_c000000',gpu=6,seed=2026323001)))
    atomic_json(tmp_path/'RESOURCE_PAUSE.json',{'status':'PAUSE'})
    receipt=tmp_path/'R2_PAUSED.json';atomic_json(receipt,dict(status='PAUSED_COMPLETE_CHUNK',gpu=3,source=dict(cm,checkpoint=str(cp))))
    request=dict(status='AUTHORIZED',exclusive_gpu=3,source_checkpoint=str(cp),authorization='steer',
                 paused_R2=[dict(path=str(receipt),sha256=sha(receipt),receipt=sp.read(receipt))])
    if attempt:
        request['attempt']=attempt
        old=tmp_path/'attempts/dedicated_gpu3/OLD.json';atomic_json(old,{'keep':True})
        atomic_json(tmp_path/'DEDICATED_GPU_RESTORE_ACCEPTANCE.json',{'previous':True})
    atomic_json(tmp_path/'DEDICATED_CAPACITY_RESUME.json',request)
    launched=[]
    def launch(entry,phase):
        launched.append((entry['gpu'],phase));atomic_json(folder/'LAUNCH.json',{'gpu':entry['gpu'],'phase':phase})
    monkeypatch.setattr(sp,'launch',launch)
    monkeypatch.setattr(sp,'compute_app_count',lambda gpu:1)
    assert sp.consume_dedicated_capacity() and launched==[]
    assert sp.read(tmp_path/'REGISTRATION.json')['entry']['gpu']==6
    monkeypatch.setattr(sp,'compute_app_count',lambda gpu:0)
    assert sp.consume_dedicated_capacity() and launched==[(3,'RESTORE')]
    assert (tmp_path/'RESOURCE_PAUSE.json').exists()
    atomic_json(folder/'RESTORE.json',dict(checkpoint=str(cp),passed=True,fresh_process=True,
        next_step_exact=True,Adam_exact=True,full_env_aux_RNG_exact=True,ring_exact=True))
    assert not sp.consume_dedicated_capacity()
    assert sp.read(tmp_path/'DEDICATED_CAPACITY_RESUME.json')['status']=='CONSUMED'
    assert (tmp_path/'DEDICATED_GPU_RESTORE_ACCEPTANCE.json').exists()
    assert not (folder/'RESTORE.json').exists()  # fresh u3042 still required
    assert not (tmp_path/'RESOURCE_PAUSE.json').exists()
    if attempt:
        assert sp.read(old)=={'keep':True}
        assert sp.read(tmp_path/'attempts'/attempt/'PREVIOUS_DEDICATED_GPU_RESTORE_ACCEPTANCE.json')=={'previous':True}


def test_shared_card_second_pause_does_not_delay_ready_exclusive_card(tmp_path,monkeypatch):
    monkeypatch.setattr(capacity,'OUT',tmp_path/'comparison')
    monkeypatch.setattr(capacity,'R2',tmp_path/'r2')
    monkeypatch.setattr(capacity,'SP',tmp_path/'sp')
    protocol=dict(authorization='steer',models={'r2_c':{'entry':{'gpu':7}},'r2_a':{'entry':{'gpu':3}}})
    decision=dict(pause_models=['r2_c','r2_a'],exclusive_gpu=3)
    atomic_json(tmp_path/'r2/capacity_pauses/a.json',dict(status='PAUSED_COMPLETE_CHUNK',gpu=3))
    capacity.advance_handoff(protocol,decision)
    request=capacity.read(tmp_path/'sp/DEDICATED_CAPACITY_RESUME.json')
    assert request['status']=='AUTHORIZED' and request['exclusive_gpu']==3
    assert len(request['paused_R2'])==1
    assert not (tmp_path/'r2/capacity_pauses/c.json').exists()


def test_restart_reuses_selection_without_repeating_evaluation(tmp_path,monkeypatch):
    monkeypatch.setattr(capacity,'OUT',tmp_path)
    atomic_json(tmp_path/'PROTOCOL.json',dict(status='FROZEN'))
    decision=dict(status='SELECTED',protocol_sha256=sha(tmp_path/'PROTOCOL.json'))
    atomic_json(tmp_path/'SELECTION.json',decision)
    original=sha(tmp_path/'SELECTION.json');calls=[]
    monkeypatch.setattr(capacity,'write_pause_request',lambda p,d:calls.append(d))
    def advance(p,d):atomic_json(tmp_path/'HANDOFF_COMPLETE.json',dict(status='PASS'))
    monkeypatch.setattr(capacity,'advance_handoff',advance)
    monkeypatch.setattr(capacity.time,'sleep',lambda _:None)
    monkeypatch.setattr(capacity,'export',lambda *a:pytest.fail('must not re-export on handoff recovery'))
    capacity.main()
    assert calls==[decision] and sha(tmp_path/'SELECTION.json')==original


def test_pause_request_hash_is_stable_on_recovery(tmp_path,monkeypatch):
    monkeypatch.setattr(capacity,'OUT',tmp_path/'comparison')
    monkeypatch.setattr(capacity,'R2',tmp_path/'r2')
    atomic_json(tmp_path/'comparison/SELECTION.json',dict(status='SELECTED'))
    protocol=dict(authorization='steer',models={'r2_a':dict(entry=dict(run='run',gpu=3),
        target_chunk_end=740,source=dict(checkpoint='point'))})
    decision=dict(status='SELECTED',pause_models=['r2_a'],exclusive_gpu=3)
    capacity.write_pause_request(protocol,decision)
    request=tmp_path/'r2/controls/SP_CAPACITY_PAUSE_AFTER_CHUNK.json';original=sha(request)
    capacity.write_pause_request(protocol,decision)
    assert sha(request)==original


def test_no_exclusive_card_still_queues_only_authorized_weakest_pause(tmp_path,monkeypatch):
    monkeypatch.setattr(capacity,'OUT',tmp_path/'comparison')
    monkeypatch.setattr(capacity,'R2',tmp_path/'r2')
    monkeypatch.setattr(capacity.subprocess,'run',lambda *a,**kw:None)
    atomic_json(tmp_path/'comparison/SELECTION.json',dict(status='BLOCKED_NO_AUTHORIZED_EXCLUSIVE_CARD'))
    protocol=dict(authorization='steer',models={'r2_c':dict(entry=dict(run='r2_c_run',gpu=7),
        target_chunk_end=740,source=dict(checkpoint='point'))})
    decision=dict(status='BLOCKED_NO_AUTHORIZED_EXCLUSIVE_CARD',pause_models=['r2_c'],exclusive_gpu=None,scores={})
    capacity.record_capacity_block(protocol,decision)
    request=capacity.read(tmp_path/'r2/controls/SP_CAPACITY_PAUSE_AFTER_CHUNK.json')
    assert set(request['arms'])=={'c'} and request['arms']['c']['gpu']==7
    assert capacity.read(tmp_path/'comparison/BLOCKED.json')['protected_BT_slots_untouched']
    assert not (tmp_path/'comparison/HANDOFF_COMPLETE.json').exists()
