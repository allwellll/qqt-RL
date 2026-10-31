import pytest
import json
from scripts import r1_plateau0105_watch as watcher
from scripts.r1_plateau0105_watch import alert_pair, classify, post_exclusive


def point(update, bombs=61, kills=49, games=64):
    return dict(update=update,metrics=dict(bombs_per_game=bombs,causal_kills=kills,games=games))


def test_strict_bomb_and_kill_thresholds():
    assert alert_pair([point(40),point(80)])
    assert not alert_pair([point(40,bombs=60),point(80)])
    assert not alert_pair([point(40),point(80,kills=50)])


def test_one_high_bomb_point_or_more_kills_does_not_alarm():
    assert not alert_pair([point(4240,bombs=64.25,kills=55)])
    assert not alert_pair([point(4200),point(4240,bombs=64.25,kills=55)])


def test_missing_scheduled_point_is_not_consecutive():
    assert not alert_pair([point(40),point(120)])
    assert not alert_pair([point(40,games=63),point(80)])


@pytest.mark.parametrize('ratio,result',[(.60,'CO_TENANCY_BOTTLENECK_OBSERVED'),
    (.59,'INCONCLUSIVE_40_TO_60'),(.40,'INCONCLUSIVE_40_TO_60'),(.399,'FULL_128_TICK_PROFILE_REQUIRED')])
def test_threshold_boundaries(ratio,result):
    assert classify(ratio)==result


def test_solo_measurement_excludes_cold_and_crossing_cycles():
    history=[dict(new_update=i,unix=100+i*10,cycle_seconds=12) for i in range(1,9)]
    assert [r['new_update'] for r in post_exclusive(history,155,0)]==[7,8]
    # A new chunk must exclude its first2 updates even if GPU is already solo.
    assert [r['new_update'] for r in post_exclusive(history,0,5)]==[8]


def test_only_real_ancestry_and_parent_cutoff_are_eligible(tmp_path,monkeypatch):
    monkeypatch.setattr(watcher,'BASE',tmp_path)
    p=tmp_path/'protocols';p.mkdir()
    for run,source in [('current',dict(checkpoint=str(tmp_path/'formal/parent/u000080'),update=80)),
                       ('parent',dict(checkpoint='external/frozen_parent/u000000',update=0))]:
        (p/f'{run}.json').write_text(json.dumps(dict(continuation_source=source)))
    (p/'unrelated_same_update.json').write_text('{}')
    assert watcher.lineage_limits('current')==dict(current=None,parent=80)


def test_smoke_cotenant_cannot_validate_formal_comparison(tmp_path, monkeypatch):
    base=tmp_path/'base';out=tmp_path/'out';out.mkdir()
    monkeypatch.setattr(watcher,'BASE',base)
    monkeypatch.setattr(watcher,'OUT',out)
    monkeypatch.setattr(watcher,'R2',tmp_path/'r2')
    monkeypatch.setattr(watcher,'verify_file',lambda *args:None)
    monkeypatch.setattr(watcher,'gpu_apps',lambda gpu:['worker:r2b'] if gpu==4 else ['worker:r2c','r1:next'])
    files={str(base/'iter2/r1_attack_20261030/stop2255/slot04_STOPPED.json'):
        dict(status='COMPLETE_CHUNK_THEN_RETIRED',last_atomic=dict(update=4240,passed=True,checkpoint='checkpoint',state_sha256='s',policy_sha256='p')),
        str(tmp_path/'r2/R2_SLOTS.json'):dict(b=dict(worker_module='worker',run='r2b')),
        str(base/'formal/next/LAUNCH.json'):dict(stage='smoke',unix=100)}
    monkeypatch.setattr(watcher,'read',lambda path,default=None:files.get(str(path),default))
    assert watcher.retirement_diagnostic({'4':{'run':'retired'},'7':{'run':'next'}})['status']=='WAITING_R1_SLOT07_FORMAL_COTENANT'
    files[str(base/'formal/next/LAUNCH.json')]=dict(stage='formal',unix=100)
    monkeypatch.setattr(watcher,'update_rows',lambda path:[dict(unix=120,new_update=4243)])
    result=watcher.retirement_diagnostic({'4':{'run':'retired'},'7':{'run':'next'}})
    assert result['status']=='WAITING_SLOT07_TWO_FORMAL_UPDATES'
    assert result['formal_updates']==[4243]
