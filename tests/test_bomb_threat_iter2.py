import json
import pytest
from scripts import bomb_threat_iter2 as I


def fixture(tmp_path):
    path=tmp_path/'games.json'
    rows=[dict(seed=100+i,seat=i%2,ticks=2400,done=True) for i in range(8)]
    episodes=[dict(seed=100+i,team=i%2,game=i,ticks=2400) for i in range(8)]
    path.write_text(json.dumps(dict(model_json_sha256='model',results={'natural_full2400':{'episodes':episodes}})))
    (tmp_path/'games.json.outcome.json').write_text(json.dumps(dict(episodes=rows,engine='production_JS',teacher=False,rule_version='stock-window-v1')))
    (tmp_path/'games.json.actions.json').write_text(json.dumps(dict(tau=.5,record=False,rows=[])))
    return path


def test_accept_requires_full_matched_games_and_exact_model(tmp_path,monkeypatch):
    monkeypatch.setattr(I,'ROOT',tmp_path);p=fixture(tmp_path)
    rows,proof=I.accept_shard(p,'model',0,100,.5)
    assert len(rows)==proof['games']==8
    with pytest.raises(AssertionError):I.accept_shard(p,'different_model',0,100,.5)
    with pytest.raises(AssertionError):I.accept_shard(p,'model',8,100,.5)
    f=tmp_path/'games.json.outcome.json';v=json.loads(f.read_text());v['episodes'][0]['done']=False;f.write_text(json.dumps(v))
    with pytest.raises(AssertionError):I.accept_shard(p,'model',0,100,.5)


def test_rank_aggregates_legal_denominators_and_zero_top1(tmp_path):
    p=tmp_path/'games.json';row=dict(seed=100,seat=0,phase='combat',ticks=3,bomb_legal_ticks=2,
        bomb_top1_count=0,bomb_top1_gap_sum=0,bomb_probability_sum=.7,rank_counts=[0,2,0],
        top_ability_counts=[3,0,0],movement_counts=[1,1,0,0,1])
    (tmp_path/'games.json.actions.json').write_text(json.dumps(dict(rows=[row],examples=[])))
    v=I.summarize_rank([p])['by_phase']['combat']
    assert v['ticks']==3 and v['bomb_legal_ticks']==2 and v['bomb_top1_mean_gap'] is None
    assert v['mean_bomb_probability_when_legal']==pytest.approx(.35)


def test_concurrent_progress_writes_keep_a_complete_json(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(I,'BASE',tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lambda i:I.progress({'shard':i,'complete':True}),range(64)))
    assert json.loads((tmp_path/'D12_PROGRESS.json').read_text())['complete'] is True


def test_rollback_requires_all_three_metrics_and_both_control_coverages():
    from scripts.bomb_threat_iter2_analysis import comparison
    row=lambda w,k,b:{'metrics':dict(win_minus_loss=w,causal_kills=k,bombs_per_game=b)}
    controls={'0':[row(12,25,28)]*3,'3':[row(12,25,28)]*3}
    assert comparison([row(11,24,27)]*3,controls)['status']=='RESTORE_ORIGINAL_RECIPE'
    assert comparison([row(13,24,27)]*3,controls)['status']=='CONTINUE'
    assert comparison([row(11,24,27)]*2,controls)['status']=='PENDING_MATCHED_DEV_DATA'
    assert comparison([row(11,24,27)]*3,{'0':controls['0'],'3':[]})['status']=='PENDING_MATCHED_DEV_DATA'
    assert comparison([row(12,24,27)]*3,controls)['status']=='CONTINUE'


def test_analysis_excludes_invalid_and_requires_u640(tmp_path,monkeypatch):
    from scripts import bomb_threat_iter2_analysis as A
    monkeypatch.setattr(A,'ROOT',tmp_path);monkeypatch.setattr(A,'BASE',tmp_path/'runs/bomb_threat_v1')
    base=A.BASE
    for update in (440,480,520,600,680):
        cp=base/'formal/bt_slot04_c000013'/f'u{update:06d}';cp.mkdir(parents=True)
        (cp/'COMMITTED.json').write_text(json.dumps(dict(passed=True,update=update,policy_sha256='policy',metrics={'unix':update})))
        marker=base/'eval'/f'point{update}';marker.mkdir(parents=True)
        (marker/'NATURAL_COMPLETE.json').write_text(json.dumps(dict(split='dev',seed=2026122900,measurement_version=2,
            checkpoint=str(cp.relative_to(tmp_path)),policy_sha256='policy',natural_js_full2400={'greedy':dict(win=10,loss=1,causal_kills=20,bombs_per_game=25,zero_bomb_games=5)})))
    (base/'protocols').mkdir()
    (base/'protocols/bt_slot04_c000013.json').write_text(json.dumps(dict(slot=4,iter2_arm='A1')))
    result=A.refresh()
    assert result['rollback_decisions']['4']['status']=='PENDING_U640_DEV'
    (base/'eval/point440/EXCLUDED_INVALID.json').write_text('{}')
    result=A.refresh()
    assert sum(b['points'] for b in result['buckets'])==4


def test_rollback_protocol_has_new_init_and_cannot_recursively_rollback(tmp_path,monkeypatch):
    from scripts import bomb_threat_iter2_schedule as S
    from scripts.bomb_threat_iter2_fallback_train import formal_configuration
    monkeypatch.setattr(S,'ROOT',tmp_path);base=tmp_path/'runs/bomb_threat_v1';monkeypatch.setattr(S,'BASE',base)
    (base/'iter2').mkdir(parents=True)
    (tmp_path/'scripts').mkdir();(tmp_path/'scripts/bomb_threat_iter2_fallback_train.py').write_text('frozen worker')
    config=dict(common={},policy_source={'checkpoint':'old','policy_sha256':'old'},source_sha256={},
        initialization_decision='decision',spawn_manifest='spawns',A1_weights=[.05,.35,.3,.15,.1,.05],fixed_actor_source={})
    (base/'iter2/TRAINING_PROTOCOL.json').write_text(json.dumps(config))
    source=dict(checkpoint='current/u000640',policy_sha256='current',state_sha256='state',update=640)
    recipe=S.rollback_recipe(dict(iter2_arm='A3',lr=7.5e-5),source)
    assert formal_configuration(recipe)['shaping_multiplier']==1.
    assert formal_configuration(recipe)['entropy_anneal'] is False
    path=S.make_protocol(4,14,None,recipe)
    p=json.loads(path.read_text())
    assert p['policy_source']['checkpoint']==source['checkpoint'] and p['continuation_source'] is None
    assert p['chunk_start_update']==0 and p['chunk_updates']==320
    assert p['worker_module']=='scripts.bomb_threat_iter2_fallback_train'
    assert p['reward_version']=='bomb-threat-v1'
    assert S.apply_rollback(4,dict(run=path.stem,iter2_arm='A3'),source,15) is None
