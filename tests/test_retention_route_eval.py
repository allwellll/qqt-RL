import pytest
from scripts.retention_route_eval import merge_shards,TASKS


def shards():
    return [dict(passed=True,model_json_sha256='same',seeds=[100,101],results={task:dict(games=2,episodes=[dict(seed=100),dict(seed=101)])}) for task in TASKS]


def test_parallel_tasks_preserve_order_seeds_and_complete_denominators():
    result=merge_shards(shards())
    assert list(result['results'])==TASKS
    assert result['seeds']==[100,101]
    assert all(v['games']==2 for v in result['results'].values())


def test_merge_rejects_identity_or_seed_or_incomplete_task_mismatch():
    for field,value in [('model_json_sha256','wrong'),('seeds',[9,10])]:
        data=shards();data[2][field]=value
        with pytest.raises(ValueError):merge_shards(data)
    with pytest.raises(ValueError):merge_shards(shards()[:-1])


def test_episode_shards_recombine_counts_in_global_game_order():
    from scripts.retention_route_eval import merge_episodes
    def part(begin):
        es=[dict(game=i,seed=100+i,team=i%2,delivery=int(i%2==0),pickup=1,oracle_opportunities=2,first_oracle_match=1,stage_funnel={'C_RETURN':dict(progress_ticks=i)}) for i in range(begin,begin+2)]
        return dict(passed=True,model_json_sha256='same',seeds=[100+i for i in range(begin,begin+2)],results={'real_drop':dict(games=2,acquisition_scope='native',episodes=es)})
    result=merge_episodes('real_drop',[part(0),part(2)])
    r=result['results']['real_drop']
    assert r['games']==4 and r['delivery_episodes']==2 and r['pickup_episodes']==4
    assert [e['game'] for e in r['episodes']]==[0,1,2,3]
    assert r['stage_funnel']['C_RETURN']['progress_ticks']==6
    with pytest.raises(ValueError):merge_episodes('real_drop',[part(0),part(0)])
