"""Fresh gate sample-size and missing-data regressions, without playing games."""
from scripts.bomb_threat_fresh256 import gate,jobs


def sample(**kw):
    return dict(games=256,win_minus_loss=40,zero_bomb_games=28,bombs_per_game=20,
        draw_fraction=.65,causal_kills=120,deaths=60,self_deaths=0,cage_kill_fraction=.03,**kw)


def test_fresh_gate_scaled_counts_and_no_categorical_gate():
    c=sample();p=sample();p.update(win_minus_loss=8,deaths=40,cage_kill_fraction=.03)
    refs={r:sample() for r in ('transport1536','selfplay3072')}
    for r in refs.values():r['win_minus_loss']=0
    assert gate(c,p,refs,.017)['status']=='PASS'
    c['zero_bomb_games']=29
    assert gate(c,p,refs,.017)['failures']==['zero_bomb_at_most28']
    c['zero_bomb_games']=28;c['win_minus_loss']=39
    assert 'win_minus_loss_parent_plus32' in gate(c,p,refs,.017)['failures']


def test_missing_without_known_failure_is_incomplete():
    assert gate({}, {}, {}, None)['status']=='INCOMPLETE'
    c=sample();c['cage_kill_fraction']=None
    p=sample();p['win_minus_loss']=8
    assert gate(c,p,{},.01)['status']=='INCOMPLETE'


def test_missing_comparator_does_not_hide_candidate_failure():
    c=sample();c['zero_bomb_games']=40
    v=gate(c,{}, {},None)
    assert v['status']=='FAIL'
    assert 'zero_bomb_at_most28' in v['failures']
    assert v['missing']


def test_actual_hard_known_failures_and_missing_reference_are_separate():
    c=sample();c.update(win_minus_loss=35,zero_bomb_games=40,deaths=55)
    p=sample();p.update(win_minus_loss=8,deaths=39)
    v=gate(c,p,{'transport1536':dict(games=256,win_minus_loss=0)},.017)
    assert v['status']=='FAIL'
    assert v['failures']==['win_minus_loss_parent_plus32','zero_bomb_at_most28']
    assert v['missing']==['selfplay3072_MATCHED_256_REQUIRED']


def test_partial_sample_cannot_prove_frozen_256_gate_failure():
    c=sample();c.update(games=8,zero_bomb_games=40,win_minus_loss=-8)
    assert gate(c,{}, {},None)['status']=='INCOMPLETE'


def test_known_failure_overrides_incomplete_matrix_without_marking_collection_complete():
    from scripts.bomb_threat_fresh256 import verdict_status
    assert verdict_status([{'status':'FAIL'},{'status':'INCOMPLETE'}])=='FAIL'
    assert verdict_status([{'status':'PASS'},{'status':'INCOMPLETE'}])=='INCOMPLETE'


def test_publish_known_fail_is_not_collection_complete(tmp_path,monkeypatch):
    import json
    from scripts import bomb_threat_fresh256 as F,bomb_threat_eval as E
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(F,'OUT',tmp_path/'fresh')
    F.OUT.mkdir();(F.OUT/'PROTOCOL.json').write_text('{}');(tmp_path/'reports').mkdir()
    for role in ('candidate','parent'):
        for start in range(0,256,8):
            p=F.job_path(dict(tier='hard',role=role,mode='greedy',start=start));p.parent.mkdir(parents=True,exist_ok=True)
            p.with_name(p.name+'.ACCEPTED.json').write_text(json.dumps(dict(files={})))
    monkeypatch.setattr(F,'validate_shard',lambda p,j,protocol:[dict(seed=i,role=j['role']) for i in range(j['start'],j['start']+8)])
    def summarize(rows):
        m=sample();m.update(win=42,draw=207,loss=7,structural_zero_bomb_games=0,draw_kill_difference=0)
        m.update(win_minus_loss=35 if rows[0]['role']=='candidate' else 8,zero_bomb_games=40 if rows[0]['role']=='candidate' else 28)
        return m
    monkeypatch.setattr(E,'summarize',summarize)
    value=F.publish(dict(native_shaping_share=.017))
    assert value['status']=='FAIL' and value['collection_status']=='INCOMPLETE'
    assert value['completed_games']==512
    assert not (F.OUT/'COMPLETE.json').exists()
    assert (tmp_path/'reports/bomb_threat_v1_fresh256_20261029.md').read_text().splitlines()[0]=='B 最终判定：FAIL（hard: W-L差27<32，单方零泡40>28）'


def test_matrix_unique_seeds_roles_modes_and_complete_budget():
    js=jobs();assert len(js)==480
    assert len({tuple(j.items()) for j in js})==480
    assert {j['mode'] for j in js[:384]}=={'greedy'}
    assert all(j['role']=='candidate' for j in js[384:])
    for t in ('hard','normal','easy'):
        for r in ('candidate','parent','transport1536','selfplay3072'):
            assert sum(8 for j in js if j['tier']==t and j['role']==r and j['mode']=='greedy')==256
