from scripts import r3_ownside_recheck_1620_20261031 as m


def test_jobs_preserve_modes_seats_and_budgets():
    p = dict(holdout=dict(seed=2026317200, games=256), holdout_modes=['greedy', 'categorical'],
             models={'baseline': {}, 'r3_a_u3760': {}, 'r3_b_u3600': {}},
             conditions={f'hunter_{t}': dict(games=256) for t in ('hard', 'normal', 'easy')})
    js = m.jobs(p)
    assert len(js)*8 == 2816
    assert len({(j['kind'],j['condition'],j['label'],j['mode'],j['start']) for j in js}) == len(js)
    for mode in p['holdout_modes']:
        assert sum(8 for j in js if j['kind']=='holdout' and j['mode']==mode) == 256
    assert {j['mode'] for j in js if j['kind']=='fresh'} == {'greedy'}


def test_selection_keeps_current_on_tie_and_blocks_weak_tier():
    def metric(values, p0=100, selfd=0):
        return {t:dict(P1=dict(win_minus_loss=v,self_deaths=selfd),
                       P0=dict(win_minus_loss=p0,self_deaths=0))
                for t,v in zip(('hunter_hard','hunter_normal','hunter_easy'),values)}
    metrics={'baseline':metric([91,110,119]),'a':metric([91,110,119]),'b':metric([85,125,125])}
    r=m.selection(metrics,['a','b'])
    assert r['winner']=='baseline' and not r['comparison']['b']['eligible']
    metrics['a']=metric([92,110,119]); assert m.selection(metrics,['a','b'])['winner']=='a'
    metrics['a']=metric([92,110,119],selfd=1); assert m.selection(metrics,['a','b'])['winner']=='baseline'


def test_existing_baseline_shard_is_reused_not_recomputed():
    j=dict(label='baseline',kind='fresh',condition='hunter_hard',start=0)
    f,model=m.reuse(j)
    assert str(f).endswith('v2_ownside_selection_20261031/shards/hunter_hard/r3_a_u3040/shard_000.json')
    assert model['policy_sha256'].startswith('80ae9005')
    assert m.reuse(dict(label='r3_a_u3760',kind='fresh')) is None


def test_paired_interval_retains_fixed_seat_and_exact_zero():
    import numpy as np
    from scripts.r3_ownside_recheck_analysis_20261031 import paired_interval
    rows=[dict(seat=s,score_difference=s) for s in (0,1) for _ in range(128)]
    a,_=paired_interval(rows,0,np.random.default_rng(1),100)
    b,_=paired_interval(rows,1,np.random.default_rng(1),100)
    assert a['delta_win_minus_loss']==0 and a['paired_seed_bootstrap_95']==[0,0]
    assert b['delta_win_minus_loss']==128 and b['paired_seed_bootstrap_95']==[128,128]
