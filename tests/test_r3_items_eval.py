from scripts import r3_items_eval_owner as E


def metrics():
    value=dict(games=64,win_minus_loss=10,unilateral_zero_bomb=0,self_deaths=0,
        active_deliveries=8,farm_fraction=.1,zero_bomb=1,enemy_stock_pickups=4,
        carrier_kills=2,wins=10,causal_kills=20,empty_fraction=.1,
        resolved_bombs=100,glue_self_contacts=0)
    import copy
    return dict(metrics=dict(holdout={m:{s:copy.deepcopy(value) for s in ('P0','P1')} for m in ('greedy','categorical')}))


def test_matched_gate_and_actual_empty_denominator():
    a,b=metrics(),metrics();assert E.gates(a,b)['status']=='PASS'
    a['metrics']['holdout']['greedy']['P1']['empty_fraction']=.31
    g=E.gates(a,b);assert g['status']=='BLOCK'
    assert g['r3']['greedy_P1']['empty_without_kill_gain']
    a['metrics']['holdout']['greedy']['P1']['causal_kills']=21
    assert not E.gates(a,b)['r3']['greedy_P1']['empty_without_kill_gain']


def test_glue_self_contact_and_safety_threshold():
    a,b=metrics(),metrics();a['metrics']['holdout']['greedy']['P0']['glue_self_contacts']=65
    assert E.gates(a,b)['r3']['greedy_P0']['self_glue']
    a['metrics']['holdout']['categorical']['P1']['self_deaths']=3
    assert E.gates(a,b)['r3']['categorical_P1']['selfharm']


def test_fixed_matrix_both_modes_seat_counts_and_seeds():
    p=E.freeze();j=E.specifications(dict(kind='holdout'),p)
    assert len(j)==32 and sum(8 for _ in j)==256
    for mode in ('greedy','categorical'):
        selected=[v for v in j if v['mode']==mode]
        seeds=[v['seed']+k for v in selected for k in range(v['start'],v['start']+8)]
        assert len(set(seeds))==128
    j=E.specifications(dict(kind='fresh'),p)
    assert len(j)==96 and {x['mode'] for x in j}=={'greedy'}
