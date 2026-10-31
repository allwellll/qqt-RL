import pytest
from scripts.retention_results import paired_interval,metrics


def test_training_seed_bootstrap_does_not_treat_games_as_independent_repeats():
    lo,hi=paired_interval([-.1,0,.1])
    assert lo<0<hi
    assert paired_interval([.05,.05,.05])==[.05,.05] or all(abs(v-.05)<1e-12 for v in paired_interval([.05,.05,.05]))


def test_natural_primary_denominator_stays_separate_from_artificial_carriers():
    s=dict(route=dict(natural_full2400=dict(games=32,delivery_episodes=4,pickup_episodes=8,self_death=0),carry_near=dict(games=32,delivery_episodes=32)),
        tactical_python=dict(safe_win=8,self_episodes=2,bombs_per_game=4),
        hunter_js={tier:dict(games=64,safe_win=4,self_episodes=0,bombs_per_game=5) for tier in ['hard','normal','easy']})
    m=metrics(s)
    assert m['natural']==.125 and m['pickup']==.25 and m['tactical']==.125
    assert 'carry_near' not in m
