from scripts import bomb_threat_a3sr_promotion as P
from scripts.population_run_state import atomic_json


def test_fork_preserves_recipe_and_requires_rekey(tmp_path,monkeypatch):
    monkeypatch.setattr(P,'ROOT',tmp_path);monkeypatch.setattr(P,'BASE',tmp_path/'runs/bomb_threat_v1')
    monkeypatch.setattr(P,'AUTH',P.BASE/'auth.json')
    source=dict(checkpoint='runs/bomb_threat_v1/formal/donor/u001320',update=1320,state_sha256='state',policy_sha256='policy')
    recipe=dict(iter2_arm='A3SR',lr=7.5e-5,entropy_coefficient=.005,learner_seat_fraction=.5)
    atomic_json(P.BASE/'protocols/donor.json',dict(arms={'donor':recipe},source_sha256={},round14_collector_seed=2026112905))
    path=P.make_protocol(7,30,source,True);d=P.read(path)
    assert d['arms'][path.stem]==recipe
    assert d['continuation_source']==source and d['chunk_updates']==320
    assert d['round14_rekey'] is True and d['round14_collector_seed']==2026112905
    assert d['sources']=={'2026112907':source} and d['pbt_lineage']['critic_warmup_updates']==0
    p2=P.make_protocol(5,30,source,False);assert P.read(p2)['round14_rekey'] is False


def test_unrelated_lane_and_unfinished_chunk_untouched(tmp_path,monkeypatch):
    monkeypatch.setattr(P,'AUTH',tmp_path/'auth.json')
    atomic_json(P.AUTH,dict(status='AUTHORIZED',boundary_runs={'7':'old'}))
    assert P.apply_switch(1,dict(run='other'),{},1) is None
    assert P.apply_switch(6,dict(run='other'),{},1) is None
    assert P.apply_switch(7,dict(run='unrelated'),{},1) is None


def test_rolling_holdout_only_large_changes_block():
    def point(w=12,z=2,s=0,d=8):
        return dict(by_seat={seat:dict(win_minus_loss=w,zero_bomb_games=z,self_deaths=s,delivery=d) for seat in ('P0','P1')})
    assert P.rolling_gate([point()]*7)['status']=='MISSING'
    assert P.rolling_gate([point()]*4+[point(w=6.1,z=4.9,s=2.9,d=4.1)]*4)['status']=='PASS'
    boundary=P.rolling_gate([point()]*4+[point(w=6,z=5,s=3,d=4)]*4)
    assert boundary['status']=='BLOCK' and len(boundary['failures'])==8
    r=P.rolling_gate([point()]*4+[point(w=5,z=6,s=4,d=3)]*4)
    assert r['status']=='BLOCK' and len(r['failures'])==8
    assert P.rolling_gate([point(d=0)]*4+[point(d=0)]*4)['status']=='PASS'


def test_eight_holdout_points_stay_eligible_between_cadence_boundaries():
    for current in range(1400,1440):
        points=[u for u in range(0,current+1,40) if u>=P.holdout_update_floor(current)]
        assert points==list(range(1120,1401,40))
    assert P.holdout_update_floor(1440)==1160


def test_four_slot_acceptance_requires_real_updates_and_independent_rng():
    import pytest
    donor=dict(checkpoint='donor',state_sha256='state',policy_sha256='policy')
    records=[]
    for slot in P.TARGETS:
        records.append(dict(slot=slot,run=f'slot{slot}',finite=True,update_l2={'actor':1,'critic':2},
            source=donor,smoke={'passed':True},restore={'passed':True},first_formal_update=1323,
            source_handoff=dict(donor_next_step_exact_before_rekey=True,Actor_Critic_Adam_env_aux_ring_exact=True,
                RNG_rekeyed=True,critic_warmup_updates=0,formal_updates_discarded=0,
                source_rng_digest='donor',target_rng_digest=f'rng{slot}')))
    assert P.final_acceptance(records[:-1],donor)['status']=='MISSING'
    assert P.final_acceptance(records,donor)['status']=='PASS'
    records[-1]['source_handoff']['target_rng_digest']='rng0'
    with pytest.raises(AssertionError,match='distinct RNG'):
        P.final_acceptance(records,donor)
