"""R1 acceptance: exact coefficients, history, seat and atomic scheduling."""
import numpy as np
import jax
import jax.numpy as j


def test_bank_symmetric_and_global_quarter():
    from scripts.bomb_threat_r1_collector import bank_selection
    n=100000;seats=j.arange(n)%2;keys=jax.random.split(jax.random.PRNGKey(2026103033),n)
    use=np.asarray(jax.jit(bank_selection)(keys,seats))
    counts=[int(use[np.asarray(seats)==p].sum()) for p in (0,1)]
    assert all(12000<c<13000 for c in counts) and abs(counts[0]-counts[1])<700
    assert .24<use.mean()<.26


def test_weighted_distance_walls_bricks_and_near_cap():
    from scripts.bomb_threat_r1_reward import weighted_distance,potential
    wall=j.ones((1,13,15),bool).at[0,6,1:6].set(False)
    brick=j.zeros_like(wall).at[0,6,3].set(True)
    d=weighted_distance(wall,brick,j.zeros_like(wall),j.array([[6,5]]),j.array([[6,1]]))
    assert int(d[0])==5
    np.testing.assert_allclose(potential(j.array([0,1,2,3,4])),[-.006]*4+[-.008])


def event(n=1):
    return dict(active=j.ones(n,bool),bad=j.zeros(n,bool),terminal=j.zeros(n,bool),settled=j.zeros(n,bool),
        placed=j.ones(n,bool),safe=j.ones(n,bool),threat=j.ones(n,bool),enemy_cell=j.zeros(n,j.int32),
        old_ids=j.full((n,195),-1,j.int32),owned_removed=j.zeros((n,13,15),bool),improved=j.zeros(n,bool),
        multibatch=j.zeros(n,bool),batch_id=j.zeros(n,j.int32),pickup_types=j.zeros((n,3),bool),
        phi_before=j.full(n,-.02),phi_after=j.full(n,-.018),phase_before=j.ones(n,bool),phase_after=j.ones(n,bool),
        danger_after=j.zeros(n,bool))


def test_gamma_pending_cap_and_void_no_double_pay():
    from scripts.bomb_threat_r1_reward import state,settle
    s=state(1);e=event();n,r,m=settle(e,s)
    np.testing.assert_allclose(r,0)
    np.testing.assert_allclose(n['pending'],.02+.995*(-.018)-(-.02),rtol=1e-5)
    e=dict(e,settled=j.ones(1,bool),threat=j.zeros(1,bool),phase_before=j.zeros(1,bool))
    nn,r,m=settle(e,n);assert float(r[0])>0 and float(nn['pending'][0])==0
    _,r2,_=settle(e,nn);np.testing.assert_allclose(r2,0)
    for key in ('bad','terminal'):
        _,r,_=settle(dict(e,**{key:j.ones(1,bool)}),n);np.testing.assert_allclose(r,0)
    _,r,_=settle(dict(e,settled=j.ones(1,bool)),dict(n,pending=j.array([9.]),gross=j.array([.29])))
    np.testing.assert_allclose(r,.01,atol=1e-7)


def test_chain_dedup_by_batch_not_cell_and_max_three():
    from scripts.bomb_threat_r1_reward import state,settle
    e=dict(event(),threat=j.zeros(1,bool),phase_before=j.zeros(1,bool))
    ids=j.full((1,195),-1,j.int32).at[0,:5].set(j.array([10,10,20,30,40]))
    n,_,m=settle(dict(e,old_ids=ids),state(1));assert int(n['chain_count'][0])==3
    np.testing.assert_allclose(n['pending'],.03)
    nn,_,_=settle(dict(e,old_ids=ids),n);np.testing.assert_array_equal(nn['pending'],n['pending'])


def test_multibatch_items_dedup_and_max_six_three():
    from scripts.bomb_threat_r1_reward import state,settle
    s=state(1);e=dict(event(),threat=j.zeros(1,bool),phase_before=j.zeros(1,bool),pickup_types=j.ones((1,3),bool),multibatch=j.ones(1,bool))
    for k in range(9):
        s,_,_=settle(dict(e,batch_id=j.full(1,k)),s)
        # Same batch cannot pay again.
        prev=s['pending'];s,_,_=settle(dict(e,batch_id=j.full(1,k)),s);np.testing.assert_array_equal(prev,s['pending'])
    assert int(s['multi_count'][0])==6 and int(s['item_seen'].sum())==3
    np.testing.assert_allclose(s['pending'],.015*6+.03,rtol=1e-5)


def test_danger_holds_phi_phase_exit_no_bonus():
    from scripts.bomb_threat_r1_reward import state,settle
    e=dict(event(),threat=j.zeros(1,bool),danger_after=j.ones(1,bool),phi_after=j.array([-.2]))
    n,_,m=settle(e,state(1));np.testing.assert_allclose(m['approach'],.995*(-.02)+.02,atol=1e-8)
    n,_,m=settle(dict(e,phase_after=j.zeros(1,bool)),state(1));np.testing.assert_allclose(m['approach'],0)


def test_native_p1_encoder_cage_inventory_and_carry():
    from scripts import combat_item_engine as T,carrier_mechanism_engine as E
    T.configure();s,_=E.initial(jax.random.split(jax.random.PRNGKey(81),1),0)
    s=s._replace(bun_stored=j.array([[[2,1],[3,4]]],s.bun_stored.dtype),bun_carried=j.array([[-1,0]],s.bun_carried.dtype))
    o=E.train.both_perspectives(s);g=E.train.both_states(s)
    np.testing.assert_array_equal(o[0,14],o[1,15]);np.testing.assert_array_equal(o[0,15],o[1,14])
    np.testing.assert_array_equal(g[0,17:21],[2,1,3,4]);np.testing.assert_array_equal(g[1,17:21],[4,3,1,2])
    assert float(g[1,13])==1 and float(g[1,14])==1 and float(g[0,13])==0


def test_hack_gate_requires_two_empty_points_but_not_training_pause():
    from scripts.bomb_threat_r1_schedule import monitor_gate
    base=dict(empty_bomb_fraction=.2,kills=20,self_harm=0)
    assert monitor_gate([base,dict(base,empty_bomb_fraction=.4)])['status']=='MISSING'
    assert monitor_gate([base,dict(base,empty_bomb_fraction=.4),dict(base,empty_bomb_fraction=.4)])['status']=='HALVE'
    assert monitor_gate([base,dict(base,empty_bomb_fraction=.4,kills=21),dict(base,empty_bomb_fraction=.4,kills=21)])['status']=='OBSERVE'
    assert monitor_gate([base,base,dict(base,self_harm=3)])['status']=='HALVE'


def test_no_r1_takeover_until_real_reset_audit_and_boundary(tmp_path,monkeypatch):
    from scripts import bomb_threat_r1_schedule as S
    monkeypatch.setattr(S,'OUT',tmp_path/'r1');monkeypatch.setattr(S,'BASE',tmp_path/'base')
    assert S.apply_switch(0,dict(iter2_arm='A3SR'),{},1) is None
    assert S.apply_switch(5,dict(iter2_arm='A3SR'),{},1) is None
    assert not S.manages(0,dict(iter2_arm='A3SR'))
    assert S.manages(0,dict(r1_applied=True))


def test_new_worker_parity_precedes_migration_and_old_path_untouched():
    from scripts.bomb_threat_r1_train import worker_source
    s=worker_source();compile(s,'r1_generated','exec')
    assert s.index("'donor next-step mismatch'")<s.index("meta['bomb_threat']=shard(credit_state(n),1)")
    assert "new_reward_ledger_reset=protocol['r1_attack']" in s
    assert 'fixed_actor' in s and 'Adam_exact=True' in s


def test_p0_p1_same_state_identity_mirror_semantics():
    from scripts import combat_item_engine as T,carrier_mechanism_engine as E
    from scripts.bomb_threat_seat_encoding import exchange_encoder_identities
    # Encoder-only identity fixture; explicitly not a training Bank transform.
    T.configure();s,_=E.initial(jax.random.split(jax.random.PRNGKey(89),2),0)
    s=s._replace(bun_stored=j.array([[[2,1],[3,4]],[[1,3],[2,4]]],s.bun_stored.dtype),
        bun_carried=j.array([[-1,0],[1,-1]],s.bun_carried.dtype))
    oldo=E.train.both_perspectives(s);oldg=E.train.both_states(s)
    reflected=exchange_encoder_identities(s);original=E.env._base_mask
    try:
        E.env._base_mask=lambda p:original(1-p)
        newo=E.train.both_perspectives(reflected);newg=E.train.both_states(reflected)
        np.testing.assert_array_equal(oldo,newo[j.array([2,3,0,1])])
        np.testing.assert_array_equal(oldg,newg[j.array([2,3,0,1])])
    finally:E.env._base_mask=original


def test_effective_multi_timing_rejects_chain_collapsed_or_less_than_eight_ticks():
    from scripts.bomb_threat_r1_reward import multi_time_cover
    from scripts import combat_item_engine as T,carrier_mechanism_engine as E
    T.configure();s,_=E.initial(jax.random.split(jax.random.PRNGKey(90),1),0)
    zeros=j.zeros((1,13,15),bool)
    fuse=j.zeros((1,13,15),j.int32).at[0,6,4].set(10).at[0,6,8].set(25)
    own=j.full((1,13,15),-1,j.int32).at[0,6,4].set(0).at[0,6,8].set(0)
    core=s.core._replace(wall=zeros,brick=zeros,pushable=zeros,owner=own,fuse=fuse,bomb_blast=j.ones_like(fuse)*2,
        pos=j.array([[[2.5,2.5],[6.5,6.5]]]))
    s=s._replace(core=core);fn=jax.jit(lambda x:multi_time_cover(x,j.array([0])))
    assert bool(fn(s)[0])
    assert not bool(fn(s._replace(core=core._replace(fuse=fuse.at[0,6,8].set(17))))[0])
    assert not bool(fn(s._replace(core=core._replace(bomb_blast=j.ones_like(fuse)*4)))[0])


def test_p1_bank_keeps_full_auxiliary_ledger(tmp_path):
    import pickle
    from scripts import bomb_threat_r1_collector as M,combat_item_engine as T,carrier_mechanism_engine as E
    from scripts.population_run_state import sha
    T.configure();st,ax=E.initial(jax.random.split(jax.random.PRNGKey(92),1),0)
    ax=dict(ax,tactics=T.events.state(1))
    # Sentinel in genuine auxiliary accounting; no state manipulation enters
    # a Bank here. This is a schema preservation unit fixture only.
    ax['tactics']=dict(ax['tactics'],banana_frontier=j.array([[1.234,2.345]],j.float32))
    p=tmp_path/'bank.pkl'
    with p.open('wb') as f:pickle.dump([dict(states=st,aux=ax)],f)
    bs,ba=M.banks(st,{k:v for k,v in ax.items() if k!='tactics'},dict(p1_file=str(p),p1_sha256=sha(p)))
    np.testing.assert_array_equal(ba[1]['tactics']['banana_frontier'],ax['tactics']['banana_frontier'])


def test_halving_survives_episode_reset_and_full_restore():
    from scripts import bomb_threat_r1_collector as M
    # This generated collector uses the exact per-env persisted mode on reset;
    # coefficient changes cannot disappear after the first new episode.
    assert "r1_scale=ax['combat_mix']['bomb_threat']['r1_scale']" in M._source
    from scripts.bomb_threat_r1_train import worker_source
    s=worker_source()
    assert "if protocol['r1_attack'] and (index-1)%40==0:" in s
    assert "ledger['r1_scale']=j.full_like(ledger['r1_scale'],factor)" in s


def test_new_safety_qualification_binds_current_rules_without_mutating_legacy():
    from scripts.bomb_threat_r1_reward import native_safety,native_safe_exits
    from scripts import combat_item_engine as T,carrier_mechanism_engine as E
    from jax_bomb import bun_safety as old
    oldenv,oldbase=old.env,old.base;T.configure();local=native_safety()
    assert local is not old and local.env is E.env and local.base is E.env.base
    assert old.env is oldenv and old.base is oldbase
    st,_=E.initial(jax.random.split(jax.random.PRNGKey(94),1),0)
    flags,cells=jax.jit(native_safe_exits)(st,j.array([1]))
    assert flags.shape==(1,5) and cells.shape==(1,5,2)


def test_migration_pool_kind_dtype_matches_reset_metadata():
    from scripts.bomb_threat_r1_train import worker_source
    assert "kind=j.asarray(kinds,j.int32)[None]" in worker_source()
