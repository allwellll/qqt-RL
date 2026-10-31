"""A3S learner-seat binding; old P0 collector must remain reproducible."""
import numpy as np
import jax
import jax.numpy as j


def test_fixed_half_allocation_reproducible_and_zero_default():
    from scripts import bomb_threat_seat5050_collector as S
    a=S.seat_allocation(2048,2026112907,.5)
    assert np.count_nonzero(a==1)==1024 and np.count_nonzero(a==0)==1024
    np.testing.assert_array_equal(a,S.seat_allocation(2048,2026112907,.5))
    assert not np.array_equal(a,S.seat_allocation(2048,2026112904,.5))
    np.testing.assert_array_equal(S.seat_allocation(2048,1,0),np.zeros(2048,np.int32))
    from scripts import bomb_threat_collector_diagnostics_v1 as old
    assert S.collector(1,4,7,.1,0.,[],[1,0,0,0,0,0],seat_fraction=0) is not None
    assert S.default_collector is old.collector


def test_observation_mask_action_reward_and_gae_follow_same_seat():
    from scripts import bomb_threat_seat5050_collector as S
    seats=j.array([0,1,1,0]);both=j.arange(8*3).reshape(8,3)
    np.testing.assert_array_equal(S.select_perspective(both,seats),both[j.array([0,5,6,3])])
    native=j.array([[11.,101.],[12.,102.],[13.,103.],[14.,104.]])
    reward=S.select_player(native,seats)
    np.testing.assert_array_equal(reward,[11,102,103,14])
    own=j.array([[0,1],[1,2],[2,0],[3,1]]);enemy=own+5
    joint=S.joint_actions(own,enemy,seats)
    np.testing.assert_array_equal(S.select_player(joint,seats),own)
    np.testing.assert_array_equal(S.select_player(joint,1-seats),enemy)
    from jax_bomb.jax_train import compute_gae
    v=j.zeros((2,4));r=j.stack([reward,reward*2]);done=j.array([[0,0,0,0],[1,1,1,1]],bool)
    adv=compute_gae(r,v,j.ones_like(v)*99,done,.995,.95)
    # Terminal second row has no bootstrap, and every column retains learner reward.
    np.testing.assert_array_equal(adv[1],reward*2)
    assert float(adv[0,1])>float(adv[0,0])


def test_bank_assignment_is_p0_only_and_preserves_aggregate_probability():
    from scripts import bomb_threat_seat5050_collector as S
    seats=S.seat_allocation(2048,2026112907,.5)
    keys=jax.random.split(jax.random.PRNGKey(123),2048)
    use=S.bank_selection(keys,j.asarray(seats),.25)
    assert not np.asarray(use)[seats==1].any()
    assert abs(float(use.mean())-.25)<.03
    assert S.conditional_bank_probability(.25,.5)==.5
    assert S.conditional_bank_probability(.25,0)==.25


def test_seat_teacher_uses_same_custody_and_window_without_fabricating():
    from scripts import bomb_threat_seat5050_collector as S
    from types import SimpleNamespace
    st=SimpleNamespace(bun_carried=j.array([[1,-1],[-1,0],[-1,-1],[-1,-1]]),
        alive=j.array([[1,1],[1,1],[1,0],[1,1]],bool),trapped=j.zeros((4,2),j.int32))
    ax=dict(terminal=j.zeros(4,bool),carry_items=dict(ledger=dict(custody=j.array([[1,0],[0,1],[0,0],[0,0]],bool))))
    seats=j.array([0,1,0,1]);np.testing.assert_array_equal(S.teacher_eligible(st,ax,seats),[1,1,1,0])
    ax['carry_items']['ledger']['custody']=j.zeros((4,2),bool)
    np.testing.assert_array_equal(S.teacher_eligible(st,ax,seats),[0,0,1,0])


def test_projected_threat_payment_hits_selected_seat_only(monkeypatch):
    from scripts import bomb_threat_seat5050_collector as S,bomb_threat_seat5050_reward as R
    def native(st,ax,actions,keys,arm):
        return st,ax,j.array([[10.,20.],[30.,40.]]),j.array([False,True]),{},j.zeros((2,2,3))
    def extra(st,after,info,ax,meta,seats):return meta['bomb_threat'],j.array([.1,.2]),{},meta['birth']
    monkeypatch.setattr(S.T,'transition',native);monkeypatch.setattr(R,'transition',extra)
    meta=dict(learner_seat=j.array([0,1]),bomb_threat={'pending':j.array([1.,2.])},birth=j.zeros((2,13,15)))
    _,ax,r,done,_,parts,_=S.transition(None,dict(combat_mix=meta,custody=j.array([[1,0],[0,1]])),None,None,0.)
    np.testing.assert_allclose(r,[[10.1,20.],[30.,40.2]])
    np.testing.assert_array_equal(ax['custody'],[[1,0],[0,1]])
    np.testing.assert_allclose(S.select_player(parts[:,:,-1],meta['learner_seat']),[10.1,40.2])
    np.testing.assert_array_equal(done,[False,True])


def test_threat_chain_uses_physical_owner_and_bad_learner(monkeypatch):
    from scripts import bomb_threat_seat5050_reward as R
    from types import SimpleNamespace
    pos=j.array([[[4.5,4.5],[8.5,8.5]],[[4.5,4.5],[8.5,8.5]]])
    fuse=j.zeros((2,13,15),j.int32).at[0,4,5].set(20).at[1,8,9].set(20)
    owner=j.full((2,13,15),-1,j.int32).at[0,4,5].set(0).at[1,8,9].set(1)
    before=SimpleNamespace(t=j.zeros(2,j.int32),pos=pos,fuse=fuse,wall=j.zeros_like(fuse,bool),brick=j.zeros_like(fuse,bool),
        core=SimpleNamespace(owner=owner,bomb_blast=j.ones_like(fuse)*3,pushable=j.zeros_like(fuse,bool)))
    after=SimpleNamespace(pos=pos,fuse=fuse.at[0,4,4].set(50).at[1,8,8].set(50),
        trapped=j.array([[0,1],[0,0]]),alive=j.ones((2,2),bool),
        core=SimpleNamespace(owner=owner,blast_linger=j.zeros_like(fuse)))
    info=dict(bomb_placed=j.array([[1,0],[0,1]],bool),tactical_bomb_newly_threatens_enemy=j.ones((2,2),bool),
        survivable_bomb_placed=j.ones((2,2),bool),death=j.array([[0,1],[0,0]],bool))
    e,_=R.chain_event(before,after,info,dict(terminal=j.zeros(2,bool)),dict(birth=j.ones_like(fuse)),j.array([0,1]))
    np.testing.assert_array_equal(e['placed'],[True,True]);np.testing.assert_array_equal(e['bad'],[False,False])
    np.testing.assert_array_equal(e['connected_uncredited'],[True,True])
