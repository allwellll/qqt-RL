from types import SimpleNamespace
import numpy as np


def fixture(n=4):
    import jax.numpy as j
    from scripts.r3_selfplay_monitor import state
    st=SimpleNamespace(t=j.zeros(n),pos=j.tile(j.array([[[4.5,2.5],[4.5,12.5]]]),(n,1,1)),
        alive=j.ones((n,2),bool),bun_carried=j.full((n,2),-1,j.int32),
        bun_stored=j.broadcast_to(j.array([[2,0],[0,2]]),(n,2,2)))
    aux=dict(terminal=j.zeros(n,bool),provenance_source=j.array([-1,-1,42,-1])[:n],
        combat_mix=dict(sp_monitor=state(n),kind=j.array([3,4,5,5])[:n],scene=j.array([False,False,False,True])[:n],
            learner_seat=j.arange(n)%2))
    info=dict(bomb_placed=j.zeros((n,2),bool),surviving_kill=j.zeros((n,2),bool),
        h_route_terminal=j.ones(n,bool),winner=j.array([0,-1,1,1])[:n],capture=j.zeros((n,2),bool))
    return st,aux,info


def test_native_denominators_exclude_bank_clear_and_terminal_tail():
    import jax.numpy as j
    from scripts.r3_selfplay_monitor import advance
    st,aux,info=fixture();_,m=advance(st,st,info,aux)
    assert all(np.asarray(x).shape==(4,3,2) for x in m.values())
    assert np.asarray(m['r3_sp_episodes']).sum()==2
    assert np.asarray(m['r3_sp_draws']).sum()==1
    assert np.asarray(m['r3_sp_wins']).sum()==1
    assert np.asarray(m['r3_sp_passive']).sum()==2
    _,tail=advance(st,st,info,dict(aux,terminal=j.ones(4,bool)))
    assert all(np.asarray(x).sum()==0 for x in tail.values())


def test_unknown_winners_never_count_as_draws():
    import jax.numpy as j
    from scripts.r3_selfplay_monitor import advance
    st,aux,info=fixture();info=dict(info,winner=j.full(4,-2))
    _,m=advance(st,st,info,aux)
    assert np.asarray(m['r3_sp_draws']).sum()==0
    assert np.asarray(m['r3_sp_unresolved']).sum()==2
    aux['combat_mix']['scene']=j.zeros(4,bool)
    aux['provenance_source']=j.full(4,-1)
    _,m=advance(st,st,info,aux)
    assert np.asarray(m['r3_sp_fixed_draws']).sum()==0
    assert np.asarray(m['r3_sp_fixed_unresolved']).sum()==2


def test_recovery_cycle_requires_actual_home_inventory_increase_and_no_double_count():
    import jax.numpy as j
    from scripts.r3_selfplay_monitor import advance
    st,aux,info=fixture();st.bun_carried=st.bun_carried.at[0,1].set(0)
    info=dict(info,surviving_kill=info['surviving_kill'].at[0,0].set(True))
    monitor,_=advance(st,st,info,aux)
    assert np.asarray(monitor['loops']).sum()==0  # another same-origin pack was already home
    after=SimpleNamespace(**vars(st));after.bun_stored=st.bun_stored.at[0,0,0].add(1)
    aux=dict(aux,combat_mix=dict(aux['combat_mix'],sp_monitor=monitor))
    info=dict(info,surviving_kill=j.zeros((4,2),bool))
    returned,_=advance(st,after,info,aux)
    assert returned['loops'][0,0]==1
    aux=dict(aux,combat_mix=dict(aux['combat_mix'],sp_monitor=returned))
    final,_=advance(after,after,info,aux)
    assert final['loops'][0,0]==1
