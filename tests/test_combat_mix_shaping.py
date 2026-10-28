import jax.numpy as j
import numpy as np

from scripts.combat_mix_shaping import state, settle


def event(**changes):
    values = dict(placed=j.array([True]),connected_uncredited=j.array([True]),
        newly_threatens=j.array([True]),safe_placement=j.array([True]),
        enemy_cell=j.array([25]),settled=j.array([False]),bad=j.array([False]),
        terminal=j.array([False]),active=j.array([True]))
    return dict(values, **changes)


def test_no_live_inventory_tick_payment_and_only_safe_settlement_pays_once():
    ledger=state(1)
    ledger,reward,metrics=settle(event(),ledger,.1)
    assert float(reward[0])==0
    for _ in range(10):
        ledger,reward,metrics=settle(event(placed=j.array([False])),ledger,.1)
        assert float(reward[0])==0
    ledger,reward,metrics=settle(event(placed=j.array([False]),settled=j.array([True])),ledger,.1)
    assert np.isclose(float(reward[0]),.1)
    ledger,reward,metrics=settle(event(placed=j.array([False]),settled=j.array([True])),ledger,.1)
    assert float(reward[0])==0


def test_no_duplicate_same_enemy_cell_and_no_unconnected_or_unsafe_payment():
    for changes in ({'connected_uncredited':j.array([False])},{'safe_placement':j.array([False])},{'newly_threatens':j.array([False])}):
        ledger,reward,_=settle(event(settled=j.array([True]),**changes),state(1),.1)
        assert float(reward[0])==0
    ledger,reward,_=settle(event(settled=j.array([True])),state(1),.1)
    assert np.isclose(float(reward[0]),.1)
    ledger,reward,_=settle(event(settled=j.array([True])),ledger,.1)
    assert float(reward[0])==0


def test_death_trap_cancels_pending_and_zero_coefficient_is_exact():
    ledger,_,_=settle(event(),state(1),.1)
    ledger,reward,_=settle(event(placed=j.array([False]),bad=j.array([True]),settled=j.array([True])),ledger,.1)
    assert float(reward[0])==0 and float(ledger['pending'][0])==0
    _,reward,_=settle(event(settled=j.array([True])),state(1),0.)
    assert float(reward[0])==0


def test_consumed_old_bomb_cannot_pay_again_for_a_new_enemy_cell_and_bad_credit_not_refunded():
    ledger,_,_=settle(event(settled=j.array([True])),state(1),.1)
    _,reward,_=settle(event(enemy_cell=j.array([26]),settled=j.array([True])),ledger,.1)
    assert float(reward[0])==0
    ledger,_,_=settle(event(),state(1),.1)
    ledger,_,_=settle(event(bad=j.array([True])),ledger,.1)
    _,reward,_=settle(event(settled=j.array([True])),ledger,.1)
    assert float(reward[0])==0


def test_terminal_never_pays_and_episode_cap_is_not_bypassable_by_repeated_cells():
    ledger=state(1)
    _,reward,_=settle(event(settled=j.array([True]),terminal=j.array([True])),ledger,.1)
    assert float(reward[0])==0
    total=0
    for i in range(12):
        ledger,reward,_=settle(event(old_cell=j.array([i]),old_id=j.array([i]),
            enemy_cell=j.array([i]),settled=j.array([True])),ledger,.1)
        total+=float(reward[0])
    assert np.isclose(total,.5)
    assert int(ledger['gross'][0])==5


def test_reset_has_no_credit_and_separate_environments_never_share_event_identity():
    fresh=state(2)
    assert np.all(np.asarray(fresh['pending'])==0)
    assert np.all(np.asarray(fresh['old_seen'])==-1)


def test_pending_credit_remains_per_environment_and_multiple_placement_events_do_not_tick_pay():
    n=2;s=state(n)
    e=dict(placed=j.array([True,True]),connected_uncredited=j.array([True,True]),
      newly_threatens=j.array([True,True]),safe_placement=j.array([True,True]),enemy_cell=j.array([25,25]),
      old_cell=j.array([3,3]),old_id=j.array([10,10]),settled=j.array([False,False]),
      bad=j.array([False,True]),terminal=j.array([False,False]),active=j.array([True,True]))
    s,r,_=settle(e,s,.1)
    assert np.array_equal(s['pending'],[1,0]) and np.array_equal(r,[0,0])
    e.update(placed=j.array([False,False]),bad=j.array([False,False]),settled=j.array([True,True]))
    s,r,_=settle(e,s,.1)
    assert np.allclose(r,[.1,0])
    assert np.array_equal(s['gross'],[1,0])
