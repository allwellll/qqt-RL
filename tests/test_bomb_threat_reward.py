import jax
import jax.numpy as j
import numpy as np
from scripts import bomb_threat_reward as R

def event(**kw):
    e=dict(active=j.array([True]),bad=j.array([False]),terminal=j.array([False]),placed=j.array([True]),
        safe=j.array([True]),threat=j.array([True]),chain=j.array([False]),old_id=j.array([4]),old_cell=j.array([6]),
        enemy_cell=j.array([5]),owned_removed=j.zeros((1,13,15),bool),improved=j.array([False]),settled=j.array([False]))
    e.update(kw);return e

def test_first_cover_delayed_dedup_and_cap():
    s,r,m=jax.jit(R.settle)(event(),R.state(1));assert float(r[0])==0
    for _ in range(20):s,r,m=R.settle(event(),s)
    s,r,m=R.settle(event(settled=j.array([True])),s);np.testing.assert_allclose(r,.02)
    for cell in range(50):s,r,_=R.settle(event(enemy_cell=j.array([cell]),settled=j.array([True])),s)
    assert float(s['gross'][0])<=R.CAP+1e-7

def test_unsafe_self_harm_idle_and_terminal_no_pay_or_refund():
    for change in [dict(safe=j.array([False])),dict(placed=j.array([False])),dict(bad=j.array([True])),dict(terminal=j.array([True]))]:
        _,r,_=R.settle(event(settled=j.array([True]),**change),R.state(1));assert float(r[0])==0
    s,_,_=R.settle(event(),R.state(1));s,r,_=R.settle(event(bad=j.array([True]),settled=j.array([True])),s);assert float(r[0])==0
    s,r,_=R.settle(event(settled=j.array([True])),s);assert float(r[0])==0

def test_topology_requires_owned_real_unique_removed_cell_and_improvement():
    brick=j.zeros((1,13,15),bool).at[0,3,4].set(True)
    for e in [event(),event(owned_removed=brick),event(improved=j.array([True]))]:
        _,r,_=R.settle(dict(e,placed=j.array([False]),settled=j.array([True])),R.state(1));assert float(r[0])==0
    e=event(placed=j.array([False]),owned_removed=brick,improved=j.array([True]),settled=j.array([True]))
    s,r,m=R.settle(e,R.state(1));np.testing.assert_allclose(r,.04);assert int(m['brick_events'][0])==1
    _,r,_=R.settle(e,s);assert float(r[0])==0

def test_temporal_old_event_reuse_rejected_and_live_inventory_pays_nothing():
    e=event(threat=j.array([False]),chain=j.array([True]),settled=j.array([True]))
    s,r,_=R.settle(e,R.state(1));np.testing.assert_allclose(r,.01)
    _,r,_=R.settle(e,s);assert float(r[0])==0
    _,r,_=R.settle(dict(e,placed=j.array([False])),R.state(1));assert float(r[0])==0

def test_exact_distance_detour_and_disconnected_are_not_dig_distance():
    b=j.zeros((1,13,15),bool).at[0,:,5].set(True);start=j.array([[6,2]]);goal=j.array([[6,8]])
    assert int(R.distance(b,goal,start)[0])==1000
    opened=b.at[0,1,5].set(False);assert int(R.distance(opened,goal,start)[0])==16
