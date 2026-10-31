import jax
import jax.numpy as j
import numpy as np
from scripts import bomb_threat_reward as R
from scripts.bomb_threat_signal_trace import audited_settle


def event(**kw):
    e=dict(active=j.array([True]),bad=j.array([False]),terminal=j.array([False]),placed=j.array([True]),
        safe=j.array([True]),threat=j.array([True]),chain=j.array([False]),old_id=j.array([4]),old_cell=j.array([6]),
        enemy_cell=j.array([5]),owned_removed=j.zeros((1,13,15),bool),improved=j.array([False]),settled=j.array([False]))
    e.update(kw);return e


def test_passive_audit_preserves_reward_and_state_and_balances_pending():
    f=audited_settle(R.settle);state=R.state(1);e=event()
    old=R.settle(e,state);new=jax.jit(f)(e,state)
    for a,b in zip(jax.tree.leaves(old[:2]),jax.tree.leaves(new[:2])):np.testing.assert_array_equal(a,b)
    assert set(new[2])==set(old[2])|{'audit'}
    np.testing.assert_allclose(new[2]['audit']['admitted'],[.02])
    np.testing.assert_allclose(new[2]['audit']['balance_residual'],[0],atol=1e-7)
    cancelled=f(event(bad=j.array([True])),new[0])
    np.testing.assert_allclose(cancelled[2]['audit']['cancel_bad'],[.02])
    np.testing.assert_allclose(cancelled[2]['audit']['balance_residual'],[0],atol=1e-7)
    paid=f(event(settled=j.array([True])),state)
    np.testing.assert_allclose(paid[2]['audit']['paid'],[.02])
    np.testing.assert_allclose(paid[2]['audit']['balance_residual'],[0],atol=1e-7)
