"""Conservative carrier credit, isolated from the live R2/R3 recipe."""
import numpy as np
import jax
import jax.numpy as j

from scripts import r4_k2_carrier_reward as r


def test_new_carrier_range_loose_and_full_loop_are_conservative():
    q=j.array([0.,.25,.5,.75,1.])
    home=r.graded_phi(True,False,False,0.,q)
    carry=r.graded_phi(False,True,False,0.,q)
    loose=r.graded_phi(False,False,True,q,q)
    np.testing.assert_allclose(carry,-.4-.8*np.asarray(q),atol=1e-7)
    np.testing.assert_allclose(loose,-.1,atol=1e-7)
    interception=r.delta(carry,loose)
    np.testing.assert_allclose(interception,.3+.8*np.asarray(q),atol=1e-7)
    total=r.delta(home,carry)+interception+r.delta(loose,home)
    np.testing.assert_allclose(total,0.,atol=1e-6)


def event(s,**overrides):
    args=dict(forced=j.array([False]),source_batch=j.array([7]),
        enemy_carrying_own=j.array([True]),eligible=j.array([True]),
        causal_kill=j.array([False]),kill_batch=j.array([7]),
        enemy_died=j.array([False]),terminal=j.array([False]))
    args.update({k:j.atleast_1d(v) for k,v in overrides.items()})
    return r.advance(s,**args)


def test_forced_advance_is_deducted_from_same_batch_kill():
    s,p,m=event(r.state(1),forced=True)
    np.testing.assert_allclose(p,.1)
    s,p2,m=event(s,causal_kill=True,enemy_died=True)
    np.testing.assert_allclose(p+p2,0.,atol=1e-7)
    np.testing.assert_allclose(m['settled_advance'],.1)
    # The custody delta, including the earlier pickup debit, is the only net
    # carrier interception credit. No additional fixed kill payment survives.
    home,carry,loose=0.,-1.2,-.1
    total=r.delta(home,carry)+p+r.delta(carry,loose)+p2+r.delta(loose,home)
    np.testing.assert_allclose(total,0.,atol=1e-6)


def test_timeout_wrong_source_terminal_and_package_loss_refund():
    for ending in [dict(causal_kill=True,kill_batch=8,enemy_died=True),
                   dict(terminal=True),dict(enemy_carrying_own=False)]:
        s,p,_=event(r.state(1),forced=True)
        s,q,m=event(s,**ending)
        np.testing.assert_allclose(p+q,0.,atol=1e-7)
        np.testing.assert_allclose(m['clawback'],.1)
    s,total,_=event(r.state(1),forced=True)
    for _ in range(20):
        s,p,_=event(s);total=total+p
    np.testing.assert_allclose(total,0.,atol=1e-7)


def test_once_per_life_no_stack_and_no_illegal_request_payment():
    s,p,_=event(r.state(1),forced=True)
    s,q,_=event(s,forced=True)
    np.testing.assert_allclose(q,0.)
    s,q,_=event(s,enemy_carrying_own=False)
    s,q,_=event(s,forced=True)
    np.testing.assert_allclose(q,0.)
    # A real death starts a new life; merely dropping/re-taking a bun does not.
    s,q,_=event(s,enemy_died=True)
    s,q,_=event(s,forced=True)
    np.testing.assert_allclose(q,.1)
    for bad in [dict(eligible=False),dict(enemy_carrying_own=False),dict(terminal=True)]:
        _,q,_=event(r.state(1),forced=True,**bad)
        np.testing.assert_allclose(q,0.)


def test_terminal_never_refunds_failed_defense_and_zero_switch_exact():
    np.testing.assert_allclose(r.delta(-1.2,0.,True),0.)
    x=j.linspace(0.,1.,11)
    from scripts.bun_native_v2_custody_reward import graded_phi as old
    for home,carry,loose in [(True,False,False),(False,True,False),(False,False,True),(False,False,False)]:
        np.testing.assert_array_equal(r.graded_phi(home,carry,loose,x,x,enabled=False),old(home,carry,loose,x,x))


def test_jit_batch_and_full_pending_state_restore():
    f=jax.jit(r.advance)
    s=r.state(2)
    args=dict(forced=j.array([True,True]),source_batch=j.array([1,2]),enemy_carrying_own=j.array([True,False]),
              eligible=j.array([True,True]),causal_kill=j.array([False,False]),kill_batch=j.array([-1,-1]),
              enemy_died=j.array([False,False]),terminal=j.array([False,False]))
    saved,p,_=f(s,**args)
    restored=jax.tree.map(lambda x:j.asarray(np.array(x)),saved)
    a=f(saved,**args);b=f(restored,**args)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
