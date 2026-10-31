"""Event settlement and pure on-policy reward wiring, CPU-only fixtures."""
import jax
import jax.numpy as j
import numpy as np
from scripts import r4_k2_collector as K


def event(s,**overrides):
    d=dict(forced=False,batch=7,kill_batch=-1,causal_kill=False,
        enemy_carry=False,enemy_died=False,eligible=True,terminal=False)
    d.update(overrides)
    return K.settle(s,**{k:j.atleast_1d(v) for k,v in d.items()})


def test_noncarrier_real_kill_and_same_batch_once():
    s,p,m=event(K.state(1),forced=True)
    np.testing.assert_array_equal(p,0)
    s,p,m=event(s,causal_kill=True,enemy_died=True,kill_batch=7)
    np.testing.assert_allclose(p,1/3+1/15)
    assert bool(m['r4_k2_forced_realized'][0])
    _,p,_=event(s)
    np.testing.assert_array_equal(p,0)


def test_wrong_source_spawn_protection_timeout_and_request_never_force_pay():
    for settings in [dict(kill_batch=8),dict(eligible=False),dict(kill_batch=-1)]:
        s,_,_=event(K.state(1),forced=True)
        _,p,m=event(s,causal_kill=True,enemy_died=True,**settings)
        np.testing.assert_allclose(p,0 if settings.get('eligible') is False else 1/3)
        assert not bool(m['r4_k2_forced_realized'][0])
    s,_,_=event(K.state(1),forced=True)
    for _ in range(21):s,_,_=event(s)
    _,p,_=event(s,causal_kill=True,enemy_died=True,kill_batch=7)
    np.testing.assert_allclose(p,1/3)


def test_carrier_full_cycle_is_conservative_and_first_terminal_flushes():
    s,p,_=event(K.state(1),forced=True,enemy_carry=True)
    s,q,_=event(s,causal_kill=True,enemy_died=True,kill_batch=7,enemy_carry=True)
    np.testing.assert_allclose(p+q,0,atol=1e-7)
    s,p,_=event(K.state(1),forced=True,enemy_carry=True)
    s,q,m=event(s,enemy_carry=True,terminal_close=True)
    np.testing.assert_allclose(p+q,0,atol=1e-7)
    np.testing.assert_allclose(s['advance']['pending'],0)
    assert float(m['r4_k2_clawback'][0])>0


def test_zero_coefficient_and_full_aux_restore_are_bit_exact():
    s,p,_=event(K.state(1),forced=True,enemy_carry=True)
    saved=jax.tree.map(lambda a:j.asarray(np.array(a)),s)
    kw=dict(forced=j.array([False]),batch=j.array([7]),kill_batch=j.array([7]),
        causal_kill=j.array([True]),enemy_carry=j.array([True]),enemy_died=j.array([True]),
        eligible=j.array([True]),terminal=j.array([False]))
    f=jax.jit(K.settle)
    a=f(s,**kw);b=f(saved,**kw)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
    _,zero,_=f(s,**kw,scale=0.)
    np.testing.assert_array_equal(zero,0)
