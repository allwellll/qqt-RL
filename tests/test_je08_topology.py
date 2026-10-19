"""Direction shaping must be topology based, bounded, recoverable and causal."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAXBOMB_RULE='bun')
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from scripts import je08_bun_potential as p


def event(before=10.,after=9.,**kw):
    e=dict(eligible=True,identity=1,before=before,after=after,boundary=False,bad=False,
           settled=True,pickup=False,capture=False,topology_changed=False)
    e.update(kw);return {k:jnp.asarray(v) for k,v in e.items()}


def started():
    r,s,m=p.phase_reward(event(),p.phase_state(),.05,2.)
    assert float(r)==0 and bool(m['phase_start'])
    return s


def test_signed_closer_farther_stationary_and_roundtrip():
    s=started()
    r,s,_=p.phase_reward(event(10,8),s,.05,2.);assert np.isclose(r,.1)
    r,s,_=p.phase_reward(event(8,10),s,.05,2.);assert np.isclose(r,-.1)
    r,s,m=p.phase_reward(event(10,10),s,.05,2.);assert r==0 and m['stationary']
    assert np.isclose(s['balance'],0)


def test_cap_clawback_roundtrip_cannot_profit_and_identity_not_switchable():
    s=started();total=0
    for _ in range(45):
        for a,b in [(10,9),(9,10)]:
            r,s,m=p.phase_reward(event(a,b),s,.05,2.);total+=float(r)
    assert total<=1e-6 and float(s['gross'])<=2.00001
    r,s,m=p.phase_reward(event(identity=2),s,.05,2.)
    assert r==0 and m['phase_start'] and m['phase_end'] and s['identity']==2
    s=dict(s,balance=jnp.float32(0))
    r,s,_=p.phase_reward(event(10,8,identity=2),s,.05,2.)
    r,s,_=p.phase_reward(event(8,7,identity=2,bad=True,boundary=True),s,.05,2.)
    assert np.isclose(r,-.1) and not s['active']


def test_boundary_variant_restore_and_unreachable():
    s=started()
    for kw in [dict(boundary=True,pickup=True),dict(boundary=True,capture=True),dict(topology_changed=True),dict(before=p.INF)]:
        r,_,_=p.phase_reward(event(**kw),s,.05,2.);assert r==0
    snapshot=pickle.loads(pickle.dumps(jax.device_get(s)))
    a=p.phase_reward(event(),snapshot,.05,2.);b=jax.jit(lambda e,s:p.phase_reward(e,s,.05,2.))(event(),s)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_allclose(x,y)


def test_exact_weighted_topology_walls_bricks_and_detour():
    wall=np.zeros((5,6),bool);wall[:4,2]=True
    brick=np.zeros_like(wall);brick[4,2]=True
    goals=np.zeros_like(wall);goals[0,4]=True
    a=np.asarray(p.shortest(jnp.asarray(wall),jnp.asarray(brick),jnp.asarray(goals)))
    reference=p.cpu_shortest(wall,brick,goals)
    np.testing.assert_array_equal(a,reference)
    # Move DOWN goes away in Manhattan but closer in the legal topology.
    assert a[1,1]<a[0,1] and a[0,2]==p.INF
    opened=p.cpu_shortest(wall,np.zeros_like(wall),goals)
    assert a[0,1]==opened[0,1]+3
