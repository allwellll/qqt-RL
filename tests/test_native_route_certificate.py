"""Proposal-only RED cases from fixed production Review one-step states."""
import numpy as np
import jax.numpy as jnp
from scripts.native_route_certificate import certify_set, certify_native_positions


def test_fixed_g8_drop_excludes_south_that_reverses_real_progress():
    target,valid=certify_set(jnp.array([[0.,.5,.5,0.,0.]]),jnp.ones((1,5),bool),
        jnp.array([30.83]),jnp.array([[32.,31.22,30.44,31.22,30.83]]),jnp.array([False]))
    np.testing.assert_allclose(target,[[0,0,1,0,0]])
    assert bool(valid[0])


def test_fixed_parent_step_prefers_actual_gain_point39_over_point05():
    target,_=certify_set(jnp.array([[0.,.5,.5,0.,0.]]),jnp.ones((1,5),bool),
        jnp.array([31.61]),jnp.array([[31.61,31.56,31.22,32.,31.61]]),jnp.array([False]))
    np.testing.assert_allclose(target,[[0,0,1,0,0]])


def test_equally_good_native_steps_remain_set_valued_and_unsafe_step_excluded():
    proposed=jnp.array([[0.,.5,.5,0.,0.]])
    after=jnp.array([[9.,8.61,8.61,9.,9.]])
    target,_=certify_set(proposed,jnp.ones((1,5),bool),jnp.array([9.]),after,jnp.array([False]))
    np.testing.assert_allclose(target,proposed)
    target,_=certify_set(proposed,jnp.ones((1,5),bool).at[0,1].set(False),jnp.array([9.]),after,jnp.array([False]))
    np.testing.assert_allclose(target,[[0,0,1,0,0]])


def test_goal_wait_is_valid_but_unreachable_or_nonprogress_has_no_label():
    proposed=jnp.array([[0,0,0,0,1.],[0,.5,.5,0,0],[0,.5,.5,0,0]])
    target,valid=certify_set(proposed,jnp.ones((3,5),bool),jnp.array([0.,32767.,8.]),
        jnp.array([[0,0,0,0,0],[32767.]*5,[9.,8.,8.2,9.,8.]]),jnp.array([True,False,False]))
    np.testing.assert_array_equal(valid,[True,False,False])
    np.testing.assert_allclose(target[0],proposed[0]);np.testing.assert_allclose(target[1:],0)


def test_all_legal_native_endpoints_recover_omitted_equal_progress_direction():
    # A .18 geometric proposal need not include all equally good real steps.
    y,x=jnp.indices((13,15));distance=(abs(10-y)+abs(10-x))[None]
    before=jnp.array([[5.5,5.5]])
    endpoints=before[:,None]+jnp.array([[[0.,0.],[.39,0.],[0.,0.],[0.,.39],[0.,0.]]])
    target,valid=certify_native_positions(before,endpoints,distance,jnp.zeros_like(distance,bool),
        jnp.ones((1,5),bool),jnp.ones((1,5),bool),jnp.array([False]))
    np.testing.assert_allclose(target,[[0,.5,0,.5,0]])
    assert bool(valid[0])


def test_native_endpoint_certificate_rejects_topology_change_and_blocked_path():
    y,_=jnp.indices((13,15));distance=(12-y)[None]
    before=jnp.array([[5.5,5.5]]);endpoints=jnp.broadcast_to(before[:,None],(1,5,2)).at[0,1,0].add(.39)
    args=(before,endpoints,distance,jnp.zeros_like(distance,bool),jnp.ones((1,5),bool),jnp.ones((1,5),bool))
    _,valid=certify_native_positions(*args,jnp.array([False]),topology_stable=jnp.array([[True,False,True,True,True]]))
    assert not bool(valid[0])
    _,valid=certify_native_positions(before,endpoints,jnp.full_like(distance,32767),args[3],args[4],args[5],jnp.array([False]))
    assert not bool(valid[0])


def test_fixed_review_g8_conflict_reproduces_in_jax_native_steering():
    import json
    from collections import deque
    from pathlib import Path
    import jax
    from jax_bomb import je05_core_env as core
    from scripts.je10_route_oracle import continuous_distance
    level=next(x for x in json.loads(Path('web/assets/maps/levels.json').read_text()) if x['qqt_id']==806)
    blocked=np.asarray(level['wall'],bool).reshape(13,15)
    distance=np.full((13,15),32767,np.int32);distance[5,9]=0;queue=deque([(5,9)])
    while queue:
        row,column=queue.popleft()
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            y,x=row+dy,column+dx
            if 0<=y<13 and 0<=x<15 and not blocked[y,x] and distance[y,x]>distance[row,column]+1:
                distance[y,x]=distance[row,column]+1;queue.append((y,x))
    # Same map, open bricks, position and native speed as fixed Review g8.
    pos=jnp.array([[6.28,5.11]]);blocked=jnp.asarray(blocked)[None];distance=jnp.asarray(distance)[None]
    endpoints=jax.jit(jax.vmap(lambda move:core._steer(pos[0],move,True,blocked[0],1.3)))(jnp.arange(5))[None]
    legal=jnp.ones((1,5),bool);safe=jnp.ones((1,5),bool)
    before=continuous_distance(pos,distance,blocked)
    after=continuous_distance(endpoints[0],jnp.repeat(distance,5,0),jnp.repeat(blocked,5,0))
    # The shared field has since changed. Verify the native certificate against
    # its current gains; the historical .18-step conflict remains in evidence.
    gains=before[0]-after
    target,valid=certify_native_positions(pos,endpoints,distance,blocked,legal,safe,jnp.array([False]))
    np.testing.assert_allclose(target,[[0,0,1,0,0]])
    assert bool(valid[0])
    assert float(gains[2])>0 and float(gains[2])>=float(gains.max())-1e-5
