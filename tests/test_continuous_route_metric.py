"""Fixed Review exposes a floor-cell boundary discontinuity in the old metric."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def fixed_geometry():
    from scripts.je08_bun_potential import cpu_shortest
    level=next(x for x in json.loads(Path('web/assets/maps/levels.json').read_text()) if x['qqt_id']==806)
    hard=np.asarray(level['wall'],bool).reshape(13,15)
    goal=np.zeros_like(hard)
    row,column=level['bun_bases'][0];goal[row:row+3,column:column+3]=True
    return hard,cpu_shortest(hard,np.zeros_like(hard),goal)


def test_fixed_review_boundary_limit_is_continuous_in_current_oracle_and_metric():
    from scripts import continuous_route_metric as metric
    from scripts.je10_route_oracle import continuous_distance as oracle_distance
    hard,distance=fixed_geometry()
    pos=jnp.array([[5.99999,9.82],[6.00001,9.82]])
    maps=jnp.broadcast_to(distance,(2,13,15));blocked=jnp.broadcast_to(hard,(2,13,15))
    value=jax.jit(metric.continuous_distance)(pos,maps,blocked)
    np.testing.assert_array_equal(value,oracle_distance(pos,maps,blocked))
    assert abs(float(value[0]-value[1]))<.0001


def test_shared_open_edges_are_continuous_at_multiple_subcell_offsets_on_native_maps():
    from scripts import continuous_route_metric as metric
    hard,distance=fixed_geometry();positions=[]
    for row in range(1,12):
        for column in range(1,14):
            for axis in [0,1]:
                other=(row+axis,column+1-axis)
                if hard[row,column] or hard[other] or distance[row,column]>=32767:continue
                for offset in [.11,.37,.51,.89]:
                    point=[row+offset,column+1.] if axis==0 else [row+1.,column+offset]
                    left=point.copy();right=point.copy();left[1-axis]-=1e-5;right[1-axis]+=1e-5
                    positions.extend([left,right])
    n=len(positions)
    values=jax.jit(metric.continuous_distance)(jnp.asarray(positions),
        jnp.broadcast_to(distance,(n,13,15)),jnp.broadcast_to(hard,(n,13,15)))
    difference=np.asarray(values).reshape(-1,2)
    assert np.isfinite(difference).all()
    assert np.abs(difference[:,0]-difference[:,1]).max()<.0002


def test_cell_centers_preserve_grid_values_and_goal_center_is_zero():
    from scripts import continuous_route_metric as metric
    hard,distance=fixed_geometry();cells=np.argwhere(~hard&(distance<32767));n=len(cells)
    values=metric.continuous_distance(jnp.asarray(cells+.5),
        jnp.broadcast_to(distance,(n,13,15)),jnp.broadcast_to(hard,(n,13,15)))
    np.testing.assert_allclose(values,distance[cells[:,0],cells[:,1]],atol=1e-5)
    goal=cells[distance[cells[:,0],cells[:,1]]==0][0]
    np.testing.assert_allclose(metric.continuous_distance(jnp.array([goal+[.5,.5]]),
        jnp.array(distance)[None],jnp.array(hard)[None]),0,atol=1e-5)


def test_diagonally_touching_components_and_infinite_maps_never_supply_fake_routes():
    from scripts import continuous_route_metric as metric
    hard=jnp.ones((1,3,3),bool).at[0,0,0].set(False).at[0,1,1].set(False)
    distance=jnp.full((1,3,3),32767).at[0,0,0].set(7).at[0,1,1].set(0)
    near=metric.continuous_distance(jnp.array([[.99,.99]]),distance,hard)
    np.testing.assert_allclose(near,[7.+.5*.98+.125*.98],atol=1e-6)
    no_diagonal=distance.at[0,1,1].set(32767)
    np.testing.assert_array_equal(near,metric.continuous_distance(jnp.array([[.99,.99]]),no_diagonal,hard))
    assert float(metric.continuous_distance(jnp.array([[.5,.5]]),jnp.full_like(distance,32767),hard)[0])==32767
    assert float(metric.continuous_distance(jnp.array([[2.5,2.5]]),distance,hard)[0])==32767


def test_native_corridor_progress_does_not_flatten_at_center_or_near_center():
    from scripts import continuous_route_metric as metric
    from jax_bomb import je05_core_env as core
    hard,distance=fixed_geometry()
    positions=np.array([cell+[dy,dx] for cell in np.argwhere(~hard&(distance>0)&(distance<32767))
        for dy in [.38,.5,.62] for dx in [.38,.5,.62]])
    pos=jnp.array(positions);n=len(pos)
    ends=jax.jit(jax.vmap(lambda p:jax.vmap(lambda move:
        core._steer(p,move,True,jnp.array(hard),1.))(jnp.arange(5))))(pos)
    maps=jnp.broadcast_to(distance,(n,13,15));blocked=jnp.broadcast_to(hard,(n,13,15))
    before=metric.continuous_distance(pos,maps,blocked)
    after=metric.continuous_distance(ends.reshape(-1,2),jnp.repeat(maps,5,0),jnp.repeat(blocked,5,0)).reshape(n,5)
    gain=np.asarray(before[:,None]-after)
    assert (gain.max(-1)>1e-5).all(),positions[gain.max(-1)<=1e-5][:8].tolist()


def test_native_steering_alignment_has_positive_signal_at_off_center_corner():
    from scripts import continuous_route_metric as metric
    from jax_bomb import je05_core_env as core
    hard,distance=fixed_geometry();positions=jnp.array([[.5,5.25],[.5,5.86],[.38,5.14]])
    ends=jax.jit(jax.vmap(lambda p:jax.vmap(lambda move:
        core._steer(p,move,True,jnp.array(hard),1.))(jnp.arange(5))))(positions)
    before=metric.continuous_distance(positions,jnp.broadcast_to(distance,(3,13,15)),jnp.broadcast_to(hard,(3,13,15)))
    after=metric.continuous_distance(ends.reshape(-1,2),jnp.broadcast_to(distance,(15,13,15)),jnp.broadcast_to(hard,(15,13,15))).reshape(3,5)
    assert np.asarray(before[:,None]-after).max(-1).min()>1e-5
